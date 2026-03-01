import argparse, os, datetime, gc, yaml
import logging
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples_custom, convert_adaround, pixart_alpha_aca_dict
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from transformers import AutoFeatureExtractor

logger = logging.getLogger(__name__)

os.environ['CURL_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

# === ADDED: GPTQ CALIBRATION LOGIC ===
def run_gptq_calibration(qnn, cali_xs, cali_ts, cali_cs, opt):
    from qdiff.gptq import GPTQ, Quantizer_GPTQ
    import gc
    logger.info("Starting Memory-Safe GPTQ Calibration...")
    
    # 1. Group target layers by their top-level block/module to prevent OOM
    # PixArt typically uses 'blocks.0', 'blocks.1', etc.
    block_names = []
    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule) and not module.ignore_reconstruction:
            # Extract the root block name (e.g., 'blocks.0' or 'proj_in')
            root_block = ".".join(name.split(".")[:2]) if "blocks" in name else name.split(".")[0]
            if root_block not in block_names:
                block_names.append(root_block)

    # 2. Run GPTQ sequentially, block-by-block
    for block_name in block_names:
        logger.info(f"Running GPTQ on block group: {block_name}...")
        
        # Find all QuantModules inside this specific block
        subset_layers = {}
        for name, module in qnn.named_modules():
            if name.startswith(block_name) and isinstance(module, QuantModule) and not module.ignore_reconstruction:
                subset_layers[name] = module
                
        if not subset_layers:
            continue

        # Setup trackers for just this block
        gptq_trackers = {}
        handles = []
        
        for name, module in subset_layers.items():
            gptq_trackers[name] = GPTQ(module)
            gptq_trackers[name].quantizer = Quantizer_GPTQ()
            gptq_trackers[name].quantizer.configure(
                bits=opt.weight_bit, perchannel=True, sym=opt.w_sym, mse=False, 
                channel_group=1, clip_ratio=opt.w_clip_ratio, quant_type="int"
            )
            
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq_trackers[name].add_batch(inp[0].data, out.data)
                return tmp
            handles.append(module.register_forward_hook(add_batch(name)))

        # Forward pass to populate Hessians for this block ONLY
        qnn.eval()
        with torch.no_grad():
            for i in range(0, cali_xs.size(0), 2): # Batch size of 2 to keep activation memory low
                qnn(cali_xs[i:i+2].cuda(), 
                    timestep=cali_ts[i:i+2].cuda(), 
                    encoder_hidden_states=cali_cs[i:i+2].cuda(), 
                    added_cond_kwargs=pixart_alpha_aca_dict(cali_xs[i:i+2]))

        # Remove hooks
        for h in handles:
            h.remove()

        # Execute GPTQ update and free memory
        for name, module in subset_layers.items():
            gptq_trackers[name].fasterquant(percdamp=opt.percdamp, groupsize=opt.weight_group_size)
            gptq_trackers[name].free()
            
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("GPTQ Calibration Complete!")
# =====================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None, help="the prompt to render")
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/txt2img-samples")
    parser.add_argument("--skip_grid", action='store_true')
    parser.add_argument("--skip_save", action='store_true')
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--plms", action='store_true')
    parser.add_argument("--laion400m", action='store_true')
    parser.add_argument("--fixed_code", action='store_true')
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_rows", type=int, default=0)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--from-file", type=str)
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast")
    
    parser.add_argument("--ptq", action="store_true")
    parser.add_argument("--quant_act", action="store_true")
    parser.add_argument("--weight_bit", type=int, default=8)
    parser.add_argument("--act_bit", type=int, default=8)
    parser.add_argument("--quant_mode", type=str, default="symmetric", choices=["linear", "squant", "qdiff"])
    parser.add_argument("--cali_st", type=int, default=20)
    parser.add_argument("--cali_batch_size", type=int, default=8)
    parser.add_argument("--cali_n", type=int, default=128)
    parser.add_argument("--cali_iters", type=int, default=20000)
    parser.add_argument('--cali_iters_a', default=5000, type=int)
    parser.add_argument('--cali_lr', default=4e-4, type=float)
    parser.add_argument('--cali_p', default=2.4, type=float)
    parser.add_argument("--cali_ckpt", type=str)
    parser.add_argument("--cali_data_path", type=str, default="pixart_calib_brecq.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adaround", action="store_true")
    parser.add_argument("--resume_w", action="store_true")
    parser.add_argument("--cond", action="store_true")
    parser.add_argument("--no_grad_ckpt", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--running_stat", action="store_true")
    parser.add_argument("--rs_sm_only", action="store_true")
    parser.add_argument("--sm_abit",type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--refiner", action="store_true", default=False)
    parser.add_argument("--sequential_w", action="store_true", default=False)
    parser.add_argument("--sequential_a", action="store_true", default=False)
    parser.add_argument("--weight_mantissa_bits", type=int, default=None)
    parser.add_argument("--act_mantissa_bits", type=int, default=None)
    parser.add_argument("--no_adaround", action="store_true")
    parser.add_argument("--attn_weight_mantissa", type=int, default=None)
    parser.add_argument("--ff_weight_mantissa", type=int, default=None)
    parser.add_argument("--asym_softmax", action="store_true", default=False)
    parser.add_argument("--weight_group_size", type=int, default=128)
    parser.add_argument("--no_fp_biased_adaround", action="store_false")
    parser.add_argument("--disable_online_act_quant", action="store_true")
    parser.add_argument("--coco_9k", action="store_true", default=False)
    parser.add_argument("--coco_10k", action="store_true", default=False)
    parser.add_argument("--coco2014", action="store_true", default=False)
    parser.add_argument("--hpsv2", action="store_true", default=False)
    parser.add_argument("--pixart", action="store_true", default=False)
    parser.add_argument("--disable_fp_quant", action="store_true")
    parser.add_argument("--disable_group_quant", action="store_true")

    # === ADDED: Q-DIT CONFIGS ===
    parser.add_argument("--w_clip_ratio", type=float, default=1.0)
    parser.add_argument("--a_clip_ratio", type=float, default=1.0)
    parser.add_argument("--w_sym", action="store_true")
    parser.add_argument("--use_gptq", action="store_true")
    parser.add_argument("--percdamp", type=float, default=0.01)
    # ============================

    opt = parser.parse_args()
    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = os.path.join(opt.outdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(outpath)

    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    from diffusers import PixArtAlphaPipeline
    model = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16).to("cuda")

    from qdiff.caption_util import get_captions
    if opt.prompt is None:
        pes, pams, npe, npam = get_captions("alpha", model, coco_9k=opt.coco_9k, coco_10k=opt.coco_10k, coco2014=opt.coco2014, hpsv2=opt.hpsv2, pixart=opt.pixart)
        model.text_encoder = model.text_encoder.to("cpu")

    if opt.coco_9k: sp = "samples_9k"
    elif opt.coco_10k: sp = "samples_10k"
    elif opt.pixart: sp = "samples_pixart"
    elif opt.coco2014: sp = "samples_2014"
    elif opt.hpsv2: sp = 'samples_hpsv2'
    else: sp = "samples"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert(opt.cond)

    if opt.ptq:
        # === UPDATED: WQ AND AQ PARAMS WITH Q-DIT INTEGRATION ===
        wq_params = {
            'n_bits': opt.weight_bit, 
            'channel_wise': True, 
            'scale_method': 'mse', 
            'mantissa_bits': opt.weight_mantissa_bits,
            'attn_weight_mantissa': opt.attn_weight_mantissa,
            'ff_weight_mantissa': opt.ff_weight_mantissa,
            'weight_group_size': opt.weight_group_size,
            'fp_biased_adaround': opt.no_fp_biased_adaround,
            'group_quant': (not opt.disable_group_quant),
            'fp': (not opt.disable_fp_quant), 
            'sym': opt.w_sym,
            'clip_ratio': opt.w_clip_ratio
        }
        aq_params = {
            'n_bits': opt.act_bit, 
            'channel_wise': False, 
            'scale_method': 'mse', 
            'leaf_param': opt.quant_act, 
            'mantissa_bits': opt.act_mantissa_bits,
            'online_act_quant': (not opt.disable_online_act_quant),
            'fp': (not opt.disable_fp_quant),
            'sym': opt.asym_softmax,
            'clip_ratio': opt.a_clip_ratio
        }
        # ========================================================

        if opt.resume:
            logger.info('Load with min-max quick initialization')
            wq_params['scale_method'] = 'max'
            aq_params['scale_method'] = 'max'
        if opt.resume_w:
            wq_params['scale_method'] = 'max'

        qnn = QuantModel(model=model.transformer, weight_quant_params=wq_params, act_quant_params=aq_params, act_quant_mode="qdiff", sm_abit=opt.sm_abit)
        qnn.to("cuda")
        qnn.eval()

        if opt.no_grad_ckpt:
            logger.info('Not use gradient checkpointing for transformer blocks')
            qnn.set_grad_ckpt(False)

        if opt.resume:
            sample_data = torch.load(opt.cali_data_path)
            cali_data = get_train_samples_custom(opt, sample_data, opt.ddim_steps)
            resume_cali_model(qnn, opt.cali_ckpt, cali_data, opt.quant_act, "qdiff", cond=opt.cond)
        else:
            logger.info(f"Sampling data from {opt.cali_st} timesteps for calibration")
            sample_data = torch.load(opt.cali_data_path)
            sample_data['xs'] = sample_data['xs'].to(torch.float16)
            sample_data['cs'] = sample_data['cs'].to(torch.float16)
            cali_data = get_train_samples_custom(opt, sample_data, opt.ddim_steps)
            del(sample_data)
            gc.collect()

            cali_xs, cali_ts, cali_cs = cali_data
            if opt.resume_w:
                resume_cali_model(qnn, opt.cali_ckpt, cali_data, False, cond=opt.cond)
            else:
                logger.info("Initializing weight quantization parameters")

            qnn.set_quant_state(weight_quant=True, act_quant=False)
            _ = qnn(cali_xs[:2].cuda(), timestep=cali_ts[:2].cuda(), encoder_hidden_states=cali_cs[:2].cuda(), added_cond_kwargs = pixart_alpha_aca_dict(cali_xs[:2]))
            logger.info("Initializing has done!")

            kwargs = dict(cali_data=cali_data, batch_size=opt.cali_batch_size, iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2), warmup=0.2, act_quant=False, opt_mode='mse', cond=opt.cond, sequential=opt.sequential_w, no_adaround=opt.no_adaround)
        
            def recon_model(model):
                for name, module in model.named_children():
                    if isinstance(module, QuantModule):
                        if module.ignore_reconstruction is True: continue
                        else: layer_reconstruction(qnn, module, **kwargs)
                    elif isinstance(module, BaseQuantBlock):
                        if module.ignore_reconstruction is True: continue
                        else: block_reconstruction(qnn, module, **kwargs)
                    else:
                        recon_model(module)

            if not opt.resume_w:
                # === UPDATED: ROUTING LOGIC ===
                if opt.use_gptq:
                    run_gptq_calibration(qnn, cali_xs, cali_ts, cali_cs, opt)
                    # Tell native quantizers to just pass the modified weights through directly
                    for name, module in qnn.named_modules():
                        if isinstance(module, QuantModule):
                            module.weight_quantizer.scale_method = 'max' 
                    qnn.set_quant_state(weight_quant=True, act_quant=False)
                else:
                    logger.info("Doing AdaRound weight calibration")
                    recon_model(qnn)
                    qnn.set_quant_state(weight_quant=True, act_quant=False)
                # ==============================

                logger.info("Saving calibrated quantized UNet model")
                for m in qnn.model.modules():
                    if isinstance(m, AdaRoundQuantizer):
                        m.zero_point = nn.Parameter(m.zero_point)
                        m.delta = nn.Parameter(m.delta)
                    elif isinstance(m, UniformAffineQuantizer) and opt.quant_act:
                        if m.zero_point is not None:
                            if not torch.is_tensor(m.zero_point):
                                m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                            else:
                                m.zero_point = nn.Parameter(m.zero_point)
                torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt_wq.pth"))

            if opt.quant_act and opt.disable_online_act_quant:
                logger.info("Doing activation calibration")
                qnn.set_quant_state(True, True)
                with torch.no_grad():
                    inds = np.random.choice(cali_xs.shape[0], 16, replace=False)
                    _ = qnn(cali_xs[inds].cuda(), timestep=cali_ts[inds].cuda(), encoder_hidden_states=cali_cs[inds].cuda(), added_cond_kwargs = pixart_alpha_aca_dict(cali_xs[inds]))
                    if opt.running_stat:
                        logger.info('Running stat for activation quantization')
                        inds = np.arange(cali_xs.shape[0])
                        np.random.shuffle(inds)
                        qnn.set_running_stat(True, opt.rs_sm_only)
                        for i in trange(int(cali_xs.size(0) / 16)):
                            _ = qnn(cali_xs[inds[i * 16:(i + 1) * 16]].cuda(), timestep=cali_ts[inds[i * 16:(i + 1) * 16]].cuda(), encoder_hidden_states=cali_cs[inds[i * 16:(i + 1) * 16]].cuda(), added_cond_kwargs = pixart_alpha_aca_dict(cali_xs[inds[i * 16:(i + 1) * 16]]))
                        qnn.set_running_stat(False, opt.rs_sm_only)

                kwargs = dict(cali_data=cali_data, batch_size=opt.cali_batch_size, iters=opt.cali_iters_a, act_quant=True, opt_mode='mse', lr=opt.cali_lr, p=opt.cali_p, cond=opt.cond, sequential=opt.sequential_a)
                recon_model(qnn)
                qnn.set_quant_state(weight_quant=True, act_quant=True)
            elif opt.quant_act:
                logger.info("Doing online activation calibration")
                qnn.set_quant_state(weight_quant=True, act_quant=True)
            
            logger.info("Saving calibrated quantized UNet model")
            for m in qnn.model.modules():
                if isinstance(m, AdaRoundQuantizer):
                    m.zero_point = nn.Parameter(m.zero_point)
                    m.delta = nn.Parameter(m.delta)
                elif isinstance(m, UniformAffineQuantizer) and opt.quant_act and opt.disable_online_act_quant:
                    if m.zero_point is not None:
                        if not torch.is_tensor(m.zero_point):
                            m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                        else:
                            m.zero_point = nn.Parameter(m.zero_point)

        qnn = qnn.to('cuda', dtype=torch.float16)
        model.transformer = qnn
    
    sample_path = os.path.join(outpath, sp)
    os.makedirs(sample_path, exist_ok=True)

    sampling_file = os.path.join(outpath, "sampling_config.yaml")
    sampling_conf = vars(opt)
    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)

    batch_size = opt.n_samples
    if opt.prompt is None:
        for i in tqdm(range(0, pes.shape[0], batch_size), desc="data"):
            torch.manual_seed(42)
            prompt_embeds = pes[i:i + batch_size].to("cuda")
            image = model(prompt=None, negative_prompt=None,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_mask = pams[i:i + batch_size].to("cuda"),
                        negative_prompt_embeds = npe.expand(prompt_embeds.shape[0], -1, -1),
                        negative_prompt_attention_mask = npam.expand(prompt_embeds.shape[0], -1),
                        height=opt.res, width=opt.res).images
            for j, img in enumerate(image):
                img.save(os.path.join(sample_path, f"{i+j}.png"))
    else:
        torch.manual_seed(42)
        prompt = [opt.prompt]
        image = model(prompt=prompt, height=opt.res, width=opt.res).images
        for j, img in enumerate(image):
            img.save(os.path.join(sample_path, f"{j}.png"))

    logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")

if __name__ == "__main__":
    main()