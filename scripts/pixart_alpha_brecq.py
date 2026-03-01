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
from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples_custom, convert_adaround, pixart_alpha_aca_dict, save_inp_oup_data
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from transformers import AutoFeatureExtractor

logger = logging.getLogger(__name__)

# load safety model
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None, help="the prompt to render; if not provided, uses dataset captions")
    parser.add_argument("--max_prompts", type=int, default=None, help="limit number of caption prompts used for generation when --prompt is not set")
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/txt2img-samples")
    parser.add_argument("--skip_grid", action='store_true', help="do not save a grid, only individual samples. Helpful when evaluating lots of samples")
    parser.add_argument("--skip_save", action='store_true', help="do not save individual samples. For speed measurements.")
    parser.add_argument("--ddim_steps", type=int, default=20, help="number of ddim sampling steps")
    parser.add_argument("--plms", action='store_true', help="use plms sampling")
    parser.add_argument("--laion400m", action='store_true', help="uses the LAION400M model")
    parser.add_argument("--fixed_code", action='store_true', help="if enabled, uses the same starting code across samples ")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often")
    parser.add_argument("--res", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--C", type=int, default=4, help="latent channels")
    parser.add_argument("--f", type=int, default=8, help="downsampling factor")
    parser.add_argument("--n_samples", type=int, default=1, help="how many samples to produce for each given prompt. A.k.a. batch size")
    parser.add_argument("--n_rows", type=int, default=0, help="rows in the grid (default: n_samples)")
    parser.add_argument("--scale", type=float, default=7.5, help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    parser.add_argument("--from-file", type=str, help="if specified, load prompts from this file")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
    
    # linear quantization configs
    parser.add_argument("--ptq", action="store_true", help="apply post-training quantization")
    parser.add_argument("--quant_act", action="store_true", help="if to quantize activations when ptq==True")
    parser.add_argument("--weight_bit", type=int, default=8, help="int bit for weight quantization")
    parser.add_argument("--act_bit", type=int, default=8, help="int bit for activation quantization")
    parser.add_argument("--quant_mode", type=str, default="symmetric", choices=["linear", "squant", "qdiff"], help="quantization mode to use")

    # qdiff specific configs
    parser.add_argument("--cali_st", type=int, default=20, help="number of timesteps used for calibration")
    parser.add_argument("--cali_batch_size", type=int, default=8, help="batch size for qdiff reconstruction")
    parser.add_argument("--cali_n", type=int, default=128, help="number of samples for each timestep for qdiff reconstruction")
    parser.add_argument("--cali_iters", type=int, default=20000, help="number of iterations for each qdiff reconstruction")
    parser.add_argument('--cali_iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, help='L_p norm minimization for LSQ')
    parser.add_argument("--cali_ckpt", type=str, help="path for calibrated model ckpt")
    parser.add_argument("--cali_data_path", type=str, default="pixart_calib_brecq.pt", help="calibration dataset name")
    parser.add_argument("--resume", action="store_true", help="resume the calibrated qdiff model")
    parser.add_argument("--adaround", action="store_true", help="resume the calibrated qdiff model")
    parser.add_argument("--resume_w", action="store_true", help="resume the calibrated qdiff model weights only")
    parser.add_argument("--cond", action="store_true", help="whether to use conditional guidance")
    parser.add_argument("--no_grad_ckpt", action="store_true", help="disable gradient checkpointing")
    parser.add_argument("--split", action="store_true", help="use split strategy in skip connection")
    parser.add_argument("--running_stat", action="store_true", help="use running statistics for act quantizers")
    parser.add_argument("--rs_sm_only", action="store_true", help="use running statistics only for softmax act quantizers")
    parser.add_argument("--sm_abit",type=int, default=16, help="attn softmax activation bit")
    parser.add_argument("--verbose", action="store_true", help="print out info like quantized model arch")
    parser.add_argument("--refiner", action="store_true", default=False, help="use SDXL refiner")
    parser.add_argument("--sequential_w", action="store_true", default=False, help="Sequential weight quantization")
    parser.add_argument("--sequential_a", action="store_true", default=False, help="Sequential activation quantization")
    parser.add_argument("--weight_mantissa_bits", type=int, default=None, help="weight_mantissa bit")
    parser.add_argument("--act_mantissa_bits", type=int, default=None, help="weight_mantissa bit")
    parser.add_argument("--no_adaround", action="store_true", help="Disable adaround in weight quantization")
    parser.add_argument("--attn_weight_mantissa", type=int, default=None, help="mantissa bit for weight quantization")
    parser.add_argument("--ff_weight_mantissa", type=int, default=None, help="mantissa bit for feed forward weight")
    parser.add_argument("--asym_softmax", action="store_true", default=False, help="Use asymmetric quantization for attention's softmax and get an extra bits, default is using asymmetric")
    parser.add_argument("--weight_group_size", type=int, default=128, help="independent quantization to groups of <size> consecutive weights")
    parser.add_argument("--no_fp_biased_adaround", action="store_false", help="Disable FP scale awared adaround")
    parser.add_argument("--disable_online_act_quant", action="store_true", help="Disable online activation quantization")
    parser.add_argument("--coco_9k", action="store_true", default=False, help="generate images for coco val prompts 1000-9999")
    parser.add_argument("--coco_10k", action="store_true", default=False, help="generate images for coco val prompts 0-9999")
    parser.add_argument("--coco2014", action="store_true", default=False, help="generate images for coco 2014 prompts 0-9999")
    parser.add_argument("--hpsv2", action="store_true", default=False, help="generate images for coco 2014 prompts 0-9999")
    parser.add_argument("--pixart", action="store_true", default=False, help="generate images for 120 high-detailed pixart prompts (no ground truth)")
    parser.add_argument("--disable_fp_quant", action="store_true", help="Use integer quantization")
    parser.add_argument("--disable_group_quant", action="store_true", help="Disable group weight quantization")
    parser.add_argument("--weight_quant_method", type=str, default="mse", choices=["max", "mse"], help="scaling method for weight quantization")
    parser.add_argument("--act_quant_method", type=str, default="mse", choices=["max", "mse"], help="scaling method for activation quantization")
    
    # Q-DiT GPTQ Arguments
    parser.add_argument("--gptq", action="store_true", help="Use GPTQ for weight quantization")
    parser.add_argument("--gptq_percdamp", type=float, default=0.01, help="Percent damping in GPTQ")
    parser.add_argument("--gptq_groupsize", type=int, default=-1, help="Groupsize for GPTQ")
    parser.add_argument("--gptq_blocksize", type=int, default=128, help="Blocksize for GPTQ")
    parser.add_argument("--w_sym", action="store_true", help="Symmetric weight quantization (Q-DiT default).")
    parser.add_argument("--w_clip_ratio", type=float, default=1.0, help="Clip ratio for weight quantization.")

    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    run_tag = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    outpath = os.path.join(opt.outdir, run_tag)
    os.makedirs(outpath, exist_ok=True)

    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    from diffusers import PixArtAlphaPipeline
    model = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16
    ).to(device)

    from qdiff.caption_util import get_captions
    pes, pams, npe, npam = None, None, None, None
    if opt.prompt is None:
        pes, pams, npe, npam = get_captions("alpha", model, 
                            coco_9k=opt.coco_9k,
                            coco_10k=opt.coco_10k,
                            coco2014=opt.coco2014,
                            hpsv2=opt.hpsv2,
                            pixart=opt.pixart)
        if opt.max_prompts is not None:
            if opt.max_prompts <= 0:
                raise ValueError("--max_prompts must be a positive integer")
            n_prompts = min(opt.max_prompts, pes.shape[0])
            logger.info(f"Using {n_prompts} prompts out of {pes.shape[0]} available prompts")
            pes = pes[:n_prompts]
            pams = pams[:n_prompts]
        model.text_encoder = model.text_encoder.to("cpu")

    if opt.coco_9k: sp = "samples_9k"
    elif opt.coco_10k: sp = "samples_10k"
    elif opt.pixart: sp = "samples_pixart"
    elif opt.coco2014: sp = "samples_2014"
    elif opt.hpsv2: sp = 'samples_hpsv2'
    else: sp = "samples"

    assert(opt.cond)
    if opt.ptq:
        wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': opt.weight_quant_method,
                    'mantissa_bits': opt.weight_mantissa_bits,
                    'attn_weight_mantissa': opt.attn_weight_mantissa,
                    'ff_weight_mantissa': opt.ff_weight_mantissa,
                    'weight_group_size': opt.weight_group_size,
                    'fp_biased_adaround': opt.no_fp_biased_adaround,
                    'fp': (not opt.disable_fp_quant),
                    'group_quant': (not opt.disable_group_quant),
                    'sym': opt.w_sym,
                    'clip_ratio': opt.w_clip_ratio
                    }
        aq_params = {'n_bits': opt.act_bit, 'channel_wise': False, 'scale_method': opt.act_quant_method, 'leaf_param':  opt.quant_act, 
                    'mantissa_bits': opt.act_mantissa_bits,
                    'asym_softmax': opt.asym_softmax,
                    'online_act_quant': (not opt.disable_online_act_quant),
                    'fp': (not opt.disable_fp_quant)
                    }
        if opt.resume:
            logger.info('Load with min-max quick initialization')
            wq_params['scale_method'] = 'max'
            aq_params['scale_method'] = 'max'
        if opt.resume_w:
            wq_params['scale_method'] = 'max'
            
        qnn = QuantModel(
            model=model.transformer, weight_quant_params=wq_params, act_quant_params=aq_params,
            act_quant_mode="qdiff", sm_abit=opt.sm_abit)
            
        qnn.to(device)
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
            logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape} {cali_data[2].shape}")

            cali_xs, cali_ts, cali_cs = cali_data
            if opt.resume_w:
                resume_cali_model(qnn, opt.cali_ckpt, cali_data, False, cond=opt.cond)
            else:
                logger.info("Initializing weight quantization parameters")

            qnn.set_quant_state(weight_quant=True, act_quant=False)

            cali_xs = cali_xs.to(device)
            cali_ts = cali_ts.to(device)
            cali_cs = cali_cs.to(device)
            _ = qnn(
                cali_xs[:2],
                timestep=cali_ts[:2],
                encoder_hidden_states=cali_cs[:2],
                added_cond_kwargs=pixart_alpha_aca_dict(cali_xs[:2]),
            )
            logger.info("Initializing has done!")

            kwargs = dict(cali_data=cali_data, batch_size=opt.cali_batch_size, 
                    iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                    warmup=0.2, act_quant=False, opt_mode='mse', cond=opt.cond, sequential=opt.sequential_w,
                    no_adaround=opt.no_adaround, multi_gpu=False)
        
            def recon_model(model):
                """
                Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
                """
                for name, module in model.named_children():
                    logger.info(f"{name} {isinstance(module, BaseQuantBlock)}")
                    if isinstance(module, QuantModule):
                        if module.ignore_reconstruction is True:
                            logger.info('Ignore reconstruction of layer {}'.format(name))
                            continue
                        else:
                            logger.info('Reconstruction for layer {}'.format(name))
                            layer_reconstruction(qnn, module, **kwargs)
                    elif isinstance(module, BaseQuantBlock):
                        if module.ignore_reconstruction is True:
                            logger.info('Ignore reconstruction of block {}'.format(name))
                            continue
                        else:
                            logger.info('Reconstruction for block {}'.format(name))
                            block_reconstruction(qnn, module, **kwargs)
                    else:
                        recon_model(module)

            # === MEMORY SAFE GPTQ BLOCK CALIBRATION ===
            if opt.gptq:
                from qdiff.gptq import GPTQ
                logger.info("Starting Memory-Safe GPTQ calibration...")
                
                block_names = []
                for name, module in qnn.named_modules():
                    if isinstance(module, QuantModule) and not module.ignore_reconstruction:
                        parts = name.split(".")
                        root_block = ".".join(parts[:3]) if "transformer_blocks" in name else parts[0]
                        if root_block not in block_names:
                            block_names.append(root_block)

                for block_name in block_names:
                    logger.info(f"Running GPTQ on block group: {block_name}...")
                    
                    subset_layers = {}
                    for name, module in qnn.named_modules():
                        if name.startswith(block_name) and isinstance(module, QuantModule) and not module.ignore_reconstruction:
                            subset_layers[name] = module
                            
                    if not subset_layers:
                        continue

                    gptqs = {}
                    handles = []
                    
                    for name, module in subset_layers.items():
                        gptqs[name] = GPTQ(module)
                        gptqs[name].quantizer = module.weight_quantizer
                        
                        def add_batch(name):
                            def tmp(_, inp, out):
                                gptqs[name].add_batch(inp[0].data, out.data)
                            return tmp
                        handles.append(module.register_forward_hook(add_batch(name)))

                    # Forward pass to populate Hessians for this block ONLY using Diffusers native routing
                    qnn.eval()
                    bs = opt.cali_batch_size
                    with torch.no_grad():
                        for i in range(0, cali_xs.size(0), bs):
                            qnn(
                                cali_xs[i:i+bs], 
                                timestep=cali_ts[i:i+bs], 
                                encoder_hidden_states=cali_cs[i:i+bs], 
                                added_cond_kwargs=pixart_alpha_aca_dict(cali_xs[i:i+bs])
                            )

                    for h in handles:
                        h.remove()

                    # Execute GPTQ update
                    for name, gptq in gptqs.items():
                        logger.info(f"Quantizing {name}...")
                        gptq.fasterquant(
                            percdamp=opt.gptq_percdamp, 
                            groupsize=opt.gptq_groupsize,
                            blocksize=opt.gptq_blocksize
                        )
                        gptq.free()
                        
                    torch.cuda.empty_cache()
                    gc.collect()

                logger.info("GPTQ Calibration Complete!")
            # ==========================================

            if not opt.resume_w:
                if not opt.gptq:
                    logger.info("Doing weight calibration (BRECQ)")
                    recon_model(qnn)
                
                qnn.set_quant_state(weight_quant=True, act_quant=False)
                
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
                logger.info(model.transformer)
                
            if opt.quant_act and opt.disable_online_act_quant:
                logger.info("UNet model")
                logger.info(model.transformer)                  
                logger.info("Doing activation calibration")
                # Initialize activation quantization parameters
                qnn.set_quant_state(True, True)
                with torch.no_grad():
                    inds = np.random.choice(cali_xs.shape[0], 16, replace=False)
                    _ = qnn(
                        cali_xs[inds],
                        timestep=cali_ts[inds],
                        encoder_hidden_states=cali_cs[inds],
                        added_cond_kwargs=pixart_alpha_aca_dict(cali_xs[inds]),
                    )
                    if opt.running_stat:
                        logger.info('Running stat for activation quantization')
                        inds = np.arange(cali_xs.shape[0])
                        np.random.shuffle(inds)
                        qnn.set_running_stat(True, opt.rs_sm_only)
                        for i in trange(int(cali_xs.size(0) / 16)):
                            _ = qnn(
                                cali_xs[inds[i * 16:(i + 1) * 16]],
                                timestep=cali_ts[inds[i * 16:(i + 1) * 16]],
                                encoder_hidden_states=cali_cs[inds[i * 16:(i + 1) * 16]],
                                added_cond_kwargs=pixart_alpha_aca_dict(cali_xs[inds[i * 16:(i + 1) * 16]]),
                            )
                        qnn.set_running_stat(False, opt.rs_sm_only)

                kwargs = dict(
                    cali_data=cali_data, batch_size=opt.cali_batch_size, iters=opt.cali_iters_a, act_quant=True, 
                    opt_mode='mse', lr=opt.cali_lr, p=opt.cali_p, cond=opt.cond,
                    sequential=opt.sequential_a, multi_gpu=False)
                recon_model(qnn)
                qnn.set_quant_state(weight_quant=True, act_quant=True)
            elif opt.quant_act:
                # To be implement
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

        qnn = qnn.to(device=device, dtype=torch.float16)
        model.transformer = qnn
    
    sample_path = os.path.join(outpath, sp)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # write config out
    sampling_file = os.path.join(outpath, "sampling_config.yaml")
    sampling_conf = vars(opt)
    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    if opt.verbose:
        logger.info("UNet model")
        logger.info(model.model)

    batch_size = opt.n_samples
    if opt.prompt is None:
        for i in tqdm(range(0, pes.shape[0], batch_size), desc="data"):
            torch.manual_seed(42) # Meaning of Life, the Universe and Everything
            prompt_embeds = pes[i:i + batch_size].to(device)
            image = model(prompt=None, negative_prompt=None,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_mask = pams[i:i + batch_size].to(device),
                        negative_prompt_embeds = npe.expand(prompt_embeds.shape[0], -1, -1),
                        negative_prompt_attention_mask = npam.expand(prompt_embeds.shape[0], -1),
                        height=opt.res, width=opt.res).images

            for j, img in enumerate(image):
                img.save(os.path.join(sample_path, f"{i+j}.png"))
    else:
        torch.manual_seed(42) # Meaning of Life, the Universe and Everything
        prompt = [opt.prompt]
        image = model(prompt=prompt, height=opt.res, width=opt.res).images

        for j, img in enumerate(image):
            img.save(os.path.join(sample_path, f"{j}.png"))

    logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \n Enjoy.")

if __name__ == "__main__":
    main()