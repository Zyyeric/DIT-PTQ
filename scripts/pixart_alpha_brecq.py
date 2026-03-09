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
import torch.distributed as dist
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
from qdiff.nvtx import (
    DenoisingStepTracker,
    nvtx_range,
    wrap_module_forward,
    wrap_named_modules_by_predicate,
    wrap_named_modules_by_suffix,
    wrap_object_method,
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from transformers import AutoFeatureExtractor

logger = logging.getLogger(__name__)

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


LEAF_NVTX_SUFFIXES = {
    ".attn1.to_out.0": "attn1.to_out",
    ".attn2.to_out.0": "attn2.to_out",
    ".ff.net.0.proj": "ff.net.0.proj",
    ".attn1.to_q": "attn1.to_q",
    ".attn1.to_k": "attn1.to_k",
    ".attn1.to_v": "attn1.to_v",
    ".attn2.to_q": "attn2.to_q",
    ".attn2.to_k": "attn2.to_k",
    ".attn2.to_v": "attn2.to_v",
    ".ff.net.2": "ff.net.2",
}

PARENT_NVTX_SUFFIXES = {
    ".attn1": "self_attn",
    ".attn2": "cross_attn",
    ".ff": "ffn",
}


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


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


def init_parallel(parallelism: str, backend: str):
    mode = parallelism.lower()
    if mode == "dp":
        mode = "ddp"

    if mode != "ddp":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return mode, False, 0, 0, 1, device

    if not torch.cuda.is_available():
        raise RuntimeError("--parallelism ddp requires CUDA.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("--parallelism ddp must be launched with torchrun (RANK/WORLD_SIZE env vars missing).")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    return mode, True, rank, local_rank, world_size, device


def finalize_parallel(use_ddp: bool):
    if use_ddp and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_transformer_block(name, module):
    return all(hasattr(module, attr) for attr in ("attn1", "ff", "norm1"))


def suppress_quant_module_prints(module):
    for submodule in module.modules():
        if isinstance(submodule, QuantModule):
            submodule.run_prints = False


def install_nvtx_instrumentation(pipeline, enabled: bool):
    step_tracker = DenoisingStepTracker()
    if not enabled:
        return step_tracker

    wrap_object_method(pipeline, "encode_prompt", "encode_prompt", enabled=enabled)
    wrap_object_method(pipeline, "prepare_latents", "prepare_latents", enabled=enabled)
    wrap_object_method(pipeline, "run_safety_checker", "run_safety_checker", enabled=enabled)
    if hasattr(pipeline, "image_processor") and pipeline.image_processor is not None:
        wrap_object_method(pipeline.image_processor, "postprocess", "image_postprocess", enabled=enabled)
    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        wrap_object_method(pipeline.vae, "decode", "vae_decode", enabled=enabled)

    transformer = pipeline.transformer
    wrap_module_forward(
        transformer,
        "transformer_forward",
        enabled=enabled,
    )
    wrap_named_modules_by_predicate(
        transformer,
        is_transformer_block,
        "block",
        enabled=enabled,
    )
    wrap_named_modules_by_suffix(
        transformer,
        PARENT_NVTX_SUFFIXES,
        enabled=enabled,
    )
    wrap_named_modules_by_suffix(
        transformer,
        LEAF_NVTX_SUFFIXES,
        enabled=enabled,
    )

    scheduler = pipeline.scheduler
    wrap_object_method(scheduler, "set_timesteps", "set_timesteps", enabled=enabled)
    original_scale_model_input = scheduler.scale_model_input
    original_scheduler_step = scheduler.step

    def wrapped_scale_model_input(*args, **kwargs):
        step_tracker.begin_step(enabled=enabled)
        with nvtx_range("scheduler_scale_model_input", enabled=enabled):
            return original_scale_model_input(*args, **kwargs)

    def wrapped_scheduler_step(*args, **kwargs):
        try:
            with nvtx_range("scheduler_step", enabled=enabled):
                return original_scheduler_step(*args, **kwargs)
        finally:
            step_tracker.end_step(enabled=enabled)

    scheduler.scale_model_input = wrapped_scale_model_input
    scheduler.step = wrapped_scheduler_step
    return step_tracker


def run_generation(pipeline, step_tracker: DenoisingStepTracker, enabled: bool, **kwargs):
    step_tracker.reset()
    try:
        with nvtx_range("pipeline_generate", enabled=enabled):
            return pipeline(**kwargs)
    finally:
        step_tracker.end_step(enabled=enabled)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    help="the prompt to render; if not provided, uses dataset captions"
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="limit number of caption prompts used for generation when --prompt is not set",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    #parser.add_argument(
    #   "--W",
    #    type=int,
    #    default=1024,
    #    help="image width, in pixel space",
    #)
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    # linear quantization configs
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_only", action="store_true",
        help="Quantize weights only and keep activations in FP16/BF16.",
    )
    parser.add_argument(
        "--act_only", action="store_true",
        help="Quantize activations only and keep weights in FP16/BF16.",
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--quant_mode", type=str, default="symmetric", 
        choices=["linear", "squant", "qdiff"], 
        help="quantization mode to use"
    )

    # qdiff specific configs
    parser.add_argument(
        "--cali_st", type=int, default=20, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=8, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=128, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=20000, 
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int, 
        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, 
        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, 
        help='L_p norm minimization for LSQ')
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cali_data_path", type=str, default="pixart_calib_brecq.pt",
        help="calibration dataset name"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--adaround", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--resume_w", action="store_true",
        help="resume the calibrated qdiff model weights only"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--no_grad_ckpt", action="store_true",
        help="disable gradient checkpointing"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="use split strategy in skip connection"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--rs_sm_only", action="store_true",
        help="use running statistics only for softmax act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=16,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument(
        "--refiner", action="store_true", default=False,
        help="use SDXL refiner")
    parser.add_argument(
        "--sequential_w", action="store_true", default=False,
        help="Sequential weight quantization")
    parser.add_argument(
        "--sequential_a", action="store_true", default=False,
        help="Sequential activation quantization")
    parser.add_argument(
        "--weight_mantissa_bits",
        type=int,
        default=None,
        help="weight_mantissa bit",
    )
    parser.add_argument(
        "--act_mantissa_bits",
        type=int,
        default=None,
        help="weight_mantissa bit",
    )
    parser.add_argument(
    "--no_adaround", action="store_true",
    help="Disable adaround in weight quantization"
    )
    parser.add_argument(
        "--attn_weight_mantissa",
        type=int,
        default=None,
        help="mantissa bit for weight quantization",
    )
    parser.add_argument(
        "--ff_weight_mantissa",
        type=int,
        default=None,
        help="mantissa bit for feed forward weight",
    )
    parser.add_argument(
    "--asym_softmax", action="store_true", default=False,
    help="Use asymmetric quantization for attention's softmax and get an extra bits, default is using asymmetric"
    )
    parser.add_argument(
        "--weight_group_size",
        type=int,
        default=128,
        help="independent quantization to groups of <size> consecutive weights",
    )
    parser.add_argument(
    "--no_fp_biased_adaround", action="store_false",
    help="Disable FP scale awared adaround"
    )
    parser.add_argument(
    "--disable_online_act_quant", action="store_true",
    help="Disable online activation quantization"
    )
    parser.add_argument(
        "--coco_9k", action="store_true", default=False,
        help="generate images for coco val prompts 1000-9999")
    parser.add_argument(
        "--coco_10k", action="store_true", default=False,
        help="generate images for coco val prompts 0-9999")
    parser.add_argument(
        "--coco2014", action="store_true", default=False,
        help="generate images for coco 2014 prompts 0-9999")
    parser.add_argument(
        "--hpsv2", action="store_true", default=False,
        help="generate images for coco 2014 prompts 0-9999")
    parser.add_argument(
        "--pixart", action="store_true", default=False,
        help="generate images for 120 high-detailed pixart prompts (no ground truth)")
    parser.add_argument(
    "--disable_fp_quant", action="store_true",
    help="Use integer quantization"
    )
    parser.add_argument(
    "--disable_group_quant", action="store_true",
    help="Disable group weight quantization"
    )
    parser.add_argument(
        "--parallelism",
        type=str,
        default="none",
        choices=["none", "dp", "ddp"],
        help="Calibration parallelism mode. 'dp' is treated as 'ddp'.",
    )
    parser.add_argument(
        "--ddp_backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend when --parallelism=ddp.",
    )
    parser.add_argument(
        "--parallel_generate",
        action="store_true",
        help="When using DDP and dataset prompts, shard prompt generation across ranks.",
    )
    parser.add_argument(
        "--nvtx_profile",
        action="store_true",
        help="Add NVTX ranges for generation, denoising steps, transformer blocks, attention, and FFN projections.",
    )
    opt = parser.parse_args()
    parallel_mode, use_ddp, rank, local_rank, world_size, device = init_parallel(
        opt.parallelism, opt.ddp_backend
    )
    is_main_process = rank == 0
    do_parallel_generate = use_ddp and opt.parallel_generate and opt.prompt is None

    seed_everything(opt.seed + rank)

    os.makedirs(opt.outdir, exist_ok=True)
    run_tag = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") if is_main_process else None
    if use_ddp:
        obj_list = [run_tag]
        dist.broadcast_object_list(obj_list, src=0)
        run_tag = obj_list[0]
    outpath = os.path.join(opt.outdir, run_tag)
    if is_main_process:
        os.makedirs(outpath, exist_ok=True)
    if use_ddp:
        dist.barrier()

    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=(
            [logging.FileHandler(log_path), logging.StreamHandler()]
            if is_main_process
            else [logging.StreamHandler()]
        )
    )
    logger = logging.getLogger(__name__)
    logger.info(f"parallelism={parallel_mode}, world_size={world_size}, rank={rank}, local_rank={local_rank}")
    if use_ddp and opt.parallel_generate and opt.prompt is not None and is_main_process:
        logger.info("--parallel_generate is ignored when --prompt is provided (single prompt path).")


    from diffusers import PixArtAlphaPipeline
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    # Diffusers 0.29.2 resolves `_execution_device` through `components`, which
    # becomes fragile after swapping in the quantized transformer. `device`
    # already resolves from the active module set and is stable here.
    DiffusionPipeline._execution_device = property(lambda self: self.device)
    model = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16
    ).to(device)

    from qdiff.caption_util import get_captions
    pes, pams, npe, npam = None, None, None, None
    if opt.prompt is None and (is_main_process or do_parallel_generate):
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

    if opt.coco_9k:
        sp = "samples_9k"
    elif opt.coco_10k:
        sp = "samples_10k"
    elif opt.pixart:
        sp = "samples_pixart"
    elif opt.coco2014:
        sp = "samples_2014"
    elif opt.hpsv2:
        sp = 'samples_hpsv2'
    else:
        sp = "samples"

    assert(opt.cond)
    if opt.ptq:
        if opt.weight_only and opt.act_only:
            raise ValueError("--weight_only and --act_only are mutually exclusive")
        if opt.resume_w and opt.act_only:
            raise ValueError("--resume_w is only valid when weight quantization is enabled")

        enable_weight_quant = not opt.act_only
        enable_act_quant = opt.quant_act or opt.act_only
        if opt.weight_only:
            enable_act_quant = False

        wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'mse', 'fp': True, 
                    'mantissa_bits': opt.weight_mantissa_bits,
                    'attn_weight_mantissa': opt.attn_weight_mantissa,
                    'ff_weight_mantissa': opt.ff_weight_mantissa,
                    'weight_group_size': opt.weight_group_size,
                    'fp_biased_adaround': opt.no_fp_biased_adaround,
                    'fp': (not opt.disable_fp_quant),
                    'group_quant': (not opt.disable_group_quant)
                    }
        aq_params = {'n_bits': opt.act_bit, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param':  enable_act_quant, 
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
        #exit(0)
        qnn.to(device)
        qnn.eval()

        if opt.no_grad_ckpt:
            logger.info('Not use gradient checkpointing for transformer blocks')
            qnn.set_grad_ckpt(False)

        if opt.resume:
            #noisy_latents = torch.randn(1, 4, 64, 64, dtype=torch.float16) #.cuda()
            #timesteps = torch.zeros(1).long() #.cuda()
            #class_labels = torch.tensor(list(range(1))) #.cuda()
            #cali_data = (noisy_latents, timesteps, class_labels)
            sample_data = torch.load(opt.cali_data_path)
            cali_data = get_train_samples_custom(opt, sample_data, opt.ddim_steps)
            resume_cali_model(
                qnn, opt.cali_ckpt, cali_data, enable_act_quant, "qdiff",
                cond=opt.cond, weight_quant=enable_weight_quant
            )
        else:
            logger.info(f"Sampling data from {opt.cali_st} timesteps for calibration")
            
            sample_data = torch.load(opt.cali_data_path)  # This is de-commented when needed
        
            # [step, batch_size * 2, in_channel, height, width] of the noise latent image
            #noisy_latents = torch.randn(opt.ddim_steps, 32, 4, 64, 64, dtype=torch.float16) #.cuda()
            #timesteps = torch.zeros(opt.ddim_steps, 32).long() #.cuda()
            #class_labels = torch.zeros(opt.ddim_steps, 32, 120, 4096) #.cuda()
            #sample_data = {'xs': noisy_latents, #[2, 4, 64, 64]
            #            'ts': timesteps,
            #            'cs': class_labels} #[2, 120, 4096]

            sample_data['xs'] = sample_data['xs'].to(torch.float16)
            sample_data['cs'] = sample_data['cs'].to(torch.float16)
            cali_data = get_train_samples_custom(opt, sample_data, opt.ddim_steps)
            del(sample_data)
            gc.collect()
            logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape} {cali_data[2].shape}")

            cali_xs, cali_ts, cali_cs = cali_data
            if opt.resume_w:
                resume_cali_model(
                    qnn, opt.cali_ckpt, cali_data, False,
                    cond=opt.cond, weight_quant=enable_weight_quant
                )
            else:
                if enable_weight_quant:
                    logger.info("Initializing weight quantization parameters")
                else:
                    logger.info("Skipping weight quantization; activations will be calibrated against FP16/BF16 weights")

            qnn.set_quant_state(weight_quant=enable_weight_quant, act_quant=False)

            #print(qnn)
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

            # TODO adjust some things here
            kwargs = dict(cali_data=cali_data, batch_size=opt.cali_batch_size, 
                    iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                    warmup=0.2, act_quant=False, opt_mode='mse', cond=opt.cond, sequential=opt.sequential_w,
                    no_adaround=opt.no_adaround, multi_gpu=use_ddp, weight_quant=enable_weight_quant)
        
            def recon_model(model):
                """
                Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
                """
                for name, module in model.named_children():
                    logger.info(f"{name} {isinstance(module, BaseQuantBlock)}")
                    """
                    if name == 'output_blocks':
                        logger.info("Finished calibrating input and mid blocks, saving temporary checkpoint...")
                        in_recon_done = True
                        torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
                    if name.isdigit() and int(name) >= 9:
                        logger.info(f"Saving temporary checkpoint at {name}...")
                        torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
                    """
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

            if enable_weight_quant and not opt.resume_w:
                logger.info("Doing weight calibration")
                recon_model(qnn)
                qnn.set_quant_state(weight_quant=True, act_quant=False)
                if use_ddp:
                    dist.barrier()
                # NOTE Checkpoint weight quantization calibation separately
                logger.info("Saving calibrated quantized UNet model")
                for m in qnn.model.modules():
                    if isinstance(m, AdaRoundQuantizer):
                        m.zero_point = nn.Parameter(m.zero_point)
                        m.delta = nn.Parameter(m.delta)
                    elif isinstance(m, UniformAffineQuantizer) and enable_act_quant:
                        if m.zero_point is not None:
                            if not torch.is_tensor(m.zero_point):
                                m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                            else:
                                m.zero_point = nn.Parameter(m.zero_point)
                if is_main_process:
                    torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt_wq.pth"))
                logger.info(model.transformer)
            if enable_act_quant and opt.disable_online_act_quant:
                logger.info("UNet model")
                logger.info(model.transformer)                    
                logger.info("Doing activation calibration")
                # Initialize activation quantization parameters
                qnn.set_quant_state(enable_weight_quant, True)
                with torch.no_grad():
                    init_bs = min(16, cali_xs.shape[0])
                    inds = np.random.choice(cali_xs.shape[0], init_bs, replace=False)
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
                        for start in trange(0, cali_xs.size(0), 16, disable=opt.nvtx_profile):
                            end = min(start + 16, cali_xs.size(0))
                            _ = qnn(
                                cali_xs[inds[start:end]],
                                timestep=cali_ts[inds[start:end]],
                                encoder_hidden_states=cali_cs[inds[start:end]],
                                added_cond_kwargs=pixart_alpha_aca_dict(cali_xs[inds[start:end]]),
                            )
                        qnn.set_running_stat(False, opt.rs_sm_only)

                # TODO change these guys too.
                kwargs = dict(
                    cali_data=cali_data, batch_size=opt.cali_batch_size, iters=opt.cali_iters_a, act_quant=True, 
                    opt_mode='mse', lr=opt.cali_lr, p=opt.cali_p, cond=opt.cond,
                    sequential=opt.sequential_a, multi_gpu=use_ddp, weight_quant=enable_weight_quant)
                recon_model(qnn)
                qnn.set_quant_state(weight_quant=enable_weight_quant, act_quant=True)
            elif enable_act_quant:
                raise NotImplementedError(
                    "Online activation quantization is not implemented. "
                    "Use --disable_online_act_quant for the supported offline activation calibration path."
                )
            # Currently, this does not work
            """
            print("Report delta change")
            for n, m in qnn.named_modules():
                if isinstance(m, QuantModule):
                    m.report_delta_shift()
            """
            
            logger.info("Saving calibrated quantized UNet model")
            for m in qnn.model.modules():
                if isinstance(m, AdaRoundQuantizer):
                    m.zero_point = nn.Parameter(m.zero_point)
                    m.delta = nn.Parameter(m.delta)
                elif isinstance(m, UniformAffineQuantizer) and enable_act_quant and opt.disable_online_act_quant:
                    if m.zero_point is not None:
                        if not torch.is_tensor(m.zero_point):
                            m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                        else:
                            m.zero_point = nn.Parameter(m.zero_point)
            if is_main_process:
                torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))

        qnn = qnn.to(device=device, dtype=torch.float16)
        model.transformer = qnn

    if opt.nvtx_profile:
        suppress_quant_module_prints(model.transformer)

    step_tracker = install_nvtx_instrumentation(model, opt.nvtx_profile)
    
    #model.text_encoder = model.text_encoder.to("cuda")

    if use_ddp:
        dist.barrier()

    if do_parallel_generate:
        sample_path = os.path.join(outpath, sp)
        if is_main_process:
            os.makedirs(sample_path, exist_ok=True)
            sampling_file = os.path.join(outpath, "sampling_config.yaml")
            sampling_conf = vars(opt)
            with open(sampling_file, 'a+') as f:
                yaml.dump(sampling_conf, f, default_flow_style=False)
        dist.barrier()

        batch_size = opt.n_samples
        npe = npe.to(device)
        npam = npam.to(device)
        total = pes.shape[0]
        local_indices = list(range(rank, total, world_size))
        for start in tqdm(
            range(0, len(local_indices), batch_size),
            desc=f"rank{rank}-data",
            disable=(rank != 0 or opt.nvtx_profile),
        ):
            torch.manual_seed(42)
            batch_idx = local_indices[start:start + batch_size]
            prompt_embeds = pes[batch_idx].to(device)
            image = run_generation(
                model,
                step_tracker,
                opt.nvtx_profile,
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=pams[batch_idx].to(device),
                negative_prompt_embeds=npe.expand(prompt_embeds.shape[0], -1, -1),
                negative_prompt_attention_mask=npam.expand(prompt_embeds.shape[0], -1),
                height=opt.res,
                width=opt.res,
            ).images

            for j, img in enumerate(image):
                img.save(os.path.join(sample_path, f"{batch_idx[j]}.png"))
        dist.barrier()
        if is_main_process:
            logging.info(f"Parallel generation complete. Samples are in:\n{outpath}")
    elif is_main_process:
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

        #start_code = None
        #if opt.fixed_code:
        #    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        batch_size = opt.n_samples
        if opt.prompt is None:
            for i in tqdm(range(0, pes.shape[0], batch_size), desc="data", disable=opt.nvtx_profile):
                torch.manual_seed(42) # Meaning of Life, the Universe and Everything
                prompt_embeds = pes[i:i + batch_size].to(device)
                image = run_generation(
                    model,
                    step_tracker,
                    opt.nvtx_profile,
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=pams[i:i + batch_size].to(device),
                    negative_prompt_embeds=npe.expand(prompt_embeds.shape[0], -1, -1),
                    negative_prompt_attention_mask=npam.expand(prompt_embeds.shape[0], -1),
                    height=opt.res,
                    width=opt.res,
                ).images

                for j, img in enumerate(image):
                    img.save(os.path.join(sample_path, f"{i+j}.png"))
        else:
            torch.manual_seed(42) # Meaning of Life, the Universe and Everything
            prompt = [opt.prompt]
            image = run_generation(
                model,
                step_tracker,
                opt.nvtx_profile,
                prompt=prompt,
                height=opt.res,
                width=opt.res,
            ).images

            for j, img in enumerate(image):
                img.save(os.path.join(sample_path, f"{j}.png"))

        logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \n"
              f" \nEnjoy.")

    finalize_parallel(use_ddp)


if __name__ == "__main__":
    main()
