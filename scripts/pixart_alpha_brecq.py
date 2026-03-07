import argparse, os, datetime, gc, yaml
import logging
import time
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
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

# ── logging helpers (mirrors pixart_calib.py style) ──────────────────────────
def _now():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(msg, level="info"):
    fn = getattr(logger, level, logger.info)
    fn(f"[{_now()}] {msg}")

def log_gpu(tag=""):
    if torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated()  / 1024**3
        reserv = torch.cuda.memory_reserved()   / 1024**3
        total  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        label  = f" [{tag}]" if tag else ""
        log(f"GPU{label}  allocated={alloc:.2f}GB  reserved={reserv:.2f}GB  total={total:.2f}GB")

def elapsed(t0):
    return f"{time.time()-t0:.1f}s"

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    quant = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n and p.dtype in [torch.float16, torch.float32])
    return total, quant
# ─────────────────────────────────────────────────────────────────────────────

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
    SCRIPT_START = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--outdir", type=str, nargs="?", default="outputs/txt2img-samples")
    parser.add_argument("--skip_grid", action='store_true')
    parser.add_argument("--skip_save", action='store_true')
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--fixed_code", action='store_true')
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--n_iter", type=int, default=1)
    # FIX 1: --res drives model selection (512 → 512x512, 1024 → 1024-MS)
    parser.add_argument("--res", type=int, default=512, choices=[512, 1024],
                        help="Resolution; also selects PixArt model variant. "
                             "Must match the resolution used when generating calib data.")
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_rows", type=int, default=0)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--from-file", type=str)
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, choices=["full", "autocast"], default="autocast")

    # quantization configs
    parser.add_argument("--ptq", action="store_true")
    parser.add_argument("--quant_act", action="store_true")
    parser.add_argument("--weight_bit", type=int, default=8)
    parser.add_argument("--act_bit", type=int, default=8)
    parser.add_argument("--quant_mode", type=str, default="symmetric", choices=["linear", "squant", "qdiff"])

    # qdiff configs
    parser.add_argument("--cali_st", type=int, default=20)
    parser.add_argument("--cali_batch_size", type=int, default=8)
    parser.add_argument("--cali_n", type=int, default=128)
    parser.add_argument("--cali_iters", type=int, default=10000,
                        help="BRECQ iterations (unused when --gptq is set)")
    parser.add_argument('--cali_iters_a', default=1000, type=int,
                        help='LSQ activation calibration iterations. '
                             'Values <100 effectively disable act calibration.')
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
    parser.add_argument("--sm_abit", type=int, default=16)
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
    parser.add_argument("--weight_quant_method", type=str, default="mse", choices=["max", "mse"])
    parser.add_argument("--act_quant_method", type=str, default="mse", choices=["max", "mse"])
    parser.add_argument("--gptq", action="store_true",
                        help="Use GPTQ. Mutually exclusive with BRECQ (skipped automatically).")
    parser.add_argument("--gptq_percdamp", type=float, default=0.01)
    parser.add_argument("--gptq_groupsize", type=int, default=-1)
    parser.add_argument("--gptq_blocksize", type=int, default=128)
    parser.add_argument("--gptq_cali_n", type=int, default=256,
                        help="Number of calibration samples for GPTQ Hessian collection. "
                             "128-256 is sufficient — GPTQ is not sensitive to sample count "
                             "beyond this. Default 256. Full cali_n (e.g. 5120) makes "
                             "Hessian collection 20-40x slower with negligible quality gain.")
    parser.add_argument("--w_sym", action="store_true")
    parser.add_argument("--w_clip_ratio", type=float, default=1.0)

    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(opt.seed)

    # ── FIX 1: model selection from --res ─────────────────────────────────────
    RES_TO_MODEL = {
        512:  "PixArt-alpha/PixArt-XL-2-512x512",
        1024: "PixArt-alpha/PixArt-XL-2-1024-MS",
    }
    model_id = RES_TO_MODEL[opt.res]

    # Early calib data shape validation — fail before loading any GPU model
    if os.path.exists(opt.cali_data_path):
        _tmp = torch.load(opt.cali_data_path, map_location="cpu")
        _xs_hw = _tmp['xs'].shape[-1]
        _expected_hw = opt.res // 8
        if _xs_hw != _expected_hw:
            raise ValueError(
                f"Calibration data spatial size {_xs_hw} does not match "
                f"--res {opt.res} (expected latent H/W={_expected_hw}). "
                f"Regenerate calib data at res={opt.res} or adjust --res."
            )
        log(f"Calib data shape validated: xs={tuple(_tmp['xs'].shape)}  "
            f"ts={tuple(_tmp['ts'].shape)}  cs={tuple(_tmp['cs'].shape)}")
        del _tmp
    # ─────────────────────────────────────────────────────────────────────────

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

    # ── Header ────────────────────────────────────────────────────────────────
    log("=" * 70)
    log("PixArt-Alpha BRECQ/GPTQ Quantization Script")
    log("=" * 70)
    log(f"Device        →  {device}"
        + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    log(f"Model         →  {model_id}")
    log(f"Resolution    →  {opt.res}x{opt.res}")
    log(f"Calib data    →  {opt.cali_data_path}")
    log(f"Output dir    →  {outpath}")
    log(f"Seed          →  {opt.seed}")
    log(f"PTQ           →  {opt.ptq}")
    if opt.ptq:
        log(f"  weight_bit={opt.weight_bit}  act_bit={opt.act_bit}  "
            f"quant_mode={opt.quant_mode}  quant_act={opt.quant_act}")
        log(f"  gptq={opt.gptq}  gptq_groupsize={opt.gptq_groupsize}  "
            f"gptq_blocksize={opt.gptq_blocksize}  gptq_percdamp={opt.gptq_percdamp}  "
            f"gptq_cali_n={opt.gptq_cali_n}")
        log(f"  cali_batch_size={opt.cali_batch_size}  cali_iters={opt.cali_iters}  "
            f"cali_iters_a={opt.cali_iters_a}")
        log(f"  weight_group_size={opt.weight_group_size}  w_sym={opt.w_sym}  "
            f"w_clip_ratio={opt.w_clip_ratio}")
        log(f"  disable_fp_quant={opt.disable_fp_quant}  "
            f"disable_group_quant={opt.disable_group_quant}")
        if opt.gptq and opt.cali_iters != 10000:
            log("  NOTE: --cali_iters is unused when --gptq is set (BRECQ skipped)", "warning")
        if opt.quant_act and opt.cali_iters_a < 100:
            log(f"  WARNING: --cali_iters_a={opt.cali_iters_a} is very low — "
                "activation calibration will be effectively disabled.", "warning")
    log_gpu("startup")

    # ── 1. Load pipeline ──────────────────────────────────────────────────────
    log("")
    log("─" * 60)
    log(f"STEP 1  Loading pipeline: {model_id} ...")
    t0 = time.time()
    from diffusers import PixArtAlphaPipeline
    model = PixArtAlphaPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    log(f"  Pipeline loaded in {elapsed(t0)}")
    log(f"  Transformer type  →  {type(model.transformer).__name__}")
    log(f"  Scheduler type    →  {type(model.scheduler).__name__}")
    total_p, _ = count_params(model.transformer)
    log(f"  Transformer params →  {total_p/1e6:.1f}M")
    log_gpu("after pipeline load")

    # ── 2. Load captions ──────────────────────────────────────────────────────
    log("")
    log("─" * 60)
    log("STEP 2  Loading captions...")
    t0 = time.time()
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
            log(f"  Limiting to {n_prompts} prompts (from {pes.shape[0]} available)")
            pes = pes[:n_prompts]
            pams = pams[:n_prompts]
        log(f"  Prompt embeds shape  →  {tuple(pes.shape)}")
        log(f"  Neg prompt embeds    →  {tuple(npe.shape)}")
        model.text_encoder = model.text_encoder.to("cpu")
        log(f"  Text encoder moved to CPU to free VRAM")
    else:
        log(f"  Using single prompt: '{opt.prompt}'")
    log(f"  Captions ready in {elapsed(t0)}")
    log_gpu("after captions")

    if opt.coco_9k: sp = "samples_9k"
    elif opt.coco_10k: sp = "samples_10k"
    elif opt.pixart: sp = "samples_pixart"
    elif opt.coco2014: sp = "samples_2014"
    elif opt.hpsv2: sp = 'samples_hpsv2'
    else: sp = "samples"

    assert(opt.cond)

    # ── 3. Quantization ───────────────────────────────────────────────────────
    if opt.ptq:
        log("")
        log("─" * 60)
        log("STEP 3  Setting up quantization...")
        t0 = time.time()

        wq_params = {
            'n_bits': opt.weight_bit, 'channel_wise': True,
            'scale_method': opt.weight_quant_method,
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
        aq_params = {
            'n_bits': opt.act_bit, 'channel_wise': False,
            'scale_method': opt.act_quant_method,
            'leaf_param': opt.quant_act,
            'mantissa_bits': opt.act_mantissa_bits,
            'asym_softmax': opt.asym_softmax,
            'online_act_quant': (not opt.disable_online_act_quant),
            'fp': (not opt.disable_fp_quant)
        }
        log(f"  wq_params: {wq_params}")
        log(f"  aq_params: {aq_params}")

        if opt.resume:
            log("  Scale method overridden to 'max' for resume")
            wq_params['scale_method'] = 'max'
            aq_params['scale_method'] = 'max'
        if opt.resume_w:
            wq_params['scale_method'] = 'max'

        log("  Building QuantModel...")
        t_qnn = time.time()
        qnn = QuantModel(
            model=model.transformer, weight_quant_params=wq_params,
            act_quant_params=aq_params, act_quant_mode="qdiff", sm_abit=opt.sm_abit)
        qnn.to(device)
        qnn.eval()
        log(f"  QuantModel built in {elapsed(t_qnn)}")

        # Count quantizable layers
        n_quant_layers = sum(1 for m in qnn.modules() if isinstance(m, QuantModule))
        n_quant_blocks = sum(1 for m in qnn.modules() if isinstance(m, BaseQuantBlock))
        log(f"  Quantizable layers  →  {n_quant_layers}")
        log(f"  Quantizable blocks  →  {n_quant_blocks}")
        log_gpu("after QuantModel build")

        if opt.no_grad_ckpt:
            log("  Gradient checkpointing disabled")
            qnn.set_grad_ckpt(False)

        # ── 3a. Load calibration data ─────────────────────────────────────────
        log("")
        log("  Loading calibration data...")
        t_cali = time.time()
        sample_data = torch.load(opt.cali_data_path)
        log(f"    xs shape   →  {tuple(sample_data['xs'].shape)}")
        log(f"    ts shape   →  {tuple(sample_data['ts'].shape)}")
        log(f"    cs shape   →  {tuple(sample_data['cs'].shape)}")
        log(f"    ucs shape  →  {tuple(sample_data['ucs'].shape)}")
        log(f"    ts unique  →  {sample_data['ts'][:,0].unique().tolist()}")

        if opt.resume:
            cali_data = get_train_samples_custom(opt, sample_data, opt.ddim_steps)
            log(f"  Resuming from checkpoint: {opt.cali_ckpt}")
            resume_cali_model(qnn, opt.cali_ckpt, cali_data, opt.quant_act, "qdiff", cond=opt.cond)
        else:
            log(f"  Casting xs, cs to float16...")
            sample_data['xs'] = sample_data['xs'].to(torch.float16)
            sample_data['cs'] = sample_data['cs'].to(torch.float16)
            cali_data = get_train_samples_custom(opt, sample_data, opt.ddim_steps)
            del sample_data
            gc.collect()

            cali_xs, cali_ts, cali_cs = cali_data
            log(f"  cali_xs shape  →  {tuple(cali_xs.shape)}  dtype={cali_xs.dtype}")
            log(f"  cali_ts shape  →  {tuple(cali_ts.shape)}  unique={cali_ts.unique().tolist()}")
            log(f"  cali_cs shape  →  {tuple(cali_cs.shape)}  dtype={cali_cs.dtype}")
            log(f"  Calibration data loaded in {elapsed(t_cali)}")

            if opt.resume_w:
                log(f"  Resuming weights from: {opt.cali_ckpt}")
                if opt.gptq:
                    # GPTQ checkpoints contain only model weights (already
                    # quantized in-place by fasterquant). They do NOT contain
                    # AdaRound quantizer params (alpha/zero_point/delta).
                    # Load directly with strict=False — no AdaRound conversion.
                    log("  GPTQ resume: loading checkpoint directly (no AdaRound)")
                    ckpt = torch.load(opt.cali_ckpt, map_location='cpu')
                    missing, unexpected = qnn.load_state_dict(ckpt, strict=False)
                    log(f"    Loaded: {len(ckpt)} keys  "
                        f"missing={len(missing)}  unexpected={len(unexpected)}")
                    if missing:
                        log(f"    Missing (first 5): {missing[:5]}")
                    if unexpected:
                        log(f"    Unexpected (first 5): {unexpected[:5]}")
                    del ckpt
                    # Sync org_weight with loaded weights for QuantModules
                    for m in qnn.modules():
                        if isinstance(m, QuantModule) and hasattr(m, 'org_weight'):
                            m.org_weight = m.weight.data.clone()
                    log("  GPTQ resume complete — weights loaded")
                else:
                    resume_cali_model(qnn, opt.cali_ckpt, cali_data, False, cond=opt.cond)
                log("  resume_w: skipping init forward pass and calibration")
            else:
                log("  Initializing weight quantization parameters...")

                # ── Cali data stays on CPU throughout ─────────────────────────
                # cali_xs alone is ~10.5 GB and cali_cs ~5 GB. Moving them to GPU
                # all at once causes a silent CUDA OOM deadlock. We keep everything
                # on CPU and .to(device) only the small slices we need per operation.
                xs_gb = cali_xs.element_size() * cali_xs.numel() / 1024**3
                cs_gb = cali_cs.element_size() * cali_cs.numel() / 1024**3
                log(f"  Cali data staying on CPU:")
                log(f"    cali_xs: {tuple(cali_xs.shape)}  {xs_gb:.2f} GB")
                log(f"    cali_cs: {tuple(cali_cs.shape)}  {cs_gb:.2f} GB")
                log(f"    total: {xs_gb + cs_gb:.2f} GB  — NOT moving to GPU")
                log_gpu("before init forward pass")

                # Weight quant init forward pass — only 2 samples needed to
                # initialize UAQ delta/zero_point for every QuantModule.
                #
                # IMPORTANT: temporarily override scale_method to 'max' for this
                # init pass. The configured method (e.g. 'mse') runs a 100-point
                # grid search per layer group during the first forward pass, which
                # for 287 layers × (weight_cols / group_size) groups can take
                # 10-30 minutes and appears as a hang. 'max' is O(1) and just
                # needs to create correctly-shaped delta/zero_point tensors.
                # The actual MSE-optimal scales are found during BRECQ/GPTQ anyway.
                n_overridden = 0
                for m in qnn.modules():
                    if hasattr(m, 'weight_quantizer') and hasattr(m.weight_quantizer, 'scale_method'):
                        m.weight_quantizer.scale_method = 'max'
                        n_overridden += 1
                log(f"  Overrode scale_method → 'max' on {n_overridden} weight quantizers")
                log(f"  (will restore to '{opt.weight_quant_method}' after init pass)")

                qnn.set_quant_state(weight_quant=True, act_quant=False)
                log(f"  Moving 2 samples to {device} and running forward pass...")
                t_init = time.time()
                with torch.no_grad():
                    _xs2 = cali_xs[:2].to(device)
                    _ts2 = cali_ts[:2].to(device)
                    _cs2 = cali_cs[:2].to(device)
                    log(f"    xs2={tuple(_xs2.shape)} dtype={_xs2.dtype}  "
                        f"ts2={tuple(_ts2.shape)}  cs2={tuple(_cs2.shape)}")
                    _ = qnn(
                        _xs2,
                        timestep=_ts2,
                        encoder_hidden_states=_cs2,
                        added_cond_kwargs=pixart_alpha_aca_dict(_xs2),
                    )
                    del _xs2, _ts2, _cs2
                torch.cuda.empty_cache()
                log(f"  Init forward pass done in {elapsed(t_init)}")
                log_gpu("after init forward pass")

                # Count how many quantizers were actually initialized
                n_inited = sum(
                    1 for m in qnn.modules()
                    if hasattr(m, 'weight_quantizer')
                    and hasattr(m.weight_quantizer, 'inited')
                    and m.weight_quantizer.inited
                )
                log(f"  Weight quantizers initialized: {n_inited}/{n_overridden}")

                # ── Restore configured scale_method for BRECQ/GPTQ ───────────
                real_scale_method = opt.weight_quant_method
                n_restored = 0
                for m in qnn.modules():
                    if hasattr(m, 'weight_quantizer') and hasattr(m.weight_quantizer, 'scale_method'):
                        m.weight_quantizer.scale_method = real_scale_method
                        m.weight_quantizer.inited = False  # force re-init with correct method
                        n_restored += 1
                log(f"  Restored scale_method → '{real_scale_method}' on {n_restored} quantizers")
                log(f"  inited=False reset so GPTQ/BRECQ will recompute with '{real_scale_method}'")

            kwargs = dict(
                cali_data=cali_data, batch_size=opt.cali_batch_size,
                iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                warmup=0.2, act_quant=False, opt_mode='mse', cond=opt.cond,
                sequential=opt.sequential_w, no_adaround=opt.no_adaround, multi_gpu=False)

            def recon_model(model, depth=0):
                """Recursive block/layer reconstruction with per-module logging."""
                indent = "  " * depth
                for name, module in model.named_children():
                    if isinstance(module, QuantModule):
                        if module.ignore_reconstruction:
                            log(f"{indent}  SKIP layer: {name}")
                            continue
                        log(f"{indent}  RECON layer: {name}")
                        t_r = time.time()
                        layer_reconstruction(qnn, module, **kwargs)
                        log(f"{indent}    done in {elapsed(t_r)}")
                    elif isinstance(module, BaseQuantBlock):
                        if module.ignore_reconstruction:
                            log(f"{indent}  SKIP block: {name}")
                            continue
                        log(f"{indent}  RECON block: {name}")
                        t_r = time.time()
                        block_reconstruction(qnn, module, **kwargs)
                        log(f"{indent}    done in {elapsed(t_r)}")
                    else:
                        recon_model(module, depth + 1)

            # ── 3b. GPTQ ──────────────────────────────────────────────────────
            if opt.gptq and not opt.resume_w:
                from qdiff.gptq import GPTQ
                log("")
                log("─" * 60)
                log("STEP 3b  GPTQ calibration")
                log("  NOTE: BRECQ (recon_model) is SKIPPED — --gptq is set")
                log(f"  gptq_percdamp={opt.gptq_percdamp}  "
                    f"gptq_groupsize={opt.gptq_groupsize}  "
                    f"gptq_blocksize={opt.gptq_blocksize}")
                t_gptq = time.time()

                # Build block groups for GPTQ.
                # Strategy: each transformer_blocks.N is one group (10-11 layers).
                # Each non-transformer-block QuantModule is its OWN group —
                # the old code used parts[0]="model" for all of them, collapsing
                # pos_embed, proj_out, adaln_single, caption_projection into one
                # massive group with ALL 287 layers, which:
                #   (a) attaches hooks to every layer for every group's forward pass
                #   (b) builds Hessians simultaneously for unrelated layers
                #   (c) holds all Hessians in VRAM at the same time
                # Correct behavior: each group should share a forward pass because
                # only layers active during that pass can accumulate Hessian signal.
                block_names = []
                seen = set()
                for name, module in qnn.named_modules():
                    if not isinstance(module, QuantModule) or module.ignore_reconstruction:
                        continue
                    parts = name.split(".")
                    if "transformer_blocks" in parts:
                        # e.g. model.transformer_blocks.0.attn1.to_q → group: model.transformer_blocks.0
                        tb_idx = parts.index("transformer_blocks")
                        root = ".".join(parts[:tb_idx + 2])  # includes the block number
                    else:
                        # Non-transformer-block layers: each is its own group.
                        # e.g. model.pos_embed.proj, model.proj_out, model.adaln_single.linear
                        # Use the full module name so each gets processed independently.
                        root = name
                    if root not in seen:
                        seen.add(root)
                        block_names.append(root)

                log(f"  Total block groups to quantize: {len(block_names)}")
                for bn in block_names:
                    log(f"    {bn}")

                for blk_idx, block_name in enumerate(block_names):
                    t_blk = time.time()
                    log(f"\n  Block group [{blk_idx+1}/{len(block_names)}]: {block_name}")

                    # Match layers belonging to this block group.
                    # Use exact prefix with dot boundary to avoid e.g.
                    # "model.pos_embed.proj" matching "model.pos_embed.proj_out"
                    subset_layers = {
                        name: module
                        for name, module in qnn.named_modules()
                        if (name == block_name or name.startswith(block_name + "."))
                        and isinstance(module, QuantModule)
                        and not module.ignore_reconstruction
                    }
                    if not subset_layers:
                        log(f"    No QuantModules found — skipping")
                        continue

                    log(f"    Layers in this block: {len(subset_layers)}")
                    for ln in subset_layers:
                        log(f"      {ln}")

                    # BUG FIX: guard against zero samples before fasterquant
                    gptqs = {}
                    handles = []
                    for name, module in subset_layers.items():
                        g = GPTQ(module)
                        g.quantizer = module.weight_quantizer
                        gptqs[name] = g

                        def add_batch(n):
                            def tmp(_, inp, out):
                                # BUG FIX: guard nsamples==0 in gptq.add_batch
                                gptqs[n].add_batch(inp[0].data, out.data)
                            return tmp
                        handles.append(module.register_forward_hook(add_batch(name)))

                    # Cap Hessian samples at gptq_cali_n.
                    # GPTQ is not sensitive to sample count past ~128-256.
                    # Using the full cali set (e.g. 5120) makes collection
                    # 20-40x slower with negligible accuracy improvement.
                    # n_total = min(cali_xs.size(0), opt.gptq_cali_n)
                    n_total = cali_xs.size(0)
                    bs = opt.cali_batch_size
                    n_passes = (n_total + bs - 1) // bs
                    log(f"    Collecting Hessians: {n_total}/{cali_xs.size(0)} samples  "
                        f"batch={bs}  ({n_passes} passes)  "
                        f"[capped by --gptq_cali_n={opt.gptq_cali_n}]")
                    # CRITICAL: disable weight quantization during Hessian
                    # collection. GPTQ only needs the input activations — the
                    # hooks capture inp before any weight interaction. Running
                    # with weight_quant=True and group_quant=True makes every
                    # forward pass ~270x slower (543s for 2 samples vs ~2s fp16).
                    # With weight_quant=False, 16 passes × ~2s = ~32s total.
                    qnn.set_quant_state(weight_quant=False, act_quant=False)
                    t_hess = time.time()
                    qnn.eval()
                    with torch.no_grad():
                        for i in range(0, n_total, bs):
                            _xs = cali_xs[i:i+bs].to(device)
                            _ts = cali_ts[i:i+bs].to(device)
                            _cs = cali_cs[i:i+bs].to(device)
                            _ = qnn(
                                _xs,
                                timestep=_ts,
                                encoder_hidden_states=_cs,
                                added_cond_kwargs=pixart_alpha_aca_dict(_xs)
                            )
                            del _xs, _ts, _cs
                    # Restore weight quant state for fasterquant
                    qnn.set_quant_state(weight_quant=True, act_quant=False)
                    log(f"    Hessian collection done in {elapsed(t_hess)}")

                    for h in handles:
                        h.remove()

                    # BUG FIX: check nsamples before quantizing
                    for name, gptq in gptqs.items():
                        if gptq.nsamples == 0:
                            log(f"    WARNING: {name} collected 0 samples — skipping", "warning")
                            continue
                        log(f"    Quantizing {name}  (nsamples={gptq.nsamples})...")
                        t_q = time.time()
                        gptq.fasterquant(
                            percdamp=opt.gptq_percdamp,
                            groupsize=opt.gptq_groupsize,
                            blocksize=opt.gptq_blocksize
                        )
                        gptq.free()
                        log(f"      done in {elapsed(t_q)}")

                    log(f"    Block group done in {elapsed(t_blk)}")
                    log_gpu(f"after block {blk_idx+1}")
                    torch.cuda.empty_cache()
                    gc.collect()

                log(f"\n  GPTQ calibration complete in {elapsed(t_gptq)}")
                log_gpu("after full GPTQ")

            # ── 3c. BRECQ ─────────────────────────────────────────────────────
            if not opt.resume_w:
                if not opt.gptq:
                    log("")
                    log("─" * 60)
                    log(f"STEP 3c  BRECQ weight calibration "
                        f"(iters={opt.cali_iters}, batch={opt.cali_batch_size})...")
                    t_brecq = time.time()
                    recon_model(qnn)
                    log(f"  BRECQ complete in {elapsed(t_brecq)}")
                    log_gpu("after BRECQ")

                qnn.set_quant_state(weight_quant=True, act_quant=False)

                log("")
                log("  Saving weight-quantized checkpoint...")
                t_save = time.time()
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

                ckpt_path = os.path.join(outpath, "ckpt_wq.pth")
                torch.save(qnn.state_dict(), ckpt_path)
                ckpt_mb = os.path.getsize(ckpt_path) / 1024**2
                log(f"  Saved weight checkpoint: {ckpt_path} ({ckpt_mb:.1f} MB) in {elapsed(t_save)}")

            # ── 3d. Activation calibration ────────────────────────────────────
            if opt.quant_act and opt.disable_online_act_quant:
                # FIX 3: loud warning if cali_iters_a too low
                if opt.cali_iters_a < 100:
                    log(f"  WARNING: cali_iters_a={opt.cali_iters_a} — "
                        "activation calibration effectively disabled!", "warning")

                log("")
                log("─" * 60)
                log(f"STEP 3d  Activation calibration (iters={opt.cali_iters_a})...")
                t_act = time.time()

                qnn.set_quant_state(True, True)
                log("  Init activation quantizers with 16 random samples...")
                with torch.no_grad():
                    inds = np.random.choice(cali_xs.shape[0], 16, replace=False)
                    log(f"    sample indices: {inds.tolist()}")
                    _xs = cali_xs[inds].to(device)
                    _ts = cali_ts[inds].to(device)
                    _cs = cali_cs[inds].to(device)
                    _ = qnn(
                        _xs, timestep=_ts, encoder_hidden_states=_cs,
                        added_cond_kwargs=pixart_alpha_aca_dict(_xs),
                    )
                    del _xs, _ts, _cs
                log("  Activation init done")

                if opt.running_stat:
                    log("  Running statistics collection for activation quantizers...")
                    t_rs = time.time()
                    inds = np.arange(cali_xs.shape[0])
                    np.random.shuffle(inds)
                    qnn.set_running_stat(True, opt.rs_sm_only)
                    n_batches = int(cali_xs.size(0) / 16)
                    for i in trange(n_batches, desc="running_stat"):
                        sl = inds[i*16:(i+1)*16]
                        _xs = cali_xs[sl].to(device)
                        _ts = cali_ts[sl].to(device)
                        _cs = cali_cs[sl].to(device)
                        _ = qnn(
                            _xs, timestep=_ts, encoder_hidden_states=_cs,
                            added_cond_kwargs=pixart_alpha_aca_dict(_xs),
                        )
                        del _xs, _ts, _cs
                    qnn.set_running_stat(False, opt.rs_sm_only)
                    log(f"  Running stat done in {elapsed(t_rs)}")

                kwargs_a = dict(
                    cali_data=cali_data, batch_size=opt.cali_batch_size,
                    iters=opt.cali_iters_a, act_quant=True,
                    opt_mode='mse', lr=opt.cali_lr, p=opt.cali_p,
                    cond=opt.cond, sequential=opt.sequential_a, multi_gpu=False)
                recon_model(qnn)
                qnn.set_quant_state(weight_quant=True, act_quant=True)
                log(f"  Activation calibration done in {elapsed(t_act)}")
                log_gpu("after act calibration")

            elif opt.quant_act:
                log("  Online activation calibration mode (no LSQ)")
                qnn.set_quant_state(weight_quant=True, act_quant=True)

            # Final checkpoint save
            log("")
            log("  Saving final quantized model checkpoint...")
            t_save = time.time()
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

        # After GPTQ, weights are already quantized in-place — disable the
        # weight quantizer to avoid double-quantization and shape mismatches.
        if opt.gptq:
            log("  GPTQ path: disabling weight quantizer for inference "
                "(weights already quantized in-place)")
            qnn.set_quant_state(weight_quant=False, act_quant=opt.quant_act)

        qnn = qnn.to(device=device, dtype=torch.float16)
        # org_weight / org_bias are plain tensor attrs (not Parameters/buffers)
        # so .to() doesn't move them. Move them explicitly.
        for m in qnn.modules():
            if isinstance(m, QuantModule):
                if hasattr(m, 'org_weight') and m.org_weight is not None:
                    m.org_weight = m.org_weight.to(device=device, dtype=torch.float16)
                if hasattr(m, 'org_bias') and m.org_bias is not None:
                    m.org_bias = m.org_bias.to(device=device, dtype=torch.float16)
        model.transformer = qnn
        log(f"  Quantized transformer attached to pipeline in {elapsed(t0)}")
        log_gpu("after full PTQ")

    # ── 4. Sampling ───────────────────────────────────────────────────────────
    log("")
    log("─" * 60)
    log("STEP 4  Sampling images...")
    t_sample = time.time()

    # Older diffusers versions have a broken _execution_device property that
    # raises AttributeError. Override it on the class with a working version.
    try:
        _ = model._execution_device
    except AttributeError:
        type(model)._execution_device = property(lambda self: device)
        log(f"  Patched _execution_device → {device}")

    sample_path = os.path.join(outpath, sp)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    sampling_file = os.path.join(outpath, "sampling_config.yaml")
    sampling_conf = vars(opt)
    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)

    if opt.verbose:
        log("Transformer architecture:")
        logger.info(model.transformer)

    batch_size = opt.n_samples
    if opt.prompt is None:
        n_total = pes.shape[0]
        n_batches = (n_total + batch_size - 1) // batch_size
        log(f"  Generating {n_total} images in {n_batches} batches of {batch_size}...")
        for i in tqdm(range(0, pes.shape[0], batch_size), desc="sampling"):
            torch.manual_seed(42)
            prompt_embeds = pes[i:i + batch_size].to(device)
            t_img = time.time()
            image = model(
                prompt=None, negative_prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=pams[i:i + batch_size].to(device),
                negative_prompt_embeds=npe.expand(prompt_embeds.shape[0], -1, -1),
                negative_prompt_attention_mask=npam.expand(prompt_embeds.shape[0], -1),
                height=opt.res, width=opt.res
            ).images
            for j, img in enumerate(image):
                img.save(os.path.join(sample_path, f"{i+j}.png"))
            if i == 0:
                log(f"  First batch generated in {elapsed(t_img)}  "
                    f"→  {os.path.join(sample_path, '0.png')}")
    else:
        torch.manual_seed(42)
        log(f"  Prompt: '{opt.prompt}'")
        image = model(prompt=[opt.prompt], height=opt.res, width=opt.res).images
        for j, img in enumerate(image):
            out_path = os.path.join(sample_path, f"{j}.png")
            img.save(out_path)
            log(f"  Saved: {out_path}")

    log(f"  Sampling complete in {elapsed(t_sample)}")
    log("")
    log("=" * 70)
    log(f"ALL DONE  —  total wall time: {elapsed(SCRIPT_START)}")
    log(f"Outputs at: {outpath}")
    log("=" * 70)


if __name__ == "__main__":
    main()
