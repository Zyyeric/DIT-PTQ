from diffusers import PixArtAlphaPipeline
import torch
import os
import time
import datetime
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from pytorch_lightning import seed_everything

seed_everything(42)

# ── tuneable knobs (override via env vars or edit directly) ──────────────────
NUM_IMAGES    = int(os.environ.get("NUM_IMAGES",    128))
VAE_BATCH     = int(os.environ.get("VAE_BATCH",      16))  # increase if VRAM allows
TEXT_BATCH    = int(os.environ.get("TEXT_BATCH",    128))  # lower if OOM
DESIRED_STEPS = int(os.environ.get("DESIRED_STEPS",  20))
OUTPUT_FILE   = os.environ.get("OUTPUT_FILE", "pixart_calib_brecq.pt")
# ─────────────────────────────────────────────────────────────────────────────

# ── logging helpers ───────────────────────────────────────────────────────────
def now():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"[{now()}] {msg}", flush=True)

def log_gpu():
    if torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated()  / 1024**3
        reserv = torch.cuda.memory_reserved()   / 1024**3
        total  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"  GPU memory  →  allocated: {alloc:.2f} GB  |  reserved: {reserv:.2f} GB  |  total: {total:.2f} GB")
    else:
        log("  GPU memory  →  running on CPU, no VRAM stats")

def tensor_stats(t, name):
    log(f"  {name}: shape={tuple(t.shape)}  dtype={t.dtype}  "
        f"min={t.float().min():.4f}  max={t.float().max():.4f}  "
        f"mean={t.float().mean():.4f}  std={t.float().std():.4f}")

def elapsed(start):
    return f"{time.time() - start:.1f}s"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_START = time.time()

if __name__ == "__main__":

    log("=" * 70)
    log("PixArt-Alpha Calibration Data Generator")
    log("=" * 70)
    log(f"Config  →  NUM_IMAGES={NUM_IMAGES}  VAE_BATCH={VAE_BATCH}  "
        f"TEXT_BATCH={TEXT_BATCH}  DESIRED_STEPS={DESIRED_STEPS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32
    log(f"Device  →  {device}  |  dtype={dtype}")
    if torch.cuda.is_available():
        log(f"GPU     →  {torch.cuda.get_device_name(0)}")
    log_gpu()

    # ── 1. Load pipeline ──────────────────────────────────────────────────────
    log("")
    log("─" * 60)
    log("STEP 1/5  Loading PixArt-Alpha pipeline...")
    t0 = time.time()

    pipeline = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-512x512", torch_dtype=dtype
    )
    if device.type == "cuda":
        try:
            pipeline.enable_model_cpu_offload()
            log("  CPU offload enabled (model moves to GPU on demand)")
        except Exception as e:
            log(f"  CPU offload failed ({e}), moving full pipeline to GPU")
            pipeline = pipeline.to(device)
    else:
        pipeline = pipeline.to(device)

    vae             = pipeline.vae
    noise_scheduler = pipeline.scheduler
    log(f"  Pipeline loaded in {elapsed(t0)}")
    log(f"  Scheduler  →  {noise_scheduler.__class__.__name__}")
    log(f"  VAE config →  scaling_factor={vae.config.scaling_factor}")
    log(f"  Scheduler num_train_timesteps  →  {noise_scheduler.config.num_train_timesteps}")
    log_gpu()

    # ── 2. COCO dataset ───────────────────────────────────────────────────────
    log("")
    log("─" * 60)
    log("STEP 2/5  Setting up COCO dataset...")
    t0 = time.time()

    coco_transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    coco_root = os.path.expanduser(os.environ.get("COCO_ROOT", "~/datasets/coco"))
    log(f"  COCO root  →  {coco_root}")

    coco_ds = CocoDetection(
        root=os.path.join(coco_root, "train2017"),
        annFile=os.path.join(coco_root, "annotations", "captions_train2017.json"),
        transform=coco_transform,
    )
    log(f"  Full dataset size  →  {len(coco_ds)} images")
    coco_ds.ids = coco_ds.ids[-NUM_IMAGES:]
    log(f"  Using last {NUM_IMAGES} images  →  IDs [{coco_ds.ids[0]} … {coco_ds.ids[-1]}]")

    coco_dl = DataLoader(
        coco_ds,
        batch_size=VAE_BATCH,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=4,
        pin_memory=True,
    )
    log(f"  DataLoader  →  {len(coco_dl)} batches of {VAE_BATCH}")
    log(f"  Dataset ready in {elapsed(t0)}")

    # Set scheduler timesteps NOW and log the real values
    # This maps DESIRED_STEPS indices → actual diffusion timesteps (e.g. 999, 949, ..., 0)
    noise_scheduler.set_timesteps(DESIRED_STEPS)
    real_timesteps = noise_scheduler.timesteps  # shape: [DESIRED_STEPS], values e.g. [999, 949, ...]
    log(f"  Scheduler real timesteps  →  {real_timesteps.tolist()}")

    # ── 3. VAE encode all images ──────────────────────────────────────────────
    log("")
    log("─" * 60)
    log(f"STEP 3/5  VAE-encoding {NUM_IMAGES} images (batch={VAE_BATCH})...")
    t0 = time.time()

    latents_list    = []
    all_prompts_raw = []

    with torch.no_grad():
        for batch_idx, img_info in enumerate(coco_dl):
            tb = time.time()
            images = torch.cat([img[0].unsqueeze(0) for img in img_info], dim=0)
            log(f"  Batch {batch_idx+1}/{len(coco_dl)}  "
                f"image tensor shape={tuple(images.shape)}  dtype={images.dtype}")

            lat = (
                vae.encode(images.to(device, dtype=dtype))
                   .latent_dist.sample()
                   .to(dtype)
                * vae.config.scaling_factor
            )
            latents_list.append(lat.cpu())

            prompts = [p[1][0]['caption'] for p in img_info]
            all_prompts_raw.extend(prompts)

            log(f"    latent batch shape={tuple(lat.shape)}  "
                f"min={lat.float().min():.3f}  max={lat.float().max():.3f}  "
                f"({elapsed(tb)})")
            log(f"    sample prompts: {prompts[:2]}")

    latents = torch.cat(latents_list, dim=0)
    noise   = torch.randn_like(latents)

    log(f"  All images encoded in {elapsed(t0)}")
    tensor_stats(latents, "latents (all)")
    tensor_stats(noise,   "noise   (all)")
    log(f"  Total prompts collected: {len(all_prompts_raw)}")
    log_gpu()

    # ── 4. Text embedding ─────────────────────────────────────────────────────
    log("")
    log("─" * 60)
    log(f"STEP 4/5  T5-XXL encoding {NUM_IMAGES} prompts (batch={TEXT_BATCH})...")
    t0 = time.time()

    pe_chunks, upe_chunks = [], []
    num_text_batches = (NUM_IMAGES + TEXT_BATCH - 1) // TEXT_BATCH

    with torch.no_grad():
        for b_idx, start in enumerate(range(0, NUM_IMAGES, TEXT_BATCH)):
            tb  = time.time()
            end = min(start + TEXT_BATCH, NUM_IMAGES)
            batch_prompts = all_prompts_raw[start:end]
            log(f"  Text batch {b_idx+1}/{num_text_batches}  "
                f"prompts [{start}:{end}]  (count={len(batch_prompts)})")

            pe, _, upe, _ = pipeline.encode_prompt(batch_prompts)
            pe_chunks.append(pe.cpu())
            upe_chunks.append(upe.cpu())

            log(f"    cond embed shape={tuple(pe.shape)}  "
                f"uncond embed shape={tuple(upe.shape)}  ({elapsed(tb)})")
            log(f"    cond   stats → min={pe.float().min():.4f}  "
                f"max={pe.float().max():.4f}  mean={pe.float().mean():.4f}")
            log(f"    uncond stats → min={upe.float().min():.4f}  "
                f"max={upe.float().max():.4f}  mean={upe.float().mean():.4f}")
            log_gpu()

    cached_pe     = torch.cat(pe_chunks,  dim=0)
    cached_uncond = torch.cat(upe_chunks, dim=0)

    log(f"  All prompts encoded in {elapsed(t0)}")
    tensor_stats(cached_pe,     "cached_pe    ")
    tensor_stats(cached_uncond, "cached_uncond")

    # ── 5. Noisy latents across timesteps ─────────────────────────────────────
    log("")
    log("─" * 60)
    log(f"STEP 5/5  Building noisy latents for {DESIRED_STEPS} timesteps "
        f"(vectorized over {NUM_IMAGES} images)...")
    log(f"  NOTE: using real scheduler timesteps, std should decrease from ~1.0 → ~{latents.std():.3f}")
    t0 = time.time()

    xs         = torch.empty(DESIRED_STEPS, NUM_IMAGES, *latents.shape[1:])
    time_calib = torch.zeros(DESIRED_STEPS, NUM_IMAGES, dtype=torch.long)
    log(f"  Allocated xs tensor: shape={tuple(xs.shape)}  "
        f"size={xs.numel()*xs.element_size()/1024**2:.1f} MB")

    with torch.no_grad():
        lat_gpu   = latents.to(device, dtype=dtype)
        noise_gpu = noise.to(device, dtype=dtype)
        log(f"  Moved latents + noise to {device}")
        log_gpu()

        for i, actual_ts in enumerate(real_timesteps):
            tb    = time.time()
            # actual_ts is the real diffusion timestep (e.g. 999, 949, ..., 0)
            t_vec = torch.full((NUM_IMAGES,), actual_ts.item(), dtype=torch.long, device=device)
            noisy = noise_scheduler.add_noise(lat_gpu, noise_gpu, t_vec)
            noisy = noise_scheduler.scale_model_input(noisy, actual_ts)

            xs[i]         = noisy.cpu()
            time_calib[i] = actual_ts.item()  # store real timestep value, not index

            log(f"  idx={i:2d}  actual_ts={actual_ts.item():4d}  [{i+1:2d}/{DESIRED_STEPS}]  "
                f"noisy shape={tuple(noisy.shape)}  "
                f"min={noisy.float().min():.3f}  max={noisy.float().max():.3f}  "
                f"std={noisy.float().std():.3f}  ({elapsed(tb)})")

    log(f"  All timesteps done in {elapsed(t0)}")
    log(f"  Sanity: xs[0] (ts={time_calib[0,0].item()}) std={xs[0].float().std():.3f}  ← should be ~1.0 (pure noise)")
    log(f"  Sanity: xs[-1] (ts={time_calib[-1,0].item()}) std={xs[-1].float().std():.3f}  ← should be ~{latents.std():.3f} (clean latent)")
    log_gpu()

    # ── 6. Expand embeddings ──────────────────────────────────────────────────
    log("")
    log("─" * 60)
    log("Expanding embeddings to [T, N, seq, dim]...")
    all_pe     = cached_pe.unsqueeze(0).expand(DESIRED_STEPS, -1, -1, -1).clone()
    all_uncond = cached_uncond.unsqueeze(0).expand(DESIRED_STEPS, -1, -1, -1).clone()
    log(f"  all_pe     shape={tuple(all_pe.shape)}  "
        f"size={all_pe.numel()*all_pe.element_size()/1024**2:.1f} MB")
    log(f"  all_uncond shape={tuple(all_uncond.shape)}  "
        f"size={all_uncond.numel()*all_uncond.element_size()/1024**2:.1f} MB")

    # ── 7. Sanity checks ──────────────────────────────────────────────────────
    log("")
    log("─" * 60)
    log("Sanity checks...")

    assert all_uncond.shape == all_pe.shape, \
        f"Shape mismatch: all_pe={all_pe.shape} vs all_uncond={all_uncond.shape}"
    log("  ✓  all_pe and all_uncond shapes match")

    assert torch.equal(all_uncond[0][0], all_uncond[0][1]), \
        "Uncond embeddings should be identical across images (empty prompt)"
    log("  ✓  Uncond embeddings are identical across images (expected for empty prompt)")

    assert not torch.equal(all_pe[0][0], all_pe[0][1]), \
        "Cond embeddings should differ across images"
    log("  ✓  Cond embeddings differ across images (different captions)")

    # Verify std decreases monotonically from high → low timestep
    stds = [xs[i].float().std().item() for i in range(DESIRED_STEPS)]
    log(f"  Noisy latent std across timesteps (should decrease): "
        f"{[f'{s:.3f}' for s in stds]}")
    assert stds[0] > stds[-1], \
        f"std should decrease from ts[0]={stds[0]:.3f} to ts[-1]={stds[-1]:.3f}"
    log("  ✓  std decreases from high timestep → low timestep (correct noise schedule)")

    log(f"  xs         shape={tuple(xs.shape)}")
    log(f"  time_calib shape={tuple(time_calib.shape)}  "
        f"unique ts={time_calib[:,0].unique().tolist()}")
    log(f"  all_pe     shape={tuple(all_pe.shape)}")
    log(f"  all_uncond shape={tuple(all_uncond.shape)}")

    # ── 8. Save ───────────────────────────────────────────────────────────────
    log("")
    log("─" * 60)
    out_path = OUTPUT_FILE
    log(f"Saving checkpoint to {out_path}...")
    t0 = time.time()

    payload = {'xs': xs, 'ts': time_calib, 'cs': all_pe, 'ucs': all_uncond}
    total_mb = sum(v.numel() * v.element_size() for v in payload.values()) / 1024**2
    log(f"  Total tensor data to save: {total_mb:.1f} MB")
    torch.save(payload, out_path)

    file_mb = os.path.getsize(out_path) / 1024**2
    log(f"  Saved in {elapsed(t0)}  |  file size on disk: {file_mb:.1f} MB")

    log("")
    log("=" * 70)
    log(f"ALL DONE  —  total wall time: {elapsed(SCRIPT_START)}")
    log("=" * 70)

