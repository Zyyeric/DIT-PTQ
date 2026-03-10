import argparse
import datetime
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import functional as TF


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _now():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"[{_now()}] {msg}", flush=True)

def elapsed(t0):
    return f"{time.time() - t0:.1f}s"


def sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def list_images(root: str) -> List[Path]:
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    files = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise RuntimeError(f"No images found under: {root}")
    files.sort(key=sort_key)
    return files


class ImageDataset(Dataset):
    def __init__(self, paths: List[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = TF.pil_to_tensor(img).float() / 255.0
        return arr, path.name


def collate(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    names = [x[1] for x in batch]
    return imgs, names


def get_inception_extractor(device: torch.device):
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, aux_logits=False, transform_input=False).to(device).eval()
    extractor = create_feature_extractor(
        model, return_nodes={"avgpool": "pool", "Mixed_6e": "mixed6e"}
    ).to(device).eval()
    mean = torch.tensor(weights.transforms().mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(weights.transforms().std, device=device).view(1, 3, 1, 1)
    return extractor, mean, std


def sqrt_psd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mat = (mat + mat.T) * 0.5
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    vals = np.sqrt(vals + eps)
    return (vecs * vals) @ vecs.T


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    sigma1_sqrt = sqrt_psd(sigma1)
    middle = sigma1_sqrt @ sigma2 @ sigma1_sqrt
    covmean = sqrt_psd(middle)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
    return max(fid, 0.0)


def running_stats_from_paths(
    paths: List[Path],
    extractor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    ds = ImageDataset(paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    log(f"  Extracting Inception features: {len(paths)} images, {len(dl)} batches (batch_size={batch_size})")
    t0 = time.time()

    sum_pool = None
    sum_outer_pool = None
    n_pool = 0

    sum_spatial = None
    sum_outer_spatial = None
    n_spatial = 0

    with torch.no_grad():
        for batch_idx, (imgs, _) in enumerate(dl):
            if batch_idx % 10 == 0:
                log(f"    Batch {batch_idx+1}/{len(dl)}...")
            imgs = imgs.to(device, non_blocking=True)
            imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
            imgs = (imgs - mean) / std
            feats = extractor(imgs)

            pool = feats["pool"].flatten(1).float().cpu().numpy()  # [B, 2048]
            spatial = feats["mixed6e"].permute(0, 2, 3, 1).reshape(-1, feats["mixed6e"].shape[1]).float().cpu().numpy()

            if sum_pool is None:
                d_pool = pool.shape[1]
                d_spatial = spatial.shape[1]
                sum_pool = np.zeros(d_pool, dtype=np.float64)
                sum_outer_pool = np.zeros((d_pool, d_pool), dtype=np.float64)
                sum_spatial = np.zeros(d_spatial, dtype=np.float64)
                sum_outer_spatial = np.zeros((d_spatial, d_spatial), dtype=np.float64)

            sum_pool += pool.sum(axis=0, dtype=np.float64)
            sum_outer_pool += pool.T @ pool
            n_pool += pool.shape[0]

            sum_spatial += spatial.sum(axis=0, dtype=np.float64)
            sum_outer_spatial += spatial.T @ spatial
            n_spatial += spatial.shape[0]

    mu_pool = sum_pool / n_pool
    cov_pool = (sum_outer_pool - n_pool * np.outer(mu_pool, mu_pool)) / max(n_pool - 1, 1)
    mu_spatial = sum_spatial / n_spatial
    cov_spatial = (sum_outer_spatial - n_spatial * np.outer(mu_spatial, mu_spatial)) / max(n_spatial - 1, 1)
    log(f"  Done in {elapsed(t0)}: n_pool={n_pool}  n_spatial={n_spatial}")
    return mu_pool, cov_pool, mu_spatial, cov_spatial, n_pool, n_spatial


def load_prompts(
    num_images: int,
    caption_mode: str,
    captions_json: str = None,
    prompt_file: str = None,
    prompt: str = None,
) -> List[str]:
    if prompt is not None:
        return [prompt] * num_images

    if prompt_file is not None:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = [x.strip() for x in f.readlines() if x.strip()]
        if len(prompts) < num_images:
            raise RuntimeError(f"Prompt file has {len(prompts)} prompts, but {num_images} images were found.")
        return prompts[:num_images]

    if captions_json is None:
        raise RuntimeError(
            "For CLIP-score, provide one of: --prompt, --prompt_file, or --captions_json with --caption_mode."
        )

    with open(captions_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    captions = [x["caption"] for x in data["annotations"]]

    if caption_mode == "coco_10k":
        prompts = captions[:10000]
    elif caption_mode == "coco_9k":
        prompts = captions[1000:10000]
    elif caption_mode == "coco_1k":
        prompts = captions[:1000]
    else:
        raise ValueError(f"Unsupported caption_mode: {caption_mode}")

    if len(prompts) < num_images:
        raise RuntimeError(f"Resolved {len(prompts)} prompts for mode {caption_mode}, but {num_images} images were found.")
    return prompts[:num_images]


def compute_clip_score(
    image_paths: List[Path],
    prompts: List[str],
    device: torch.device,
    batch_size: int,
) -> Tuple[float, float]:
    import clip  # openai/CLIP

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    n_batches = (len(image_paths) + batch_size - 1) // batch_size
    log(f"  Computing CLIP scores: {len(image_paths)} images, {n_batches} batches")
    t0 = time.time()
    all_cos = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_num = i // batch_size + 1
            if batch_num % 10 == 0:
                log(f"    CLIP batch {batch_num}/{n_batches}...")
            chunk_paths = image_paths[i:i + batch_size]
            chunk_prompts = prompts[i:i + batch_size]

            imgs = []
            for p in chunk_paths:
                with Image.open(p) as im:
                    imgs.append(preprocess(im.convert("RGB")))
            image_tensor = torch.stack(imgs, dim=0).to(device)
            text_tokens = clip.tokenize(chunk_prompts, truncate=True).to(device)

            img_feat = model.encode_image(image_tensor)
            txt_feat = model.encode_text(text_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            cos = (img_feat * txt_feat).sum(dim=-1)
            all_cos.append(cos.detach().cpu())

    cos_all = torch.cat(all_cos, dim=0).float()
    mean_cos = float(cos_all.mean().item())
    clip_score = float(max(mean_cos, 0.0) * 100.0)
    log(f"  CLIP done in {elapsed(t0)}: mean_cos={mean_cos:.4f}  score={clip_score:.2f}")
    return mean_cos, clip_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated images with FID, sFID, and CLIP-score.")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory containing generated images.")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory containing real reference images.")
    parser.add_argument("--captions_json", type=str, default=None, help="COCO captions JSON (e.g., captions_val2017.json).")
    parser.add_argument(
        "--caption_mode",
        type=str,
        default="coco_10k",
        choices=["coco_10k", "coco_9k", "coco_1k"],
        help="How to slice captions_json for prompt-image matching.",
    )
    parser.add_argument("--prompt_file", type=str, default=None, help="Plain text file: one prompt per line.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt repeated for all generated images.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--clip_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_gen_images", type=int, default=None)
    parser.add_argument("--max_real_images", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", type=str, default=None, help="Optional output json path.")
    args = parser.parse_args()

    device = torch.device(args.device)

    log("=" * 60)
    log("Eval Metrics")
    log("=" * 60)
    log(f"  gen_dir       = {args.gen_dir}")
    log(f"  real_dir      = {args.real_dir}")
    log(f"  caption_mode  = {args.caption_mode}")
    log(f"  device        = {device}")
    if torch.cuda.is_available():
        log(f"  GPU           = {torch.cuda.get_device_name(0)}")

    t_total = time.time()

    gen_images = list_images(args.gen_dir)
    real_images = list_images(args.real_dir)
    log(f"  Found {len(gen_images)} generated images, {len(real_images)} real images")
    if args.max_gen_images is not None:
        gen_images = gen_images[: args.max_gen_images]
        log(f"  Capped generated images to {len(gen_images)}")
    if args.max_real_images is not None:
        real_images = real_images[: args.max_real_images]
        log(f"  Capped real images to {len(real_images)}")

    if len(gen_images) < 2 or len(real_images) < 2:
        raise RuntimeError("Need at least 2 generated and 2 real images to compute FID/sFID.")

    log("")
    log("─" * 40)
    log("Loading Inception-v3...")
    t0 = time.time()
    extractor, mean, std = get_inception_extractor(device)
    log(f"  Inception loaded in {elapsed(t0)}")

    log("")
    log("─" * 40)
    log("Extracting features from GENERATED images...")
    mu_g, cov_g, mu_gs, cov_gs, n_g, ns_g = running_stats_from_paths(
        gen_images, extractor, mean, std, device, args.batch_size, args.num_workers
    )
    log("")
    log("Extracting features from REAL images...")
    mu_r, cov_r, mu_rs, cov_rs, n_r, ns_r = running_stats_from_paths(
        real_images, extractor, mean, std, device, args.batch_size, args.num_workers
    )

    log("")
    log("─" * 40)
    log("Computing FID & sFID...")
    fid = frechet_distance(mu_g, cov_g, mu_r, cov_r)
    sfid = frechet_distance(mu_gs, cov_gs, mu_rs, cov_rs)
    log(f"  FID  = {fid:.4f}")
    log(f"  sFID = {sfid:.4f}")

    log("")
    log("─" * 40)
    log("Loading prompts for CLIP scoring...")
    prompts = load_prompts(
        num_images=len(gen_images),
        caption_mode=args.caption_mode,
        captions_json=args.captions_json,
        prompt_file=args.prompt_file,
        prompt=args.prompt,
    )
    log(f"  Loaded {len(prompts)} prompts (mode={args.caption_mode})")
    log(f"  First prompt: {prompts[0][:80]}...")

    log("")
    log("─" * 40)
    log("Computing CLIP score...")
    mean_cos, clip_score = compute_clip_score(
        image_paths=gen_images, prompts=prompts, device=device, batch_size=args.clip_batch_size
    )

    result = {
        "num_generated_images": n_g,
        "num_real_images": n_r,
        "num_generated_spatial_vectors": ns_g,
        "num_real_spatial_vectors": ns_r,
        "FID": fid,
        "sFID": sfid,
        "CLIP_mean_cosine": mean_cos,
        "CLIP_score_x100": clip_score,
    }

    log("")
    log("=" * 60)
    log("RESULTS")
    log("=" * 60)
    print(json.dumps(result, indent=2))
    if args.save_json is not None:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(f"  Saved to {out}")
    log(f"Total wall time: {elapsed(t_total)}")


if __name__ == "__main__":
    main()

