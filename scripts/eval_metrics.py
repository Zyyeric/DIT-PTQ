import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import functional as TF


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


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


def materialize_image_subset(paths: List[Path], target_dir: str) -> None:
    for idx, path in enumerate(paths):
        dst = Path(target_dir) / f"{idx:06d}{path.suffix.lower()}"
        try:
            os.symlink(path.resolve(), dst)
        except OSError:
            shutil.copy2(path, dst)


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
            arr = TF.resize(arr, [299, 299], antialias=True)
        return arr, path.name


def collate(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    names = [x[1] for x in batch]
    return imgs, names


def get_inception_extractor(device: torch.device):
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights).to(device).eval()
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

    sum_pool = None
    sum_outer_pool = None
    n_pool = 0

    sum_spatial = None
    sum_outer_spatial = None
    n_spatial = 0

    with torch.no_grad():
        for imgs, _ in dl:
            imgs = imgs.to(device, non_blocking=True)
            imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
            imgs = (imgs - mean) / std
            feats = extractor(imgs)

            pool = feats["pool"].flatten(1).float().cpu().numpy()  # [B, 2048]
            spatial = feats["mixed6e"].reshape(feats["mixed6e"].shape[0], -1).float().cpu().numpy()

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

    all_cos = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
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
    return mean_cos, clip_score


def compute_image_reward(
    image_paths: List[Path],
    prompts: List[str],
) -> float:
    try:
        import ImageReward as RM
    except ImportError as exc:
        raise RuntimeError(
            "ImageReward is not installed. Install it with `pip install image-reward` "
            "or pass `--skip_imagereward`."
        ) from exc

    model = RM.load("ImageReward-v1.0")
    scores = []
    with torch.no_grad():
        for path, prompt in tqdm(
            list(zip(image_paths, prompts)),
            desc="ImageReward",
            total=len(image_paths),
        ):
            score = model.score(prompt, str(path))
            if isinstance(score, (list, tuple)):
                score = score[0]
            scores.append(float(score))
    return float(np.mean(scores))


def compute_clean_fid_from_paths(gen_images: List[Path], real_images: List[Path], mode: str) -> float:
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise RuntimeError(
            "clean-fid is not installed. Install it with `pip install clean-fid` "
            "or switch to `--fid_backend custom`."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="cleanfid_gen_") as gen_tmp, tempfile.TemporaryDirectory(
        prefix="cleanfid_real_"
    ) as real_tmp:
        materialize_image_subset(gen_images, gen_tmp)
        materialize_image_subset(real_images, real_tmp)
        return float(fid.compute_fid(gen_tmp, real_tmp, mode=mode))


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated images with FID, CLIP-score, and ImageReward.")
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
    parser.add_argument(
        "--fid_backend",
        type=str,
        default="clean-fid",
        choices=["clean-fid", "custom"],
        help="FID implementation. `clean-fid` is the default and recommended setting.",
    )
    parser.add_argument(
        "--clean_fid_mode",
        type=str,
        default="clean",
        choices=["clean", "legacy", "clip"],
        help="clean-fid preprocessing mode when `--fid_backend clean-fid` is used.",
    )
    parser.add_argument(
        "--compute_sfid",
        action="store_true",
        help="Compute the repo-local sFID metric. Disabled by default because it is expensive and not needed for standard eval runs.",
    )
    parser.add_argument(
        "--skip_imagereward",
        action="store_true",
        help="Skip ImageReward scoring. By default ImageReward is computed using the validation prompts.",
    )
    parser.add_argument("--save_json", type=str, default=None, help="Optional output json path.")
    args = parser.parse_args()

    device = torch.device(args.device)

    gen_images = list_images(args.gen_dir)
    real_images = list_images(args.real_dir)
    if args.max_gen_images is not None:
        gen_images = gen_images[: args.max_gen_images]
    if args.max_real_images is not None:
        real_images = real_images[: args.max_real_images]

    if len(gen_images) < 2 or len(real_images) < 2:
        raise RuntimeError("Need at least 2 generated and 2 real images to compute FID.")

    if args.fid_backend == "clean-fid":
        fid = compute_clean_fid_from_paths(gen_images, real_images, mode=args.clean_fid_mode)
        n_g = len(gen_images)
        n_r = len(real_images)
        sfid = None
        ns_g = None
        ns_r = None
    else:
        extractor, mean, std = get_inception_extractor(device)
        mu_g, cov_g, _, _, n_g, _ = running_stats_from_paths(
            gen_images, extractor, mean, std, device, args.batch_size, args.num_workers
        )
        mu_r, cov_r, _, _, n_r, _ = running_stats_from_paths(
            real_images, extractor, mean, std, device, args.batch_size, args.num_workers
        )
        fid = frechet_distance(mu_g, cov_g, mu_r, cov_r)
        sfid = None
        ns_g = None
        ns_r = None

    if args.compute_sfid:
        extractor, mean, std = get_inception_extractor(device)
        _, _, mu_gs, cov_gs, _, ns_g = running_stats_from_paths(
            gen_images, extractor, mean, std, device, args.batch_size, args.num_workers
        )
        _, _, mu_rs, cov_rs, _, ns_r = running_stats_from_paths(
            real_images, extractor, mean, std, device, args.batch_size, args.num_workers
        )
        sfid = frechet_distance(mu_gs, cov_gs, mu_rs, cov_rs)

    prompts = load_prompts(
        num_images=len(gen_images),
        caption_mode=args.caption_mode,
        captions_json=args.captions_json,
        prompt_file=args.prompt_file,
        prompt=args.prompt,
    )
    mean_cos, clip_score = compute_clip_score(
        image_paths=gen_images, prompts=prompts, device=device, batch_size=args.clip_batch_size
    )
    image_reward = None if args.skip_imagereward else compute_image_reward(gen_images, prompts)

    result = {
        "num_generated_images": n_g,
        "num_real_images": n_r,
        "FID": fid,
        "FID_backend": args.fid_backend,
        "CLIP_mean_cosine": mean_cos,
        "CLIP_score_x100": clip_score,
        "ImageReward_mean": image_reward,
    }
    if sfid is not None:
        result["sFID"] = sfid
        result["num_generated_spatial_features"] = ns_g
        result["num_real_spatial_features"] = ns_r

    print(json.dumps(result, indent=2))
    if args.save_json is not None:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
