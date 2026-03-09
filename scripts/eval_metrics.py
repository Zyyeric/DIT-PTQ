import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


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


def get_coco_subset_spec(caption_mode: str) -> Dict[str, int]:
    if caption_mode == "coco_1k":
        return {"skip": 0, "take": 1000}
    if caption_mode == "coco_9k":
        return {"skip": 1000, "take": 9000}
    if caption_mode == "coco_10k":
        return {"skip": 0, "take": 10000}
    raise ValueError(f"Unsupported caption_mode: {caption_mode}")


def resolve_coco_subset(captions_json: str, caption_mode: str) -> List[Dict[str, object]]:
    with open(captions_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    spec = get_coco_subset_spec(caption_mode)
    image_id_to_name = {img["id"]: img["file_name"] for img in data.get("images", [])}
    unique_records = []
    seen_image_ids = set()

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id in seen_image_ids:
            continue
        seen_image_ids.add(image_id)
        unique_records.append(
            {
                "image_id": image_id,
                "file_name": image_id_to_name.get(image_id, f"{int(image_id):012d}.jpg"),
                "caption": ann["caption"],
            }
        )
        if len(unique_records) >= spec["skip"] + spec["take"]:
            break

    start = spec["skip"]
    end = start + spec["take"]
    selected = unique_records[start:end]
    if len(selected) < spec["take"]:
        raise RuntimeError(
            f"Resolved only {len(selected)} unique COCO images for mode {caption_mode}; expected {spec['take']}."
        )
    return selected


def resolve_coco_reference_images(
    real_dir: str,
    coco_subset: List[Dict[str, object]],
    num_images: int,
) -> List[Path]:
    if len(coco_subset) < num_images:
        raise RuntimeError(
            f"Resolved {len(coco_subset)} unique COCO images, but {num_images} generated images were found."
        )

    real_root = Path(real_dir)
    paths = []
    for record in coco_subset[:num_images]:
        path = real_root / str(record["file_name"])
        if not path.exists():
            raise FileNotFoundError(f"Reference image for image_id={record['image_id']} was not found: {path}")
        paths.append(path)
    return paths


def materialize_image_subset(paths: List[Path], target_dir: str) -> None:
    for idx, path in enumerate(paths):
        dst = Path(target_dir) / f"{idx:06d}{path.suffix.lower()}"
        try:
            os.symlink(path.resolve(), dst)
        except OSError:
            shutil.copy2(path, dst)


def load_prompts(
    num_images: int,
    caption_mode: str,
    captions_json: str = None,
    prompt_file: str = None,
    prompt: str = None,
    coco_subset: List[Dict[str, object]] = None,
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

    if coco_subset is None:
        coco_subset = resolve_coco_subset(captions_json, caption_mode)
    prompts = [str(record["caption"]) for record in coco_subset]
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
            "clean-fid is not installed. Install it with `pip install clean-fid`."
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
    parser.add_argument("--max_gen_images", type=int, default=None)
    parser.add_argument("--max_real_images", type=int, default=None)
    parser.add_argument(
        "--align_real_to_coco_annotations",
        action="store_true",
        help="Resolve real reference images from the COCO unique-image subset so generated index i is compared to image-subset entry i.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--clean_fid_mode",
        type=str,
        default="clean",
        choices=["clean", "legacy", "clip"],
        help="clean-fid preprocessing mode.",
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
    if args.max_gen_images is not None:
        gen_images = gen_images[: args.max_gen_images]

    coco_subset = None
    if args.captions_json is not None:
        coco_subset = resolve_coco_subset(args.captions_json, args.caption_mode)

    if args.align_real_to_coco_annotations:
        if args.captions_json is None:
            raise RuntimeError("--align_real_to_coco_annotations requires --captions_json.")
        real_images = resolve_coco_reference_images(
            real_dir=args.real_dir,
            coco_subset=coco_subset,
            num_images=len(gen_images),
        )
    else:
        real_images = list_images(args.real_dir)
        if args.max_real_images is not None:
            real_images = real_images[: args.max_real_images]

    if len(gen_images) < 2 or len(real_images) < 2:
        raise RuntimeError("Need at least 2 generated and 2 real images to compute FID.")

    fid = compute_clean_fid_from_paths(gen_images, real_images, mode=args.clean_fid_mode)
    n_g = len(gen_images)
    n_r = len(real_images)

    prompts = load_prompts(
        num_images=len(gen_images),
        caption_mode=args.caption_mode,
        captions_json=args.captions_json,
        prompt_file=args.prompt_file,
        prompt=args.prompt,
        coco_subset=coco_subset,
    )
    mean_cos, clip_score = compute_clip_score(
        image_paths=gen_images, prompts=prompts, device=device, batch_size=args.clip_batch_size
    )
    image_reward = None if args.skip_imagereward else compute_image_reward(gen_images, prompts)

    result = {
        "num_generated_images": n_g,
        "num_real_images": n_r,
        "FID": fid,
        "FID_backend": "clean-fid",
        "CLIP_mean_cosine": mean_cos,
        "CLIP_score_x100": clip_score,
        "ImageReward_mean": image_reward,
    }

    print(json.dumps(result, indent=2))
    if args.save_json is not None:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
