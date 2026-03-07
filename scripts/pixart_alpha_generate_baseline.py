#!/usr/bin/env python3
"""Generate an image with PixArt-Alpha (no quantization) for baseline comparison."""
import argparse
import torch
from diffusers import PixArtAlphaPipeline


def main():
    parser = argparse.ArgumentParser(description="PixArt-Alpha baseline image generation")
    parser.add_argument("--prompt", type=str, default="A cat sitting on a beach at sunset, photorealistic")
    parser.add_argument("--model_id", type=str, default="PixArt-alpha/PixArt-XL-2-512x512")
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument("--output", type=str, default="baseline_output.png")
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Fix for older diffusers versions
    try:
        _ = pipe._execution_device
    except AttributeError:
        type(pipe)._execution_device = property(lambda self: torch.device("cuda"))

    print(f"Generating: '{args.prompt}'")
    torch.manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        height=args.res,
        width=args.res,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg_scale,
    ).images[0]

    image.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
