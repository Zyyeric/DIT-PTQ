#!/bin/bash
#SBATCH --account=e32695
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=w16a8_eval
#SBATCH --output=/scratch/cia5572/pixart_outputs/w16a8/eval_slurm-%j.out
#SBATCH --error=/scratch/cia5572/pixart_outputs/w16a8/eval_slurm-%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=mingyezhang2026@u.northwestern.edu

set -euo pipefail

export REPO_DIR=/home/cia5572/DIT-PTQ
export RUN_DIR=/scratch/cia5572/pixart_outputs/w16a8/2026-03-08-13-24-43
export COCO_ROOT=/scratch/cia5572/datasets/coco
export TORCH_HOME=/scratch/cia5572/torch_cache
export HF_HOME=/scratch/cia5572/hf_cache
export XDG_CACHE_HOME=/scratch/cia5572/.cache
export CUDA_CACHE_PATH=/scratch/cia5572/.nv_cache

export CAPTIONS_JSON="$REPO_DIR/captions/captions_val2017.json"
export CAPTION_MODE=coco_10k
export SAMPLE_SUBDIR=samples_10k
export FID_BACKEND=clean-fid
export CLEAN_FID_MODE=clean
export BATCH_SIZE=32
export CLIP_BATCH_SIZE=64
export NUM_WORKERS=1
export DEVICE=cuda
export MAX_GEN_IMAGES=6328

cd "$REPO_DIR"
source .venv/bin/activate

GEN_DIR="$RUN_DIR/$SAMPLE_SUBDIR"
REAL_DIR="$COCO_ROOT/val2017"
SAVE_JSON="$RUN_DIR/metrics_${CAPTION_MODE}_759.json"

test -d "$GEN_DIR"
test -d "$REAL_DIR"
test -f "$CAPTIONS_JSON"

python scripts/eval_metrics.py \
  --gen_dir "$GEN_DIR" \
  --real_dir "$REAL_DIR" \
  --captions_json "$CAPTIONS_JSON" \
  --caption_mode "$CAPTION_MODE" \
  --align_real_to_coco_annotations \
  --batch_size "$BATCH_SIZE" \
  --clip_batch_size "$CLIP_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --fid_backend "$FID_BACKEND" \
  --clean_fid_mode "$CLEAN_FID_MODE" \
  --max_gen_images "$MAX_GEN_IMAGES" \
  --save_json "$SAVE_JSON"
