#!/bin/bash
#SBATCH --account=e32695
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pixart_w16a8_1gpu
#SBATCH --output=slurm_outlog.log
#SBATCH --error=slurm_errors.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mingyezhang2026@u.northwestern.edu

set -euo pipefail

cd /home/cia5572/DIT-PTQ
source .venv/bin/activate

export COCO_ROOT=/scratch/cia5572/datasets/coco
export TORCH_HOME=/scratch/cia5572/torch_cache
export HF_HOME=/scratch/cia5572/hf_cache
export XDG_CACHE_HOME=/scratch/cia5572/.cache

mkdir -p outputs/pixart_w16a8

python scripts/pixart_alpha_brecq.py \
  --plms \
  --cond \
  --n_samples 1 \
  --outdir outputs/pixart_w16a8 \
  --ptq \
  --quant_mode qdiff \
  --weight_bit 16 \
  --act_only \
  --act_bit 8 \
  --cali_data_path /scratch/cia5572/pixart_calib_brecq.pt \
  --cali_batch_size 16 \
  --cali_iters 10000 \
  --cali_iters_a 1000 \
  --res 512 \
  --coco_10k
