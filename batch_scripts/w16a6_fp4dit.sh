#!/bin/bash
#SBATCH --account=e32695
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:3
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=pixart_w16a6_3gpu
#SBATCH --output=w16a6_3gpu_slurm_outlog.log
#SBATCH --error=w16a6_3gpu_slurm_errors.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mingyezhang2026@u.northwestern.edu

set -euo pipefail

cd /home/cia5572/DIT-PTQ
source .venv/bin/activate

export COCO_ROOT=/scratch/cia5572/datasets/coco
export TORCH_HOME=/scratch/cia5572/torch_cache
export HF_HOME=/scratch/cia5572/hf_cache
export XDG_CACHE_HOME=/scratch/cia5572/.cache
export CUDA_CACHE_PATH=/scratch/cia5572/.nv_cache
export OMP_NUM_THREADS=4

mkdir -p outputs/pixart_w16a6

torchrun --standalone --nnodes=1 --nproc_per_node=3 scripts/pixart_alpha_brecq.py \
  --cond \
  --n_samples 2 \
  --outdir outputs/pixart_w16a6 \
  --ptq \
  --quant_mode qdiff \
  --weight_bit 16 \
  --act_only \
  --act_bit 6 \
  --cali_data_path /scratch/cia5572/pixart_calib_brecq.pt \
  --cali_batch_size 16 \
  --cali_iters 10000 \
  --cali_iters_a 1000 \
  --res 512 \
  --coco_10k \
  --parallelism ddp \
  --parallel_generate
