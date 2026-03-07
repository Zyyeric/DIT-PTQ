#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm__int4_gptq_resume.bash  –  Resume INT4 + GPTQ inference from checkpoint
#
# Usage:
#   bash slurm__int4_gptq_resume.bash
# ─────────────────────────────────────────────────────────────────────────────

OUTDIR="${OUTDIR:-outputs/int4_gptq_run}"
CALI_DATA="${CALI_DATA:-pixart_calib_brecq.pt}"
CKPT="${CKPT:-outputs/int4_gptq_run/2026-03-02-20-03-04/ckpt_wq.pth}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=int4_gptq_resume
#SBATCH --output=logs/int4_gptq_resume_%j.out
#SBATCH --error=logs/int4_gptq_resume_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gengpu
#SBATCH --account=e33188

echo "=========================================="
echo "SLURM Job ID   : \$SLURM_JOB_ID"
echo "Node           : \$SLURMD_NODENAME"
echo "Started at     : \$(date)"
echo "=========================================="
echo "Config:"
echo "  MODE          = INT4 + GPTQ (resume from checkpoint)"
echo "  OUTDIR        = ${OUTDIR}"
echo "  CALI_DATA     = ${CALI_DATA}"
echo "  CKPT          = ${CKPT}"
echo "=========================================="

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

export HF_HOME=/gpfs/projects/e33188/hf_cache
export TRANSFORMERS_CACHE=/gpfs/projects/e33188/hf_cache
export HF_DATASETS_CACHE=/gpfs/projects/e33188/hf_cache
mkdir -p /gpfs/projects/e33188/hf_cache

source .env 2>/dev/null || true
source /gpfs/projects/e33188/praneeth/DIT-PTQ/.venv/bin/activate
cd "\${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

echo "Starting INT4 + GPTQ inference (resume from checkpoint)..."

python scripts/pixart_alpha_brecq.py \\
  --cond \\
  --n_samples 1 \\
  --res 512 \\
  --coco_10k \\
  --outdir "${OUTDIR}" \\
  --cali_data_path "${CALI_DATA}" \\
  --ptq \\
  --quant_mode qdiff \\
  --cali_batch_size 16 \\
  --weight_bit 4 \\
  --weight_group_size 128 \\
  --disable_fp_quant \\
  --gptq \\
  --gptq_groupsize 128 \\
  --gptq_blocksize 128 \\
  --w_sym \\
  --w_clip_ratio 1.0 \\
  --gptq_cali_n 256 \\
  --resume_w \\
  --cali_ckpt "${CKPT}"

EXIT_CODE=\$?

echo ""
echo "=========================================="
echo "Finished at : \$(date)"
echo "Exit code   : \${EXIT_CODE}"
if [ \${EXIT_CODE} -eq 0 ]; then
    echo "Status      : SUCCESS"
    echo "Samples at  : ${OUTDIR}"
else
    echo "Status      : FAILED — check logs/int4_gptq_resume_\${SLURM_JOB_ID}.err"
fi
echo "=========================================="

exit \${EXIT_CODE}
EOF

echo "Submitted INT4 GPTQ resume job."
echo "  Logs  →  logs/int4_gptq_resume_<jobid>.out / .err"
echo "  Track →  squeue -u $USER"
