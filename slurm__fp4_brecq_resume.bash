#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm__fp4_brecq_resume.bash  –  Resume FP4 + BRECQ inference from checkpoint
#
# Usage:
#   bash slurm__fp4_brecq_resume.bash
#   CKPT=outputs/fp4_brecq/<timestamp>/ckpt_wq.pth bash slurm__fp4_brecq_resume.bash
# ─────────────────────────────────────────────────────────────────────────────

OUTDIR="${OUTDIR:-outputs/fp4_brecq}"
CALI_DATA="${CALI_DATA:-pixart_calib_brecq.pt}"
# ── UPDATE this path to match your actual checkpoint ──
CKPT="${CKPT:-outputs/fp4_brecq/<TIMESTAMP>/ckpt_wq.pth}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fp4_brecq_resume
#SBATCH --output=logs/fp4_brecq_resume_%j.out
#SBATCH --error=logs/fp4_brecq_resume_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gengpu
#SBATCH --account=e33188

echo "=========================================="
echo "SLURM Job ID   : \$SLURM_JOB_ID"
echo "Node           : \$SLURMD_NODENAME"
echo "Started at     : \$(date)"
echo "=========================================="
echo "Config:"
echo "  MODE          = FP4 + BRECQ (resume from checkpoint)"
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

echo "Starting FP4 + BRECQ inference (resume from checkpoint)..."

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
  --weight_quant_method mse \\
  --w_clip_ratio 1.0 \\
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
    echo "Status      : FAILED — check logs/fp4_brecq_resume_\${SLURM_JOB_ID}.err"
fi
echo "=========================================="

exit \${EXIT_CODE}
EOF

echo "Submitted FP4 BRECQ resume job."
echo "  Logs  →  logs/fp4_brecq_resume_<jobid>.out / .err"
echo "  Track →  squeue -u $USER"
