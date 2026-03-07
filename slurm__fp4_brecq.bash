#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_fp4_brecq.sh  –  Submit PixArt-Alpha FP4 BRECQ/AdaRound Job
#
# Usage:
#   bash submit_fp4_brecq.sh                        # run with defaults
#   OUTDIR=my_custom_dir bash submit_fp4_brecq.sh   # override output dir
# ─────────────────────────────────────────────────────────────────────────────

# ── Configurable from outside ─────────────────────────────────────────────────
OUTDIR="${OUTDIR:-outputs/fp4_brecq_run}"
CALI_DATA="${CALI_DATA:-pixart_calib_brecq.pt}"
# ─────────────────────────────────────────────────────────────────────────────

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=fp4_brecq
#SBATCH --output=logs/fp4_brecq_%j.out
#SBATCH --error=logs/fp4_brecq_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gengpu
#SBATCH --account=e33188

# ── Job info ──────────────────────────────────────────────────────────────────
echo "=========================================="
echo "SLURM Job ID   : \$SLURM_JOB_ID"
echo "Node           : \$SLURMD_NODENAME"
echo "Started at     : \$(date)"
echo "=========================================="
echo "Config:"
echo "  OUTDIR        = ${OUTDIR}"
echo "  CALI_DATA     = ${CALI_DATA}"
echo "=========================================="
set -e
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# ── Cache dirs ────────────────────────────────────────────────────────────────
export HF_HOME=/gpfs/projects/e33188/hf_cache
export TRANSFORMERS_CACHE=/gpfs/projects/e33188/hf_cache
export HF_DATASETS_CACHE=/gpfs/projects/e33188/hf_cache
mkdir -p /gpfs/projects/e33188/hf_cache

# ── Pass tuning knobs through to the script ───────────────────────────────────
export OUTDIR="${OUTDIR}"
export CALI_DATA="${CALI_DATA}"

source .env

# ── Activate environment ──────────────────────────────────────────────────────
source /gpfs/projects/e33188/praneeth/DIT-PTQ/.venv/bin/activate

# ── Move to working directory ─────────────────────────────────────────────────
cd "\${SLURM_SUBMIT_DIR}"
mkdir -p logs

# ── GPU health check ──────────────────────────────────────────────────────────
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Starting FP4 BRECQ Calibration and Generation..."

python scripts/pixart_alpha_brecq.py \
  --cond \
  --n_samples 1 \
  --res 512 \
  --coco_10k \
  --outdir "${OUTDIR}" \
  --cali_data_path "${CALI_DATA}" \
  --ptq \
  --quant_mode qdiff \
  --cali_batch_size 16 \
  --weight_bit 4 \
  --act_bit 8 \
  --weight_group_size 128 \
  --quant_weight \
  --weight_quant_method mse \
  --cali_iters 20000 \
  --weight_mantissa_bits 1 \
  --ff_weight_mantissa 0 \
  --quant_act \
  --act_mantissa_bits 4
EXIT_CODE=\$?

# ── Post-run summary ──────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Finished at : \$(date)"
echo "Exit code   : \${EXIT_CODE}"
if [ \${EXIT_CODE} -eq 0 ]; then
    echo "Status      : SUCCESS"
else
    echo "Status      : FAILED — check logs/fp4_brecq_\${SLURM_JOB_ID}.err"
fi
echo "=========================================="

exit \${EXIT_CODE}
EOF

echo "Submitted FP4 BRECQ job."
echo "  Logs  →  logs/fp4_brecq_<jobid>.out / .err"
echo "  Track →  squeue -u \$USER"