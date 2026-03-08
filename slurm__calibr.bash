#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_calib.sh  –  Submit PixArt-Alpha calibration data generation job
#
# Usage:
#   bash submit_calib.sh                        # run with defaults
#   COCO_ROOT=/my/coco bash submit_calib.sh     # override COCO path
#   NUM_IMAGES=256 bash submit_calib.sh         # override image count
# ─────────────────────────────────────────────────────────────────────────────

# ── Configurable from outside ─────────────────────────────────────────────────
COCO_ROOT="${COCO_ROOT:-/gpfs/projects/e33188/datasets/coco}"
NUM_IMAGES="${NUM_IMAGES:-128}"
VAE_BATCH="${VAE_BATCH:-16}"
TEXT_BATCH="${TEXT_BATCH:-128}"
DESIRED_STEPS="${DESIRED_STEPS:-20}"
OUTPUT_FILE="${OUTPUT_FILE:-pixart_calib_brecq.pt}"
# ─────────────────────────────────────────────────────────────────────────────

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=pixart_calib
#SBATCH --output=logs/calib_%j.out
#SBATCH --error=logs/calib_%j.err
#SBATCH --time=04:00:00
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
echo "  COCO_ROOT     = ${COCO_ROOT}"
echo "  NUM_IMAGES    = ${NUM_IMAGES}"
echo "  VAE_BATCH     = ${VAE_BATCH}"
echo "  TEXT_BATCH    = ${TEXT_BATCH}"
echo "  DESIRED_STEPS = ${DESIRED_STEPS}"
echo "  OUTPUT_FILE   = ${OUTPUT_FILE}"
echo "=========================================="

# ── Cache dirs ────────────────────────────────────────────────────────────────
export HF_HOME=/gpfs/projects/e33188/hf_cache
export TRANSFORMERS_CACHE=/gpfs/projects/e33188/hf_cache
export HF_DATASETS_CACHE=/gpfs/projects/e33188/hf_cache
mkdir -p /gpfs/projects/e33188/hf_cache

# ── Pass COCO + tuning knobs through to the script ───────────────────────────
export COCO_ROOT="${COCO_ROOT}"
export NUM_IMAGES="${NUM_IMAGES}"
export VAE_BATCH="${VAE_BATCH}"
export TEXT_BATCH="${TEXT_BATCH}"
export DESIRED_STEPS="${DESIRED_STEPS}"
export OUTPUT_FILE="${OUTPUT_FILE}"

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
echo "Starting calibration..."
python scripts/pixart_alpha_calib.py

EXIT_CODE=\$?

# ── Post-run summary ──────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Finished at : \$(date)"
echo "Exit code   : \${EXIT_CODE}"
if [ \${EXIT_CODE} -eq 0 ]; then
    echo "Status      : SUCCESS"
    if [ -f "${OUTPUT_FILE}" ]; then
        SIZE=\$(du -sh "${OUTPUT_FILE}" | cut -f1)
        echo "Output file : ${OUTPUT_FILE} (\${SIZE})"
    fi
else
    echo "Status      : FAILED — check logs/calib_\${SLURM_JOB_ID}.err"
fi
echo "=========================================="

exit \${EXIT_CODE}
EOF

echo "Submitted calibration job."
echo "  Logs  →  logs/calib_<jobid>.out / .err"
echo "  Track →  squeue -u \$USER"
