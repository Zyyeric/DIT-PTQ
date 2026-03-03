#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm__int4_gptq.bash  –  INT4 + GPTQ quantization + generation
#
# Usage:
#   bash slurm__int4_gptq.bash
#   OUTDIR=my_output bash slurm__int4_gptq.bash
# ─────────────────────────────────────────────────────────────────────────────

# ── Configurable from outside ─────────────────────────────────────────────────
OUTDIR="${OUTDIR:-outputs/int4_gptq}"
CALI_DATA="${CALI_DATA:-pixart_calib_brecq.pt}"
# ─────────────────────────────────────────────────────────────────────────────

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=int4_gptq
#SBATCH --output=logs/int4_gptq_%j.out
#SBATCH --error=logs/int4_gptq_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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
echo "  MODE          = INT4 + GPTQ"
echo "  OUTDIR        = ${OUTDIR}"
echo "  CALI_DATA     = ${CALI_DATA}"
echo "=========================================="

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# ── Cache dirs ────────────────────────────────────────────────────────────────
export HF_HOME=/gpfs/projects/e33188/hf_cache
export TRANSFORMERS_CACHE=/gpfs/projects/e33188/hf_cache
export HF_DATASETS_CACHE=/gpfs/projects/e33188/hf_cache
mkdir -p /gpfs/projects/e33188/hf_cache

source .env 2>/dev/null || true

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
echo "Starting INT4 + GPTQ quantization and generation..."

python scripts/pixart_alpha_brecq.py \\
  --ptq \\
  --plms \\
  --cond \\
  --weight_bit 4 \\
  --quant_mode qdiff \\
  --disable_fp_quant \\
  --int_quant_method gptq \\
  --weight_group_size 128 \\
  --gptq_blocksize 128 \\
  --gptq_percdamp 0.01 \\
  --cali_data_path "${CALI_DATA}" \\
  --cali_batch_size 8 \\
  --quant_act \\
  --act_bit 8 \\
  --disable_online_act_quant \\
  --running_stat \\
  --rs_sm_only \\
  --res 512 \\
  --n_samples 1 \\
  --coco_10k \\
  --outdir "${OUTDIR}"

EXIT_CODE=\$?

# ── Post-run summary ──────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Finished at : \$(date)"
echo "Exit code   : \${EXIT_CODE}"
if [ \${EXIT_CODE} -eq 0 ]; then
    echo "Status      : SUCCESS"
    echo "Samples at  : ${OUTDIR}"
else
    echo "Status      : FAILED — check logs/int4_gptq_\${SLURM_JOB_ID}.err"
fi
echo "=========================================="

exit \${EXIT_CODE}
EOF

echo "Submitted INT4 + GPTQ job."
echo "  Logs  →  logs/int4_gptq_<jobid>.out / .err"
echo "  Track →  squeue -u $USER"
