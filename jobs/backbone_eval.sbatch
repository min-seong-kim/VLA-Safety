#!/bin/bash
#SBATCH --job-name=vla-backbone-eval
#SBATCH --partition=natStr
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# One node, 4 GPUs. GPU0: pi0 then pi05 (openpi JAX venv). GPU1/2/3: openvla / prismatic / openvla_oft (site PyTorch
# container + envs/ctpkgs). Each lane = independent srun step (no --kill-on-bad-exit). Then probes + summary on CPU.
# Usage: sbatch [--export=ALL,EXTRA="--smoke 16"] jobs/backbone_eval.sbatch

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
case "$(pwd -P)" in
  "/scratch/$USER/"*) ;;
  *) exit 2 ;;
esac
module purge
module load singularity/4.5.0
set -u
EXTRA="${EXTRA:-}"
ROOT="$(pwd -P)"
IMG=/apps/common/kisti-container/images/pytorch-25.03-py3-aarch64.sif
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ROOT/.cache/huggingface" XDG_CACHE_HOME="$ROOT/.cache/xdg"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p runs/backbone_eval_v1 logs
printf 'job=%s host=%s extra=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$EXTRA"
nvidia-smi -L

# Every lane (jobs/backbone_eval_lane.sh) first runs a 16-item smoke pass (all validation checks, output under
# runs/backbone_eval_v1/smoke) and only continues to the full extraction when the smoke pass exits 0.
# EXTRA="--smoke 16" makes every lane stop after the smoke pass.
export ROOT IMG EXTRA
export CT_ENV="PYTHONPATH=$ROOT/envs/ctpkgs,PYTHONDONTWRITEBYTECODE=1,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HOME=$HF_HOME,OMP_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8,TOKENIZERS_PARALLELISM=false,PYTHONUNBUFFERED=1"
LANE="srun --mpi=none --ntasks=1 --gpus-per-task=1 --exact --cpus-per-task=16 bash jobs/backbone_eval_lane.sh"
MODELS="pi0 pi05 paligemma paligemma_lb openvla prismatic openvla_oft"   # probes/summary cover every model with features on disk
if [ -n "${ONLY:-}" ]; then           # ONLY="openvla_oft prismatic pi0": re-run lanes; pi* models share one openpi lane
  PI=""; for m in $ONLY; do case "$m" in pi0|pi05) PI="$PI $m" ;; *) $LANE ct "$m" & ;; esac; done
  [ -n "$PI" ] && $LANE openpi $PI &
else
  $LANE openpi pi0 pi05 &        # GPU lane 0: openpi JAX venv
  $LANE ct openvla &             # GPU lane 1: PyTorch container
  $LANE ct prismatic &           # GPU lane 2
  $LANE ct openvla_oft &         # GPU lane 3
fi
wait
echo ALL_EXTRACTION_DONE

# --- probes + summary (CPU, container python has scikit-learn) ------------------------------------------------------
RUN=runs/backbone_eval_v1; case "$EXTRA" in *--smoke*) RUN=runs/backbone_eval_v1/smoke ;; esac
srun --mpi=none --ntasks=1 --cpus-per-task=16 singularity exec -B "$ROOT" --env PYTHONPATH="$ROOT/envs/ctpkgs",OMP_NUM_THREADS=8,OPENBLAS_NUM_THREADS=8 "$IMG" \
  python3 scripts/backbone_eval/train_linear_probe.py --models $MODELS --run_dir "$RUN" \
  > "logs/beval-probes-$SLURM_JOB_ID.log" 2>&1 || echo "PROBES FAILED"
srun --mpi=none --ntasks=1 --cpus-per-task=4 singularity exec -B "$ROOT" --env PYTHONPATH="$ROOT/envs/ctpkgs" "$IMG" \
  python3 scripts/backbone_eval/summarize_eval.py --run_dir "$RUN" >> "logs/beval-probes-$SLURM_JOB_ID.log" 2>&1 || echo "SUMMARY FAILED"
echo ALL_TASKS_DONE
