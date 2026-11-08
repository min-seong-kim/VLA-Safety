#!/bin/bash
#SBATCH --job-name=vla-checkpoint-eval
#SBATCH --partition=natStr
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
ROOT=$(pwd -P)
test "$ROOT" = /scratch/t512a07/minseong/VLA-Safety
module purge
module load singularity/4.5.0
set -u
RUN="$ROOT/runs/backbone_eval_v2_checkpoints"
IMG=/apps/common/kisti-container/images/pytorch-25.03-py3-aarch64.sif
export ROOT RUN IMG
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$ROOT/.cache/xdg" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN" logs
printf 'job=%s host=%s run=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$RUN"
nvidia-smi -L
models=(pi0_libero pi05_libero openvla_libero_spatial openvla_oft_goal)
pids=()
for model in "${models[@]}"; do
    srun --mpi=none --ntasks=1 --gpus-per-task=1 --exact --cpus-per-task=16 \
        bash jobs/checkpoint_eval_lane.sh "$model" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
if [ "$failed" != 0 ]; then
    echo 'One or more extraction lanes failed. Inspect per-model logs; no success summary is produced.' >&2
    exit 1
fi
cpu=(srun --mpi=none --ntasks=1 --cpus-per-task=16 singularity exec -B "$ROOT"
     --env "PYTHONPATH=$ROOT/envs/ctpkgs" "$IMG" python3)
"${cpu[@]}" scripts/backbone_eval/train_linear_probe.py --models "${models[@]}" --run_dir "$RUN" \
    > "logs/checkpoint-probes-$SLURM_JOB_ID.log" 2>&1
"${cpu[@]}" scripts/backbone_eval/summarize_eval.py --models "${models[@]}" --run_dir "$RUN"
srun --mpi=none --ntasks=1 --cpus-per-task=4 "$ROOT/openpi/.venv/bin/python" \
    scripts/backbone_eval/diagnostics.py --models "${models[@]}" --run_dir "$RUN" --unique-item-ids \
    > "logs/checkpoint-diagnostics-$SLURM_JOB_ID.log" 2>&1
"${cpu[@]}" scripts/backbone_eval/compare_checkpoints.py --run_dir "$RUN"
echo CHECKPOINT_EVALUATION_COMPLETE
