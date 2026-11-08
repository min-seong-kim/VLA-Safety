#!/bin/bash
#SBATCH --job-name=vla-robot-semantic
#SBATCH --partition=natStr
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# STAGE=smoke: one small validation allocation; no automatic full submission.
# STAGE=full: use a separately approved allocation after the smoke job COMPLETED.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
ROOT=$(pwd -P)
test "$ROOT" = /scratch/t512a07/minseong/VLA-Safety
module purge
module load apptainer/1.4.5
STAGE=${STAGE:-smoke}
case "$STAGE" in smoke|full) ;; *) exit 2 ;; esac
IMG=/apps/common/kisti-container/images/pytorch-25.03-py3-aarch64.sif
test -r "$IMG"
RUN="$ROOT/runs/robot_semantic_safety_v1/$STAGE-$SLURM_JOB_ID"
mkdir -p "$RUN" logs
export ROOT IMG RUN STAGE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$ROOT/.cache/xdg" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
printf 'job=%s stage=%s run=%s host=%s\n' "$SLURM_JOB_ID" "$STAGE" "$RUN" "$(hostname)"
nvidia-smi -L

pids=()
for lane in openpi oft; do
    srun --mpi=none --ntasks=1 --gpus-per-task=1 --cpus-per-task=72 --exact \
        bash jobs/robot_semantic_safety_lane.sh "$lane" \
        > "logs/robot-semantic-$lane-$SLURM_JOB_ID.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
test "$failed" = 0
if [ "$STAGE" = full ]; then
    srun --mpi=none --ntasks=1 --cpus-per-task=72 --exact \
        apptainer exec -B "$ROOT" --env "PYTHONPATH=$ROOT/envs/ctpkgs" "$IMG" \
        python3 scripts/semantic_safety/analyze_features.py --run "$RUN"
fi
printf 'ROBOT_SEMANTIC_%s_COMPLETE %s\n' "$STAGE" "$RUN"
