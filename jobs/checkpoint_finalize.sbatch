#!/bin/bash
#SBATCH --job-name=vla-checkpoint-finalize
#SBATCH --partition=natStr
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:10:00
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
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cpu=(srun --mpi=none --ntasks=1 --cpus-per-task=16 singularity exec -B "$ROOT"
     --env "PYTHONPATH=$ROOT/envs/ctpkgs" "$IMG" python3)
if [ "${REPAIR_PI0_ONLY:-0}" != 1 ]; then
    "${cpu[@]}" scripts/backbone_eval/train_linear_probe.py --models openvla_oft --datasets sqa_img_mc \
        --max_iter 10000 --run_dir "$RUN"
fi
"${cpu[@]}" scripts/backbone_eval/train_linear_probe.py --models pi0 --datasets sqa_img_mc \
    --feature_run_dir "$ROOT/runs/backbone_eval_v1" --max_iter 10000 --run_dir "$RUN"
models=(pi0_libero pi05_libero openvla openvla_oft openvla_libero_spatial openvla_oft_goal)
"${cpu[@]}" scripts/backbone_eval/summarize_eval.py --models "${models[@]}" --run_dir "$RUN"
srun --mpi=none --ntasks=1 --cpus-per-task=4 "$ROOT/openpi/.venv/bin/python" \
    scripts/backbone_eval/diagnostics.py --models "${models[@]}" --run_dir "$RUN" --unique-item-ids \
    > "logs/checkpoint-final-diagnostics-$SLURM_JOB_ID.log" 2>&1
"${cpu[@]}" scripts/backbone_eval/compare_checkpoints.py --run_dir "$RUN"
echo CHECKPOINT_EVALUATION_FINALIZED
