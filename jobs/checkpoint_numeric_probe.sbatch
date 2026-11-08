#!/bin/bash
#SBATCH --job-name=vla-numeric-probe
#SBATCH --partition=natStr
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:20:00
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$ROOT/.cache/xdg" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
pids=()
for model in openvla_libero_spatial openvla_oft_goal openvla openvla_oft; do
    srun --mpi=none --ntasks=1 --gpus-per-task=1 --exact --cpus-per-task=16 \
        singularity exec --nv -B "$ROOT" --env "PYTHONPATH=$ROOT/envs/ctpkgs" "$IMG" \
        python3 scripts/backbone_eval/diagnose_batch_numerics.py --model "$model" --out "$RUN/numerics" \
        > "logs/numerics-$model-$SLURM_JOB_ID.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
test "$failed" = 0
srun --mpi=none --ntasks=1 --cpus-per-task=16 singularity exec -B "$ROOT" \
    --env "PYTHONPATH=$ROOT/envs/ctpkgs" "$IMG" python3 scripts/backbone_eval/train_linear_probe.py \
    --models pi0_libero pi05_libero --run_dir "$RUN" > "logs/checkpoint-pi-probes-$SLURM_JOB_ID.log" 2>&1
echo NUMERIC_DIAGNOSIS_COMPLETE
