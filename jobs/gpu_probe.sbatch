#!/bin/bash
#SBATCH --job-name=vla-safety-probe
#SBATCH --partition=natStr
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
case "$(pwd -P)" in
  "/scratch/$USER/"*) ;;
  *) exit 2 ;;
esac
module purge
source scripts/hangang-env.sh
set -u
printf 'job=%s host=%s python=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(command -v python)"
srun --mpi=none --ntasks=1 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun --mpi=none --ntasks=1 --kill-on-bad-exit=1 python scripts/gpu_probe_jax.py
