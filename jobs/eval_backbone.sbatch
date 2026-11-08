#!/bin/bash
#SBATCH --job-name=vla-safety-eval
#SBATCH --partition=natStr
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# One node, 4 GPUs, one checkpoint per GPU (independent processes, not DDP).
# Usage:  sbatch [--time=..] --export=ALL,CKPTS="pi0_base pi0_libero pi05_base pi05_libero",EXTRA="--limit 16" jobs/eval_backbone.sbatch

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
case "$(pwd -P)" in
  "/scratch/$USER/"*) ;;
  *) exit 2 ;;
esac
module purge
source scripts/hangang-env.sh
set -u
: "${CKPTS:?set CKPTS=\"pi0_base pi05_libero ...\"}"
EXTRA="${EXTRA:-}"
read -r -a ckpt_arr <<< "$CKPTS"
printf 'job=%s host=%s ckpts=%s extra=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$CKPTS" "$EXTRA"

# Each task picks its checkpoint by SLURM_PROCID and sees exactly one GPU (--gpus-per-task=1).
# No --kill-on-bad-exit: the four evaluations are independent, one failure must not cancel the others.
srun --mpi=none --ntasks="${#ckpt_arr[@]}" --gpus-per-task=1 bash -c '
  read -r -a arr <<< "$CKPTS"
  name="${arr[$SLURM_PROCID]}"
  case "$name" in pi05*) flag=--pi05 ;; *) flag= ;; esac
  echo "task=$SLURM_PROCID name=$name gpus=$(nvidia-smi -L | wc -l)"
  exec python scripts/eval_backbone.py --name "$name" --ckpt_dir "checkpoints/$name" $flag $EXTRA \
      > "logs/eval-${name}-${SLURM_JOB_ID}.log" 2>&1
'
echo ALL_TASKS_DONE
