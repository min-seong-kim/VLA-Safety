#!/bin/bash
# One GPU lane of jobs/backbone_eval.sbatch. Usage: backbone_eval_lane.sh <runner: openpi|ct|dry> <model> [<model> ...]
# Env (exported by the sbatch): ROOT, IMG, CT_ENV, EXTRA, SLURM_JOB_ID.
# For each model: 16-item smoke pass (all validation checks) -> only if it exits 0, the full extraction.
set -o pipefail
RUNNER="$1"; shift
EXTRA="${EXTRA:-}"

run_py() {  # <model> <args...>
  local m="$1"; shift
  local env="$CT_ENV"
  case "$m" in paligemma*) env="${CT_ENV/PYTHONPATH=$ROOT\/envs\/ctpkgs/PYTHONPATH=$ROOT\/envs\/ctpkgs_pg:$ROOT\/envs\/ctpkgs}" ;; esac  # transformers 4.44 for PaliGemma
  case "$RUNNER" in
    openpi) python scripts/backbone_eval/extract_and_score.py --model "$m" --batch 16 "$@" ;;
    ct) singularity exec --nv -B "$ROOT" --env "$env" "$IMG" python3 scripts/backbone_eval/extract_and_score.py --model "$m" --batch 8 "$@" ;;
    dry) echo "DRY $RUNNER $m $*" ;;
    *) echo "unknown runner $RUNNER" >&2; return 2 ;;
  esac
}

if [ "$RUNNER" = openpi ]; then
  source scripts/hangang-env.sh
  export OPENBLAS_NUM_THREADS=8
fi

for m in "$@"; do
  if run_py "$m" --smoke 16 --imgshuf 8 > "logs/beval-$m-smoke-${SLURM_JOB_ID:-local}.log" 2>&1; then
    echo "SMOKE OK $m"
    case "$EXTRA" in *--smoke*) continue ;; esac
    # shellcheck disable=SC2086
    run_py "$m" $EXTRA > "logs/beval-$m-${SLURM_JOB_ID:-local}.log" 2>&1 || echo "LANE $m FAILED (logs/beval-$m-${SLURM_JOB_ID:-local}.log)"
  else
    echo "SMOKE FAILED $m (logs/beval-$m-smoke-${SLURM_JOB_ID:-local}.log)"
  fi
done
echo "LANE DONE $RUNNER $*"
