#!/bin/bash
# Each lane owns one allocated GPU and one model's files.
set -euo pipefail
model=${1:?model required}
case "$model" in
    pi0_libero|pi05_libero)
        source scripts/hangang-env.sh
        runner=("$ROOT/openpi/.venv/bin/python")
        batch=16
        ;;
    openvla|openvla_oft|openvla_libero_spatial|openvla_oft_goal)
        runner=(singularity exec --nv -B "$ROOT" --env "PYTHONPATH=$ROOT/envs/ctpkgs" "$IMG" python3)
        batch=${HF_BATCH:-8}
        ;;
    *) echo "Unsupported model: $model" >&2; exit 2 ;;
esac
datasets=(pope_yn mme_yn sqa_img_mc vqav2_mc coco_capchoice siuo_mcqa vlguard_request)
args=(scripts/backbone_eval/extract_and_score.py --model "$model" --batch "$batch"
      --out "$RUN" --unique-item-ids --strict-validation --datasets "${datasets[@]}")
"${runner[@]}" "${args[@]}" --smoke 16 --imgshuf 8 --overwrite > "logs/checkpoint-$model-smoke-$SLURM_JOB_ID.log" 2>&1
echo "SMOKE_OK $model"
"${runner[@]}" "${args[@]}" > "logs/checkpoint-$model-$SLURM_JOB_ID.log" 2>&1
echo "EXTRACTION_OK $model"
