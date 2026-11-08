#!/bin/bash
set -euo pipefail
lane=${1:?lane required}
extra=()
if [ "$STAGE" = smoke ]; then extra=(--smoke 16); fi
case "$lane" in
    openpi)
        source scripts/hangang-env.sh
        for model in pi05 pi05_libero; do
            "$ROOT/openpi/.venv/bin/python" scripts/semantic_safety/extract_features.py \
                --model "$model" --out "$RUN" "${extra[@]}"
        done
        ;;
    oft)
        apptainer exec --nv -B "$ROOT" --env "PYTHONPATH=$ROOT/envs/ctpkgs" "$IMG" \
            python3 scripts/semantic_safety/extract_features.py \
            --model openvla_oft --out "$RUN" "${extra[@]}"
        ;;
    *) exit 2 ;;
esac
