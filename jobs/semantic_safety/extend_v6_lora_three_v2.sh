#!/usr/bin/env bash
set -euo pipefail
cd /NHNHOME/26msit001_B/BASE/edge_ai_lab/minseong/VLA-Safety
export PYTHONPATH="$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
openpi/.venv/bin/python - <<'PY'
import time
from pathlib import Path
while not Path('runs/semantic_motor_v6_lora_v2/three_training_runs_complete.json').exists():
    time.sleep(20)
PY
for scope in ae vlm all; do
    out="runs/semantic_motor_v6_lora_v2/train_${scope}_extension"
    openpi/.venv/bin/python scripts/semantic_safety/train_v6_scoped_lora.py \
        --manifest data/semantic_motor_v6_lora_v2/combined_train_manifest.jsonl \
        --out "$out" --scope "$scope" --steps 2000 --batch-size 16 \
        --learning-rate 0.00002 --seed 20271111 --save-every 1000 \
        --resume "runs/semantic_motor_v6_lora_v2/train_$scope/adapter_step_06000.pt" \
        > "runs/semantic_motor_v6_lora_v2/train_${scope}_extension.log" 2>&1
done
openpi/.venv/bin/python - <<'PY'
import json,time
from pathlib import Path
p=Path('runs/semantic_motor_v6_lora_v2/three_extensions_complete.json')
p.write_text(json.dumps(dict(complete=True,additional_steps_per_scope=2000,total_steps_per_scope=8000,
                            finished_unix=time.time()),indent=2)+'\n')
PY
