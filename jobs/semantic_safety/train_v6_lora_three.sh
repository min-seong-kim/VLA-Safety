#!/usr/bin/env bash
set -euo pipefail
cd /NHNHOME/26msit001_B/BASE/edge_ai_lab/minseong/VLA-Safety
export PYTHONPATH="$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
openpi/.venv/bin/python - <<'PY'
import json,time
from pathlib import Path
root=Path('.')
while True:
    data=root/'data/semantic_motor_v6_lora_v1/training_data_audit.json'
    preflight=root/'runs/semantic_motor_v6_lora_v1/pipeline_preflight.json'
    merged=root/'data/semantic_motor_v6_lora_v1/rehearsal_merge_audit.json'
    if data.exists() and preflight.exists() and merged.exists():
        d=json.loads(data.read_text());p=json.loads(preflight.read_text())
        if d.get('collection_complete') and p.get('passed') and json.loads(merged.read_text()).get('passed'):
            assert d['conditions_with_demonstrations']>=50
            assert d['all_accepted_demos_replayed'] and d['cache_arrays_bit_identical']
            break
    time.sleep(20)
PY
for scope in ae vlm all; do
    out="runs/semantic_motor_v6_lora_v1/train_$scope"
    if [[ -f "$out/freeze_audit.json" && -f "$out/adapter_step_06000.pt" ]]; then
        continue
    fi
    openpi/.venv/bin/python scripts/semantic_safety/train_v6_scoped_lora.py \
        --manifest data/semantic_motor_v6_lora_v1/combined_train_manifest.jsonl \
        --out "$out" --scope "$scope" --steps 6000 --batch-size 16 \
        --learning-rate 0.0002 --seed 20261111 --save-every 2000 \
        > "runs/semantic_motor_v6_lora_v1/train_$scope.log" 2>&1
done
openpi/.venv/bin/python - <<'PY'
import sys,time
sys.path.insert(0,'scripts/semantic_safety')
from rollout_common import write_json
from v6_lora_data import RUN
write_json(RUN/'three_training_runs_complete.json',dict(complete=True,steps_per_scope=6000,finished_unix=time.time()))
PY
