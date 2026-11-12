#!/usr/bin/env bash
set -euo pipefail
cd /NHNHOME/26msit001_B/BASE/edge_ai_lab/minseong/VLA-Safety
export PYTHONPATH="$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
openpi/.venv/bin/python - <<'PY'
import json,time
from pathlib import Path
data=Path('data/semantic_motor_v6_lora_v2')
run=Path('runs/semantic_motor_v6_lora_v2')
while not (run/'experiment_protocol.json').exists():time.sleep(10)
audit=json.loads((data/'training_data_audit.json').read_text())
assert audit['collection_complete'] and audit['conditions_with_demonstrations']==63
assert audit['all_accepted_demos_replayed'] and audit['cache_arrays_bit_identical']
assert json.loads((data/'rehearsal_merge_audit.json').read_text())['passed']
assert json.loads((run/'pipeline_preflight.json').read_text())['passed']
PY
for scope in ae vlm all; do
    out="runs/semantic_motor_v6_lora_v2/train_$scope"
    if [[ -f "$out/freeze_audit.json" && -f "$out/adapter_step_06000.pt" ]]; then
        openpi/.venv/bin/python - "$out" "$scope" <<'PY'
import json,sys
from pathlib import Path
audit=json.loads((Path(sys.argv[1])/'freeze_audit.json').read_text())
assert audit['all_original_weights_unchanged']
assert sys.argv[2]!='ae' or audit['fixed_input_vlm_hidden_exact']
PY
        continue
    fi
    openpi/.venv/bin/python scripts/semantic_safety/train_v6_scoped_lora.py \
        --manifest data/semantic_motor_v6_lora_v2/combined_train_manifest.jsonl \
        --out "$out" --scope "$scope" --steps 6000 --batch-size 16 \
        --learning-rate 0.0002 --seed 20261111 --save-every 2000 \
        > "runs/semantic_motor_v6_lora_v2/train_$scope.log" 2>&1
done
openpi/.venv/bin/python - <<'PY'
import sys,time
from pathlib import Path
sys.path.insert(0,'scripts/semantic_safety')
from rollout_common import write_json
write_json(Path('runs/semantic_motor_v6_lora_v2/three_training_runs_complete.json'),
           dict(complete=True,steps_per_scope=6000,dataset_version=2,finished_unix=time.time()))
PY
