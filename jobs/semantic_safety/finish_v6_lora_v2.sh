#!/usr/bin/env bash
set -euo pipefail
cd /NHNHOME/26msit001_B/BASE/edge_ai_lab/minseong/VLA-Safety
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
run="runs/semantic_motor_v6_lora_v2"
data="data/semantic_motor_v6_lora_v2"
export_path="checkpoints/pi05_libero_v6_lora_v2"
openpi/.venv/bin/python - <<'PY'
import json,time
from pathlib import Path
run=Path('runs/semantic_motor_v6_lora_v2')
while True:
    try:
        ready=all(json.loads((run/name).read_text())['complete'] for name in
                  ['three_extensions_complete.json','validation_extension_complete.json'])
    except (FileNotFoundError,json.JSONDecodeError):
        ready=False
    if ready:
        break
    time.sleep(20)
PY
if [[ ! -f "$run/training_comparison_audit.json" ]]; then
    openpi/.venv/bin/python scripts/semantic_safety/audit_v6_training_comparison.py \
        --run-root "$run" --data-root "$data"
fi
if [[ ! -f "$run/selection_frozen.json" ]]; then
    openpi/.venv/bin/python scripts/semantic_safety/freeze_v6_lora_selection.py \
        --run-root "$run" --data-root "$data"
fi
if [[ ! -f "$run/final_evaluation_complete.json" ]]; then
    openpi/.venv/bin/python scripts/semantic_safety/evaluate_v6_lora_final.py \
        --selection "$run/selection_frozen.json" --run-root "$run" --data-root "$data" \
        --export-root "$export_path"
fi
if [[ ! -f "$run/report_complete.json" ]]; then
    envs/libero-render/bin/python scripts/semantic_safety/report_v6_lora_final.py \
        --run-root "$run" --data-root "$data" --export-root "$export_path"
fi
if [[ ! -f "$run/browser_audit.json" ]]; then
    PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/v4-preview-browser" \
        envs/v4-preview-check/bin/python scripts/semantic_safety/audit_v6_lora_review.py --run-root "$run"
fi
if [[ ! -f "$run/review_bundle_audit.json" ]]; then
    openpi/.venv/bin/python scripts/semantic_safety/package_v6_lora_review.py --run-root "$run"
fi
openpi/.venv/bin/python - <<'PY'
import hashlib,json,shutil,time
from pathlib import Path
root=Path.cwd();run=root/'runs/semantic_motor_v6_lora_v2'
for name in ['training_comparison_audit.json','final_cross_model_input_audit.json',
             'browser_audit.json','review_bundle_audit.json']:
    assert json.loads((run/name).read_text())['passed']
assert json.loads((run/'final_evaluation_complete.json').read_text())['complete']
snapshot=run/'final_source_snapshot';snapshot.mkdir(exist_ok=True)
paths=set((root/'scripts/semantic_safety').glob('*.py'))
paths.update(root/'jobs/semantic_safety'/name for name in
             ['train_v6_lora_three_v2.sh','extend_v6_lora_three_v2.sh','finish_v6_lora_v2.sh'])
paths.update(root/name for name in json.loads((run/'environment.json').read_text())['source_files'])
manifest={}
for path in sorted(paths):
    relative=path.relative_to(root);target=snapshot/relative
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    manifest[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
(snapshot/'manifest.json').write_text(json.dumps(dict(files=manifest,
    note='Training-specific snapshots are also preserved under each training stage directory.'),indent=2)+'\n')
(run/'postprocessing_complete.json').write_text(json.dumps(dict(complete=True,
    finished_unix=time.time(),source_files=len(manifest),review_bundle='review_bundle.zip',
    report='doc/Semantic_Safety_v6_LoRA_Adaptation.md'),indent=2)+'\n')
print('PIPELINE_COMPLETE',len(manifest),'source files archived',flush=True)
PY
