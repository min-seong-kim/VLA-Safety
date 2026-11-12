"""Replay native rehearsal and freeze one audited manifest for all three LoRA scopes."""

import argparse
from collections import Counter
import csv
import json
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np

from expansion_v5_motor import DATA, RUN, ROOT, conditions, load_jsonl, sha, write_jsonl
from evaluate_expansion_v5_lora import native_environment
from expansion_v5_data_quality import audit_episode, write_training_review
from rollout_common import write_json


def native_one(summary):
    source=RUN/"native_rehearsal_collection/episodes"/summary["item_id"]/"trajectory.h5"
    folder=DATA/"native_rehearsal"/summary["item_id"]
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"receipt.json").exists():
        row=json.loads((folder/"receipt.json").read_text())
        assert row["source_sha256"]==sha(source)
        return row
    with h5py.File(source) as raw:
        item=json.loads(raw.attrs["item_json"])
        assert item["initial_state_id"] in range(2,6) and summary["success"] and not summary["initial_success"]
        actions=raw["actions/executed"][:]
        states=raw["observations/sim_state"][:]
        wrapper,_,instruction=native_environment(item,0)
        try:
            env=wrapper.env
            error=float(np.max(np.abs(env.sim.get_state().flatten()-states[0])))
            for i,action in enumerate(actions):
                env.step(action)
                error=max(error,float(np.max(np.abs(env.sim.get_state().flatten()-states[i+1]))))
            success=bool(env._check_success())
        finally:
            wrapper.close()
        assert error<1e-6 and success,(item,error,success)
        starts=raw["chunk_start_steps"][:]
        valid=np.flatnonzero(starts+10<=len(actions))
        assert len(valid)>=1
        arrays={
            "observations/main_rgb_raw":raw["policy_inputs/main_rgb_raw"][:][valid],
            "observations/wrist_rgb_raw":raw["policy_inputs/wrist_rgb_raw"][:][valid],
            "observations/state":raw["policy_inputs/state"][:][valid],
            "action_chunks":np.stack([actions[start:start+10] for start in starts[valid]]),
            "source_control_steps":starts[valid]}
        assert instruction==raw.attrs["instruction"]
    with h5py.File(folder/"inputs.h5","w") as output:
        output.attrs["spec_json"]=json.dumps({**item,"instruction":instruction,"split":"train"})
        for key,value in arrays.items():
            assert np.isfinite(value).all()
            output.create_dataset(key,data=value,compression="lzf",shuffle=True,chunks=(1,*value.shape[1:]))
    with h5py.File(folder/"inputs.h5") as check:
        assert all(np.array_equal(check[k][:],v) for k,v in arrays.items())
    audit=dict(passed=True,independent_replay=True,max_sim_state_abs_error=error,success=success,
        arrays_exactly_match_policy_inputs_and_executed_actions=True,source_sha256=sha(source),
        input_sha256=sha(folder/"inputs.h5"),full_chunks_only=True,evaluation_initial_states_excluded=[0,1,7])
    write_json(folder/"audit.json",audit)
    row=dict(item_id=item["item_id"],data_type="native_rehearsal",split="train",accepted=True,
        suite=item["suite"],task_id=item["task_id"],initial_state_id=item["initial_state_id"],instruction=instruction,
        path=str(source.relative_to(ROOT)),training_cache=str((folder/"inputs.h5").relative_to(ROOT)),
        cache_audit=str((folder/"audit.json").relative_to(ROOT)),chunks=len(valid),steps=len(actions),
        source_sha256=audit["source_sha256"],cache_sha256=audit["input_sha256"])
    write_json(folder/"receipt.json",row)
    return row


def native(workers):
    source=RUN/"native_rehearsal_collection"
    assert json.loads((source/"completion.json").read_text())["complete"]
    assert json.loads((source/"model.json").read_text())["adapter_sha256"] is None
    rows=load_jsonl(source/"episodes.jsonl")
    accepted=[r for r in rows if r["success"] and r["physics_stable"] and r["steps"]>=10]
    with mp.get_context("spawn").Pool(workers) as pool:
        manifest=list(pool.imap_unordered(native_one,accepted))
    manifest.sort(key=lambda r:r["item_id"])
    write_jsonl(DATA/"native_rehearsal_manifest.jsonl",manifest)
    audit=dict(passed=True,attempted=len(rows),episodes=len(manifest),
        all_successful_episodes_independently_replayed=True,
        suites=dict(Counter(r["suite"] for r in manifest)),
        tasks=len({(r["suite"],r["task_id"]) for r in manifest}),
        initial_state_ids=[2,3,4,5],validation_initial_state_ids=[7],test_initial_state_ids=[0,1],
        failed_collection=[r["item_id"] for r in rows if not r["success"]],
        source="successful official pi05_libero policy trajectories, not original human LIBERO demonstrations",
        manifest_sha256=sha(DATA/"native_rehearsal_manifest.jsonl"))
    write_json(DATA/"native_rehearsal_audit.json",audit)
    print(json.dumps(audit),flush=True)


def layout_key(spec):
    return tuple(np.round([*next(p for p in spec["props"] if p["role"]=="actor")["xy"],
        *next(p for p in spec["props"] if p["role"]=="target")["xy"],spec["actor_yaw"]],8))


def freeze(tag):
    from collect_expansion_v5_demonstrations import signature
    collection=DATA/tag
    motor=load_jsonl(collection/"train_manifest.jsonl")
    native_rows=load_jsonl(DATA/"native_rehearsal_manifest.jsonl")
    assert motor and native_rows
    by_condition=Counter()
    quality=[]
    definitions=conditions()
    layouts=set()
    expected_signature=signature()
    assert len({r["item_id"] for r in motor})==len(motor)
    for row in motor:
        assert row["accepted"] and row["split"]=="train"
        assert row["signature"]==expected_signature
        audit=json.loads((ROOT/row["replay"]).read_text())
        assert audit["accepted"] and audit["trajectory_sha256"]==row["h5_sha256"]==sha(ROOT/row["path"])
        with h5py.File(ROOT/row["path"]) as data:
            spec=json.loads(data.attrs["spec_json"])
            assert spec["split"]=="train" and len(data["actions"])==row["steps"]
            assert len(data["observations/main_rgb_raw"])==row["steps"]+1
            assert spec["instruction"]==row["instruction"]
            source=definitions[row["condition_index"]]
            for key in ["source_scene_id","instruction","actor_asset","actor_name","target_name","skill","side"]:
                assert spec[key]==source[key],(row["item_id"],key)
            quality.append(audit_episode(row,data,spec,audit))
            layouts.add(layout_key(spec))
        by_condition[row["condition_index"]]+=1
    assert set(by_condition)==set(range(len(definitions))) and min(by_condition.values())>=32
    for split in ["validation","test"]:
        assert all(layout_key(row) not in layouts for row in load_jsonl(DATA/"holdouts"/(split+".jsonl")))
    for row in native_rows:
        assert row["cache_sha256"]==sha(ROOT/row["training_cache"])
        native_audit=json.loads((ROOT/row["cache_audit"]).read_text())
        assert native_audit["passed"] and native_audit["input_sha256"]==row["cache_sha256"]
    if list(RUN.glob("train_*/config.json")):
        assert load_jsonl(DATA/"combined_train_manifest.jsonl")==motor+native_rows,"Training data is frozen for existing main runs"
    write_jsonl(DATA/"train_manifest.jsonl",motor)
    write_jsonl(DATA/"combined_train_manifest.jsonl",motor+native_rows)
    coverage=[]
    for spec in definitions:
        i=spec["condition_index"]
        coverage.append(dict(condition_index=i,source_scene_id=spec["source_scene_id"],side=spec["side"],
            category=spec["category"],actor=spec["catalog_branch"]["actor"],target=spec["catalog_branch"]["target"],
            verb=spec["verb"],proxy=spec["skill"],instruction=spec["instruction"],demonstrations=by_condition[i]))
    with (DATA/"condition_coverage.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
    audit=dict(passed=True,episodes=len(motor),control_steps=sum(r["steps"] for r in motor),
        conditions_total=len(definitions),conditions_with_demonstrations=len(by_condition),
        minimum_episodes_among_covered_conditions=min(by_condition.values()),
        missing_conditions=[r["source_scene_id"] for r in coverage if not r["demonstrations"]],
        all_accepted_demos_independently_replayed=True,no_exact_train_holdout_layout_overlap=True,
        full_h5_hashes_verified_at_freeze=True,physical_source_signature=expected_signature,
        observation_action_alignment_verified=True,finite_states_and_bounded_actions=True,
        camera_shapes_dtypes_and_three_frames_per_episode_checked=True,
        replayed_motor_success_and_physics_stability_verified=True,
        extra_grasp_search_episodes=sum("recovery_script_sha256" in row for row in motor),
        extra_grasp_search_scope="training layouts only; same physics, teacher controller, motor oracle, and independent replay",
        native_episodes=len(native_rows),native_sampling_fraction=.1,
        manifest_sha256=sha(DATA/"combined_train_manifest.jsonl"),teacher_label_filter_train_only=True,
        all_conditions_retained_in_evaluation=True)
    write_json(DATA/"training_data_audit.json",audit)
    write_jsonl(DATA/"trajectory_quality_audit.jsonl",quality)
    write_training_review(DATA,motor,definitions)
    # Fix category-stratified paired checkpoint selection before any model scores are read.
    validation=load_jsonl(DATA/"holdouts/validation.jsonl")
    groups={}
    for i in range(0,len(definitions),2):
        groups.setdefault(definitions[i]["category"],[]).append(i//2)
    rng=np.random.default_rng(819023)
    selected=set()
    for category,ids in sorted(groups.items()):
        selected.update(int(i) for i in rng.permutation(ids)[:(len(ids)+1)//2])
    selection=[s for s in validation if s["condition_index"]//2 in selected]
    confirmation=[s for s in validation if s["condition_index"]//2 not in selected]
    write_jsonl(DATA/"holdouts/selection.jsonl",selection)
    write_jsonl(DATA/"holdouts/confirmation.jsonl",confirmation)
    protocol=dict(base_checkpoint="checkpoints/pi05_libero_pytorch",scopes=["ae","vlm","all"],
        adapted_models_total=3,
        phase_order=["collect","replay_integrity_quality_and_split_audit","train_three_scopes",
                     "select_on_validation","freeze_checkpoints","evaluate_heldout_motor_and_native","report"],
        manifest_sha256=audit["manifest_sha256"],steps_initial=16000,batch_size=32,micro_batch_size=16,rank=16,alpha=32,
        learning_rate=2e-4,save_steps=[4000,8000,12000,16000],seed=20261112,
        native_fraction=.1,selection_episodes=len(selection),confirmation_episodes=len(confirmation),
        final_test_episodes=len(load_jsonl(DATA/"holdouts/test.jsonl")),
        new_environment_test_scope="v5 safety/unsafe scenarios with unseen layouts; same object and semantic families as adaptation",
        native_test_episodes=80,native_test_suites=["libero_spatial","libero_object","libero_goal","libero_10"],
        native_rehearsal_states=[2,3,4,5],native_validation_states=[7],native_test_states=[0,1],
        selection_rule="Prefer checkpoints within two native-validation successes of baseline, then motor proxy success, native success, earlier step",
        extension_rule="Before test access: if any scope selects 16k and improves motor validation by at least two percentage points over 12k, extend all scopes by 8k at 2e-5",
        identical_samples_and_flow_noise_across_scopes=True,all_original_weights_frozen=True,
        recovery_checkpoints_every_updates=200,
        recovery_preserves_optimizer_samples_noise_and_completed_evaluations=True,
        evaluation_teacher_success_filter=False,semantic_safety_probe_not_performed=True,
        no_claim_of_full_instruction_or_harm_success=True)
    write_json(RUN/"experiment_protocol.json",protocol)
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("mode",choices=["native","freeze"])
    p.add_argument("--workers",type=int,default=12)
    p.add_argument("--tag",default="collection_r1")
    args=p.parse_args()
    native(args.workers) if args.mode=="native" else freeze(args.tag)
