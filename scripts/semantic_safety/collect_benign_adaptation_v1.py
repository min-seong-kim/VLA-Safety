"""Calibrate benign skills, collect physical demonstrations, then freeze audited data."""

import argparse
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import h5py
import numpy as np

import benign_adaptation_v1 as b
from rollout_common import write_json

READY=False


def setup():
    global READY
    if READY:return
    b.configure()
    import expansion_v5_teacher as teacher
    teacher.make_environment=b.make_environment
    teacher.Teacher=b.teacher_class()
    READY=True


def task(request):
    setup()
    from expansion_v5_teacher import replay, rollout
    index,mode,count,profile_limit=request
    spec=b.make_spec(index,"calibration",0)
    profile_path=b.DATA/"collection/profiles"/f"c{index:03d}.json"
    sig=b.signature()
    if mode=="calibrate":
        cached=json.loads(profile_path.read_text()) if profile_path.exists() else {}
        if cached.get("signature")==sig and cached.get("success"):return cached
        proposals=list(b.candidate_profiles(spec))[:profile_limit]
        if cached.get("profile"):
            proposals=[cached["profile"]]+[p for p in proposals if p!=cached["profile"]]
        trials=[];chosen=None
        for profile in proposals:
            try:result=rollout(spec,profile,horizon=540)
            except Exception:
                result=dict(passed=False,error=traceback.format_exc())
                trials.append(result)
                break
            trials.append(result)
            if result["passed"]:chosen=profile;break
        result=dict(condition=index,success=chosen is not None,profile=chosen,signature=sig,trials=trials)
        if chosen:
            folder=b.DATA/"collection/calibration"/f"c{index:03d}"
            recorded=rollout(spec,chosen,folder,224,horizon=540)
            check=replay(folder)
            assert recorded["passed"] and check["accepted"],(index,recorded,check)
            result["independent_replay_verified"]=True
            # Publish the upright images actually used by this curriculum/probe.
            from PIL import Image
            with h5py.File(folder/"trajectory.h5") as data:
                for name,offset in [("initial",0),("final",-1)]:
                    Image.fromarray(data["observations/main_rgb_raw"][offset][::-1]).save(folder/(name+"_upright.png"))
        write_json(profile_path,result)
        print(json.dumps(dict(condition=index,success=result["success"],trials=len(trials))),flush=True)
        return result
    calibration=json.loads(profile_path.read_text())
    assert calibration["signature"]==sig and calibration["success"],(index,"calibration required")
    accepted=[]
    for repeat in range(count*6):
        spec=b.make_spec(index,"train",repeat)
        folder=b.DATA/"collection/train"/f"c{index:03d}"/spec["item_id"]
        receipt=folder/"receipt.json"
        if receipt.exists():
            row=json.loads(receipt.read_text());assert row["signature"]==sig
        else:
            proposals=[calibration["profile"]]
            proposals.extend(p for p in list(b.candidate_profiles(spec))[:12] if p not in proposals)
            chosen=None;attempts=[]
            for profile in proposals:
                result=rollout(spec,profile,horizon=540)
                attempts.append(dict(passed=result["passed"],stage=result["stage"],final_metric=result["final_metric"]))
                if result["passed"]:chosen=profile;break
            row=dict(item_id=spec["item_id"],condition_index=index,split="train",data_type="benign_motor_v1",
                     accepted=False,instruction=spec["instruction"],signature=sig,motor_skill=spec["motor_skill"],
                     actor_identity=spec["actor_identity"],target_id=spec["target_id"],verb=spec["verb"])
            if chosen:
                recorded=rollout(spec,chosen,folder,224,horizon=540)
                check=replay(folder)
                assert recorded["passed"] and check["accepted"],(index,repeat,check)
                row.update(accepted=True,steps=recorded["steps"],path=str((folder/"trajectory.h5").relative_to(b.ROOT)),
                           h5_sha256=check["trajectory_sha256"],replay=str((folder/"replay_validation.json").relative_to(b.ROOT)))
            write_json(folder/"attempts.json",attempts);write_json(receipt,row)
        if row["accepted"]:accepted.append(row)
        if len(accepted)>=count:break
    result=dict(condition=index,accepted=accepted,complete=len(accepted)>=count,target=count)
    write_json(b.DATA/"collection/summaries"/f"c{index:03d}.json",result)
    print(json.dumps(dict(condition=index,accepted=len(accepted),target=count)),flush=True)
    return result


def layout(spec):
    return tuple(round(x,8) for p in spec["props"] if p["role"] in {"actor","target"} for x in p["xy"])


def audit():
    definitions=b.conditions();sig=b.signature();rows=[];train_layouts=set();counts=Counter();steps=0
    for spec in definitions:
        summary=json.loads((b.DATA/"collection/summaries"/f"c{spec['condition_index']:03d}.json").read_text())
        assert summary["complete"] and len(summary["accepted"])>=32
        for row in summary["accepted"]:
            assert row["signature"]==sig and row["accepted"]
            proof=json.loads((b.ROOT/row["replay"]).read_text())
            assert proof["accepted"] and proof["trajectory_sha256"]==row["h5_sha256"]==b.sha(b.ROOT/row["path"])
            with h5py.File(b.ROOT/row["path"]) as data:
                scene=json.loads(data.attrs["spec_json"])
                assert scene["split"]=="train" and scene["instruction"]==row["instruction"]
                assert len(data["observations/state"])==len(data["actions"])+1==row["steps"]+1
                for key in ["actions","observations/state","observations/sim_state"]:assert np.isfinite(data[key][:]).all()
                actions=data["actions"][:]
                assert actions.shape[1:]==(7,) and np.abs(actions).max()<=1.
                for camera in ["main_rgb_raw","wrist_rgb_raw"]:
                    image=data["observations/"+camera]
                    assert image.shape==(row["steps"]+1,224,224,3) and image.dtype==np.uint8
                    assert all(np.std(image[t])>5 for t in [0,len(image)//2,len(image)-1])
                train_layouts.add(layout(scene))
            rows.append(row);counts[row["condition_index"]]+=1;steps+=row["steps"]
    for split in ["validation","test"]:
        assert all(layout(r) not in train_layouts for r in b.read_jsonl(b.DATA/"holdouts"/(split+".jsonl")))
    native_source=b.ROOT/"data/semantic_expansion_v5_lora_v1"
    native=b.read_jsonl(native_source/"native_rehearsal_manifest.jsonl")
    prior=json.loads((native_source/"native_rehearsal_audit.json").read_text())
    assert prior["passed"] and prior["all_successful_episodes_independently_replayed"]
    for row in native:
        assert row["initial_state_id"] in [2,3,4,5]
        assert b.sha(b.ROOT/row["training_cache"])==row["cache_sha256"]
        assert json.loads((b.ROOT/row["cache_audit"]).read_text())["passed"]
    combined=b.DATA/"combined_train_manifest.jsonl"
    if combined.exists():assert b.read_jsonl(combined)==rows+native
    else:b.write_jsonl(combined,rows+native)
    b.write_jsonl(b.DATA/"train_manifest.jsonl",rows)
    report=dict(passed=True,motor_episodes=len(rows),motor_control_steps=steps,conditions=len(definitions),
        minimum_episodes_per_condition=min(counts.values()),native_episodes=len(native),native_tasks=prior["tasks"],
        native_sampling_fraction=.1,native_replay_audit_reused=True,
        all_demonstrations_independently_replayed=True,no_train_holdout_layout_overlap=True,
        finite_bounded_actions=True,observation_action_alignment_verified=True,physical_signature=sig,
        manifest_sha256=b.sha(combined),probe_manifest_sha256=b.sha(b.BASELINE/"manifest.json"))
    write_json(b.DATA/"training_data_audit.json",report)
    write_json(b.RUN/"experiment_data_ready.json",{"full_data_audit_passed":True,**report})
    review()
    return report


def review():
    import html
    cards=[]
    for row in b.conditions():
        index=row["condition_index"]
        folder=f"collection/calibration/c{index:03d}"
        profile=b.DATA/"collection/profiles"/f"c{index:03d}.json"
        result=json.loads(profile.read_text()) if profile.exists() else {}
        cards.append(f'<article><h3>{index:03d} · {html.escape(row["verb"])} · {html.escape(row["actor_id"])}</h3>'
                     f'<p>{html.escape(row["instruction"])}</p><p>시연 검증: {result.get("success",False)}</p>'
                     f'<img loading="lazy" src="{folder}/initial_upright.png" alt="initial">'
                     f'<img loading="lazy" src="{folder}/final_upright.png" alt="final">'
                     f'<p>motor 기준: {html.escape(row["motor_skill"])} / 실제 절단·세척 효과는 모사하지 않습니다.</p></article>')
    text='<!doctype html><meta charset="utf-8"><title>Benign adaptation curriculum</title><style>body{font:16px sans-serif;max-width:1100px;margin:30px auto;background:#f4f6f8}article{background:white;padding:20px;margin:18px 0;border-radius:10px}img{width:224px;margin-right:12px}</style><h1>안전한 물체·조작 적응 시연</h1><p>기존 probe와 물체는 공유하지만 대상·목표 조합은 분리한 학습 과제입니다. 두 이미지: 초기 장면 / teacher 종료 장면. 위험 라벨을 학습하지 않습니다.</p>'+''.join(cards)
    (b.DATA/"preview.html").write_text(text)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("mode",choices=["prepare","calibrate","collect","audit","review"])
    p.add_argument("--ids");p.add_argument("--workers",type=int,default=4)
    p.add_argument("--count",type=int,default=32);p.add_argument("--profile-limit",type=int,default=80)
    args=p.parse_args()
    if args.mode=="prepare":b.prepare();return
    if args.mode=="audit":print(json.dumps(audit()),flush=True);return
    if args.mode=="review":review();return
    ids=list(map(int,args.ids.split(","))) if args.ids else [r["condition_index"] for r in b.conditions()]
    started=time.time()
    with mp.get_context("spawn").Pool(args.workers,maxtasksperchild=4) as pool:
        result=list(pool.imap_unordered(task,[(i,args.mode,args.count,args.profile_limit) for i in ids]))
    summary=dict(mode=args.mode,conditions=len(ids),seconds=time.time()-started,
                 complete=all(r.get("success",r.get("complete",False)) for r in result),
                 failed=[r["condition"] for r in result if not r.get("success",r.get("complete",False))])
    write_json(b.DATA/"collection"/(args.mode+"_summary.json"),summary)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()
