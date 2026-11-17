"""Refit probes on frozen adapted models with exactly the original scenes and folds."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from unittest.mock import patch

import h5py
import numpy as np

import benign_adaptation_v1 as b
import contrast_initial_linear as original
from representation_common import feature_sites
from rollout_common import write_json


def configure():
    original.RUN=b.RUN


def prepare():
    b.RUN.mkdir(parents=True,exist_ok=True)
    manifest=b.BASELINE/"manifest.json"
    destination=b.RUN/"manifest.json"
    if destination.exists():assert destination.read_bytes()==manifest.read_bytes()
    else:destination.write_bytes(manifest.read_bytes())
    protocol=json.loads((b.RUN/"experiment_protocol.json").read_text())
    protocol["probe_protocol"]=json.loads((b.BASELINE/"experiment_protocol.json").read_text())
    protocol["probe_models"]=["base","libero","ae","vlm","all"]
    protocol["baseline_reuse"]="existing frozen features and probes, unchanged; same scenes, inputs and group folds"
    write_json(b.RUN/"experiment_protocol.json",protocol)
    for kind in ["features","probes/A","probes/B"]:
        folder=b.RUN/kind;folder.mkdir(parents=True,exist_ok=True)
        for model in ["base","libero"]:
            source=b.BASELINE/kind/(model+".h5" if kind=="features" else model)
            target=folder/source.name
            if not target.exists():target.symlink_to(source)
    print(json.dumps(dict(inputs=len(json.loads(manifest.read_text())["rows"]),manifest_sha256=b.sha(manifest))),flush=True)


def extract(scope,limit=None):
    configure()
    if limit is not None and limit<12:
        raise ValueError("Use at least 12 smoke inputs to match the frozen reference extraction batch")
    from openpi.policies import policy_config
    from v6_scoped_lora import load
    factory=policy_config.create_trained_policy
    selection=json.loads((b.RUN/"frozen_selection.json").read_text())["selected"][scope]
    path=b.ROOT/selection["adapter"]
    assert b.sha(path)==selection["adapter_sha256"]
    def adapted_factory(*args,**kwargs):
        policy=factory(*args,**kwargs)
        meta=load(policy._model,path)
        assert meta["lora"]["scope"]==scope
        policy._model.requires_grad_(False);policy._model.eval()
        return policy
    with patch("openpi.policies.policy_config.create_trained_policy",new=adapted_factory), patch.object(original,"milestone",lambda *args:None):
        original.extract(scope,limit)
    baseline=b.BASELINE/"features/libero.h5"
    output=b.RUN/("smoke_features" if limit else "features")/(scope+".h5")
    with h5py.File(baseline) as a,h5py.File(output) as other:
        count=len(other["written"])
        assert np.array_equal(a["input_sha256"][:count],other["input_sha256"][:])
        if scope=="ae":
            # BF16 kernels can round differently for different batch sizes. The
            # reference used batches of 12; compare only identically shaped batches
            # in limited smoke runs. Full extraction has the identical final tail.
            matched=count if count==len(a["written"]) else count-count%12
            assert matched>0
            for site,_ in feature_sites():
                if site.startswith("vlm/"):assert np.array_equal(a[site][:matched],other[site][:matched]),site
            write_json(output.parent/"ae_vlm_identity_audit.json",dict(passed=True,
                unchanged_frozen_vlm_features_all_18_layers=True,rows_extracted=count,rows_compared=matched,
                smoke_tail_excluded_from_bit_exact_check=count-matched,reference_batch_size=12,
                inputs_exact_for_all_extracted_rows=True,
                interpretation="AE-only prefix invariance is checked with identical input batch shapes; full extraction compares every row"))
    if not limit:
        from experiment_notifications import notify_progress
        result=json.loads((output.parent/(scope+"_audit.json")).read_text())
        write_json(b.RUN/"milestones"/("features_"+scope+".json"),dict(complete=True,
            title=scope+" 첫 장면 VLM·AE 추출 및 비교 조건 검증 완료",result=result))
        notify_progress(b.RUN)


def job(args):
    configure()
    return original.probe_job(args)


def probes(workers):
    jobs=[(scope,track,site) for scope in ["ae","vlm","all"] for track in ["A","B"] for site,_ in feature_sites()]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(job,item) for item in jobs]
        for i,future in enumerate(as_completed(futures)):
            future.result();write_json(b.RUN/"probe_progress.json",dict(completed=i+1,total=len(jobs)))
    write_json(b.RUN/"milestones/probes.json",dict(complete=True,title="안전한 조작 적응 세 모델의 linear probe 학습·평가 완료",
               result=dict(sites=78,models=3,tracks=2,selected_fold_classifiers=len(jobs)*5)))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("command",choices=["prepare","extract","probe"])
    p.add_argument("--scope",choices=["ae","vlm","all"]);p.add_argument("--limit",type=int)
    p.add_argument("--workers",type=int,default=8);args=p.parse_args()
    if args.command=="prepare":prepare()
    elif args.command=="extract":extract(args.scope,args.limit)
    else:probes(args.workers)


if __name__=="__main__":main()
