"""Counterfactual mask checks, paired-task provenance and exact single-modality label ambiguity."""

from collections import Counter, defaultdict
import hashlib

import numpy as np

from ae_output_nullspace_pilot import inputs, read
from modality_ablation import COMMON, REFERENCE, RUN, SITES, forward, load_policy, physical
from rollout_common import ROOT, write_json


def validate(name, rows):
    import torch

    policy = load_policy(name)
    net = policy._model
    group = rows[:12]
    values, batch, _ = inputs(policy, group)
    seed = read(RUN / "protocol.json")["noise_seed"]
    noise = np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
    noise = torch.as_tensor(np.repeat(noise[None], len(group), axis=0), device="cuda")
    alternatives = dict(text=batch.replace(tokenized_prompt=batch.tokenized_prompt.flip(0),
                                          tokenized_prompt_mask=batch.tokenized_prompt_mask.flip(0)),
                        image=batch.replace(images={k:v.flip(0) for k,v in batch.images.items()}))
    assert not torch.equal(batch.images["base_0_rgb"], alternatives["image"].images["base_0_rgb"])
    assert not torch.equal(batch.tokenized_prompt, alternatives["text"].tokenized_prompt)
    checks = {}
    for modality, changed in alternatives.items():
        for condition in ("intact", "global_" + modality):
            standard, a = forward(net, batch, noise, condition, capture=True)
            replaced, b = forward(net, changed, noise, condition, capture=True)
            diff = {site: float(np.max(np.abs(a[site] - b[site]))) for site in SITES}
            diff["physical_action"] = float(np.max(np.abs(physical(policy, values, standard)-physical(policy, values, replaced))))
            checks[modality+"_"+condition] = diff
            if condition == "intact":
                assert all(n > 0 for n in diff.values())
            else:
                assert all(n == 0 for n in diff.values()), (name, condition, diff)
    return checks


def single_modality(rows):
    output = []
    for track in ("A", "B"):
        track_rows = [r for r in rows if r["track"] == track]
        for field in ("instruction", "image_hash"):
            groups = defaultdict(Counter)
            for r in track_rows:
                groups[r[field]][r["label"]] += 1
            output.append(dict(track=track, key=field, n=len(track_rows), distinct=len(groups),
                               mixed_keys=sum(len(v)==2 for v in groups.values()),
                               rows_in_mixed_keys=sum(sum(v.values()) for v in groups.values() if len(v)==2),
                               max_deterministic_correct=sum(max(v.values()) for v in groups.values())))
    return output


if __name__ == "__main__":
    manifest = read(REFERENCE / "manifest.json")
    rows = manifest["rows"]
    states = set()
    for path in {r["observation_path"] for r in rows}:
        with np.load(ROOT / path) as data:
            states.add(hashlib.sha256(data["robot_state"].tobytes()).hexdigest())
    write_json(RUN / "input_audit.json", dict(decisions=len(rows), unique_initial_states=len(states),
                                               label_ambiguity=single_modality(rows)))
    for name in COMMON:
        result = validate(name, rows)
        write_json(RUN / "checks" / (name + ".json"), dict(complete=True, n=12, maxima=result))
        print(name, "passed", flush=True)
