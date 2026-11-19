"""Freeze a common pilot protocol and check last-flow null removal on the actual policies."""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import contrast_initial_linear as contrast
from rollout_common import ROOT, write_json


RUN = ROOT / "runs/semantic_safety_ae_output_nullspace_pilot_v1"
REFERENCE = ROOT / "runs/semantic_benign_adaptation_v1"
MODELS = ("base", "libero", "ae", "vlm", "all")
PRE = "velocity_pre/flow_10"
POST = "velocity_post/flow_10"


def read(path):
    return json.loads(path.read_text())


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def prepare():
    from benign_adaptation_v1 import sha

    RUN.mkdir(exist_ok=True)
    src = REFERENCE / "manifest.json"
    assert src.read_bytes() == (contrast.RUN / "manifest.json").read_bytes()
    copy = RUN / "manifest.json"
    if copy.exists():
        assert copy.read_bytes() == src.read_bytes()
    else:
        copy.write_bytes(src.read_bytes())
    manifest = read(copy)
    rows = manifest["rows"]
    # Use the first validation partition with at least three groups (fold 0 has only one).
    fold = next(f for f in manifest["folds"] if len(f["validation_groups"]) >= 3)
    panel, scenes, groups = [], set(), Counter()
    for track in ("A", "B"):
        for label in (0, 1):
            for _ in range(3):
                candidates = [i for i, r in enumerate(rows) if r["group_id"] in fold["validation_groups"]
                              and r["track"] == track and r["label"] == label and r["scene_id"] not in scenes
                              and i < len(rows) // 12 * 12]
                i = min(candidates, key=lambda j: (groups[rows[j]["group_id"]], rows[j]["item_id"]))
                row = rows[i]
                panel.append(dict(index=i, **{k: row[k] for k in ("item_id", "track", "label", "group_id")}))
                groups[row["group_id"]] += 1
                scenes.add(row["scene_id"])
    selections = read(REFERENCE / "frozen_selection.json")["selected"]
    protocol = dict(models=list(MODELS), sites=[PRE, POST], manifest_sha256=sha(src),
        selected_adapters={m: {k: selections[m][k] for k in ("adapter", "adapter_sha256", "cumulative_step")}
                           for m in ("ae", "vlm", "all")},
        dataset="contrast v1, initial scenes", tracks={"A": 336, "B": 121}, group_count=20, folds=5,
        common_readout="LIBERO config and normalization for all five checkpoints; 10 tokens x 32, 10 flows",
        noise_seed=contrast.SEED, intervention_fold=fold["fold"], panel=panel,
        panel_rule="validation only; three per track/label, least represented group then item ID; distinct scenes, batch 12",
        rank_cutoff="float64 eps * max(W.shape) * largest singular value",
        projection="uncentered raw features, 1024-coordinate projected features for independently standardized probes",
        fixed_probe="original pre-projection scaler and classifier of each OOF fold",
        tolerance=dict(algebra_absolute=1e-10, identity_exact=True, native_floor=1e-5, native_relative=2e-3,
                       fp32_roundoff=2e-6, strict_absolute=1e-5),
        tolerance_rule="per sample velocity budget = 1e-5 + 2e-3 * max(1,maxabs(reference velocity)); "
                       "accounts for high-precision-setting TF32 matmul. 32D budget = 0.1 * velocity budget + "
                       "2e-6 * max(1,maxabs(action32)); 7D budget scales this by the LIBERO output-transform gain "
                       "per channel plus 2e-6 * max(1,maxabs(action7)). Also report raw maxima and strict 1e-5 test.",
        model_training=False, steering=False, environment_steps=0)
    path = RUN / "protocol.json"
    if path.exists():
        assert read(path) == protocol
    else:
        write_json(path, protocol)
    write_json(RUN / "reporting/delivery_mode.json", {"mode": "foreground"})
    print("Frozen panel:", len(panel), "inputs,", len(groups), "groups, fold", fold["fold"], flush=True)


def load_model(name, protocol):
    from benign_adaptation_v1 import sha
    from benign_policy_v1 import load_policy
    from openpi.policies import policy_config
    from openpi.training import checkpoints
    from v6_scoped_lora import load

    checkpoint = ROOT / "checkpoints" / ("pi05_base_pytorch" if name == "base" else "pi05_libero_pytorch")
    stats = checkpoints.load_norm_stats(ROOT / "checkpoints/pi05_libero_pytorch/assets", "physical-intelligence/libero")
    factory = policy_config.create_trained_policy
    def common_readout(cfg, default_path, **kwargs):
        return factory(cfg, checkpoint, norm_stats=stats, **kwargs)
    with patch.object(policy_config, "create_trained_policy", new=common_readout):
        policy, _ = load_policy("pi05_libero")
    net = policy._model
    if name in protocol["selected_adapters"]:
        selection = protocol["selected_adapters"][name]
        path = ROOT / selection["adapter"]
        assert sha(path) == selection["adapter_sha256"]
        meta = load(net, path)
        assert meta["lora"]["scope"] == name and meta["step"] == selection["cumulative_step"]
    net.requires_grad_(False)
    net.eval()
    return policy


def inputs(policy, rows):
    from openpi_client import image_tools
    from motor_lora import tensor_batch

    values, hashes = [], []
    for row in rows:
        with np.load(ROOT / row["observation_path"]) as data:
            value = policy._input_transform({
                "observation/image": image_tools.resize_with_pad(data["agentview_rgb"], 224, 224),
                "observation/wrist_image": image_tools.resize_with_pad(data["wrist_rgb"], 224, 224),
                "observation/state": data["robot_state"], "prompt": row["instruction"]})
        hasher = hashlib.sha256()
        for array in [value["image"][k] for k in sorted(value["image"])] + [
                value[k] for k in ("state", "tokenized_prompt", "tokenized_prompt_mask")]:
            hasher.update(np.ascontiguousarray(array).tobytes())
        hashes.append(hasher.hexdigest())
        values.append(value)
    return values, tensor_batch(values)[0], hashes


def traced_forward(net, batch, noise, basis, remove):
    import torch

    times, velocities, seen = [], [], {}
    denoise = net.denoise_step
    def trace_time(*args):
        t = args[-1]
        assert torch.all(t == t[0])
        times.append(t[0].item())
        return denoise(*args)
    def pre_hook(module, args):
        assert len(times) == len(velocities) + 1
        if len(times) == 10:
            x = args[0]
            seen["before"] = x.detach()
            if remove:
                x = ((x.double() @ basis) @ basis.T).to(x.dtype)
            seen["after"] = x.detach()
            return (x,)
    def post_hook(module, args, output):
        velocities.append(output.detach())
    with ExitStack() as ctx:
        ctx.enter_context(patch.object(net, "denoise_step", new=trace_time))
        ctx.callback(net.action_out_proj.register_forward_pre_hook(pre_hook).remove)
        ctx.callback(net.action_out_proj.register_forward_hook(post_hook).remove)
        chunk = net.sample_actions("cuda", batch, noise=noise.clone(), num_steps=10)
    np.testing.assert_allclose(times, np.arange(10, 0, -1) / 10, atol=2e-7, rtol=0)
    assert len(velocities) == 10 and seen["after"].shape[1:] == (10, 1024)
    return chunk, velocities, times, seen


def maxabs(x):
    return float(np.abs(x).max())


def difference(a, b):
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return dict(max_abs=maxabs(delta), rmse=float(np.sqrt(np.mean(delta**2))))


def intervention(name):
    import torch
    from motor_lora import frozen_parameter_digest
    from v6_scoped_lora import LoRALinear

    protocol = read(RUN / "protocol.json")
    tol = protocol["tolerance"]
    folder = RUN / "intervention" / name
    folder.mkdir(parents=True, exist_ok=True)
    manifest = read(RUN / "manifest.json")
    idx = [r["index"] for r in protocol["panel"]]
    rows = [manifest["rows"][i] for i in idx]
    policy = load_model(name, protocol)
    net = policy._model
    weight_hash = frozen_parameter_digest(net)
    values, batch, hashes = inputs(policy, rows)
    with h5py.File(REFERENCE / "features" / (name + ".h5")) as ref:
        assert ref.attrs["complete"] and ref.attrs["model_weight_digest"] == weight_hash
        assert [ref["input_sha256"][i].decode() for i in idx] == hashes
        ref_hidden = np.stack([ref[PRE][i] for i in idx])
        ref_velocity = np.stack([ref[POST][i] for i in idx])
    head = net.action_out_proj
    lora = isinstance(head, LoRALinear)
    assert lora == (name in ("ae", "all"))
    base = head.base if lora else head
    weight = base.weight.detach().double().cpu().numpy()
    bias = base.bias.detach().double().cpu().numpy()
    if lora:
        a = head.lora_a.detach().double().cpu().numpy()
        b = head.lora_b.detach().double().cpu().numpy()
        weight = weight + head.scale * (b @ a)
    _, singular, vt = np.linalg.svd(weight, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(weight.shape) * singular[0]
    q = np.ascontiguousarray(vt[singular > cutoff].T)
    orth_error = maxabs(q.T @ q - np.eye(q.shape[1]))
    assert orth_error <= tol["algebra_absolute"]
    noise0 = np.random.default_rng(protocol["noise_seed"]).standard_normal((10, 32)).astype(np.float32)
    audit_root = contrast.RUN if name in ("base", "libero") else REFERENCE
    assert array_sha(noise0) == read(audit_root / "features" / (name + "_audit.json"))["fixed_initial_noise_sha256"]
    noise = torch.as_tensor(np.repeat(noise0[None], len(rows), axis=0), device="cuda")
    with torch.inference_mode():
        baseline = net.sample_actions("cuda", batch, noise=noise.clone(), num_steps=10)
        gpu_q = torch.as_tensor(q, device="cuda")
        ident, vi, times, hi = traced_forward(net, batch, noise, gpu_q, False)
        removed, vr, _, hr = traced_forward(net, batch, noise, gpu_q, True)
        # Supplementary arithmetic check on already captured inputs; keeps all policy forwards at native precision.
        native_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            strict_full = head(hi["before"]).float().cpu().numpy()
            strict_out = head(hr["after"]).float().cpu().numpy()
        finally:
            torch.set_float32_matmul_precision(native_precision)
    assert torch.equal(baseline, ident) and all(torch.equal(x, y) for x, y in zip(vi[:-1], vr[:-1]))
    assert torch.equal(hi["before"], hr["before"])
    h = hi["before"].double().cpu().numpy()
    y = vi[-1].float().cpu().numpy()
    y_removed = vr[-1].float().cpu().numpy()
    algebra_null = maxabs((h - (h @ q) @ q.T) @ weight.T)
    np.testing.assert_allclose(algebra_null, 0, rtol=0, atol=tol["algebra_absolute"])
    expected = h @ weight.T + bias
    def pooled(x):
        return x.float().mean(dim=1).cpu().numpy()
    full_pool, out_pool = pooled(hi["before"]), pooled(hr["after"])
    hidden_replay = maxabs(full_pool - ref_hidden)
    velocity_replay = maxabs(pooled(vi[-1]) - ref_velocity)
    def physical(chunk):
        return np.stack([policy._output_transform({"state": value["state"].copy(), "actions": array.copy()})["actions"]
                         for value, array in zip(values, chunk)])
    x0, xi, xr = [x.float().cpu().numpy() for x in (baseline, ident, removed)]
    p0, pi, pr = [physical(x) for x in (x0, xi, xr)]
    assert np.array_equal(p0, pi) and p0.shape[1:] == (10, 7) and x0.shape[1:] == (10, 32)
    def controller(action):
        cmd = np.clip(action, -1., 1.)
        cmd[..., 6] = np.where(action[..., 6] > 0, 1., -1.)
        return cmd
    cmd0, cmd_removed = controller(p0), controller(pr)
    gain = np.abs(physical(np.ones_like(x0)) - physical(np.zeros_like(x0)))
    sample_stats = []
    for j, row in enumerate(rows):
        velocity_budget = tol["native_floor"] + tol["native_relative"] * max(1, maxabs(y[j]))
        chunk_budget = .1 * velocity_budget + tol["fp32_roundoff"] * max(1, maxabs(x0[j]))
        action_budget = gain[j] * chunk_budget + tol["fp32_roundoff"] * max(1, maxabs(p0[j]))
        effective = difference(expected[j], y[j])
        vel, chunk, action = difference(y[j], y_removed[j]), difference(x0[j], xr[j]), difference(p0[j], pr[j])
        sample_stats.append(dict(item_id=row["item_id"], effective=effective, velocity32=vel, chunk32=chunk, action7=action,
                          controller7=difference(cmd0[j], cmd_removed[j]),
                          gripper_sign_changes=int(np.sum(cmd0[j,:,6] != cmd_removed[j,:,6])),
                          head_only_highest_precision=difference(strict_full[j], strict_out[j]),
                          limits=dict(velocity=velocity_budget, chunk=chunk_budget, min_action7=float(action_budget.min()),
                                      max_action7=float(action_budget.max())),
                          effective_head_pass=effective["max_abs"] <= velocity_budget,
                          invariance_pass=bool(vel["max_abs"] <= velocity_budget and chunk["max_abs"] <= chunk_budget
                                               and np.all(np.abs(p0[j]-pr[j]) <= action_budget)),
                          strict_pass=all(s["max_abs"] <= tol["strict_absolute"] for s in (vel,chunk,action))))
    np.savez_compressed(folder / "readout.npz", W=weight, bias=bias, Q=q, singular=singular,
                        panel_full=full_pool, panel_out=out_pool, global_indices=idx)
    result = dict(model=name, checkpoint_parameter_digest=weight_hash, effective_W_sha256=array_sha(weight),
                  rank=q.shape[1], cutoff=float(cutoff), singular_max=float(singular[0]), singular_min=float(singular[-1]),
                  LoRA_in_out_proj=lora, algebra_max_abs_Wnull=algebra_null, orthogonality_max_abs=orth_error,
                  identity_chunk32_max_abs=0., identity_action7_max_abs=0., first_9_velocities_equal=True,
                  final_preinput_equal=True, flow_times=times, input_hashes=hashes, noise_sha256=array_sha(noise0),
                  live_minus_archived_pre_max_abs=hidden_replay, live_minus_archived_post_max_abs=velocity_replay,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), matmul=torch.get_float32_matmul_precision(),
                  input_dtype=str(hi["before"].dtype), head_weight_dtype=str(base.weight.dtype), output_dtype=str(vi[-1].dtype),
                  passed=all(r["effective_head_pass"] and r["invariance_pass"] for r in sample_stats), samples=sample_stats)
    write_json(folder / "numerics.json", result)
    print(name, "GPU audit", result["passed"], "archive replay", hidden_replay, velocity_replay, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "intervention"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.model:
        intervention(args.model)
    else:
        parser.error("intervention requires --model")
