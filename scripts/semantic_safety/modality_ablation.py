"""Token- and route-specific masking on frozen pi0.5 policies, offline first scenes and shared rollout readout."""

import argparse
from contextlib import ExitStack
import dataclasses
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

import ae_output_nullspace_pilot as prior
import contrast_initial_linear as contrast
from representation_common import immutable_json
from rollout_common import ROOT, write_json

RUN = ROOT / "runs/semantic_safety_modality_ablation_v1"
REFERENCE = prior.REFERENCE
COMMON = prior.MODELS
STATE_BASE = "base_state"
CONDITIONS = ("intact", "global_text", "global_image", "ae_text", "ae_image")
STATE_CONDITIONS = ("intact", "global_state", "ae_state")
VLM = "vlm/layer_17"
AE = "ae/flow_10/layer_17"
SITES = {VLM: 2048, AE: 1024}
BATCH = 12
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def conditions(name):
    return STATE_CONDITIONS if name == STATE_BASE else CONDITIONS


def prepare():
    import hashlib

    manifest = REFERENCE / "manifest.json"
    rows = prior.read(manifest)["rows"]
    selection = prior.read(REFERENCE / "frozen_selection.json")["selected"]
    protocol = dict(
        protocol_version=1, models=list(COMMON), supplementary_base=STATE_BASE,
        selected_adapters={m: {k: selection[m][k] for k in ("adapter", "adapter_sha256", "cumulative_step")}
                           for m in ("ae", "vlm", "all")},
        manifest=str(manifest.relative_to(ROOT)), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        common_conditions=list(CONDITIONS), base_state_conditions=list(STATE_CONDITIONS),
        common_readout="LIBERO normalization, horizon 10, discrete_state_input=False, ten denoising calls",
        base_state_readout="same Base weights and LIBERO normalization, state in prompt, horizon 50, ten flows",
        global_mask="remove selected valid prefix positions before attention and positional indexing; pool surviving positions",
        ae_mask="remove only selected prefix KV columns from AE attention at every denoising step; preserve all position IDs",
        features=list(SITES), tracks=["A", "B"], decisions=len(rows), noise_seed=contrast.SEED, batch=BATCH,
        probes="reuse unmasked fold probe and independently refit each condition; unchanged 5 group folds and C grid",
        rollout=dict(suites=list(SUITES), task_ids=[0, 4, 8], initial_states=[0], train_or_selection_scores_used=False,
                     horizon=460, libero_10_horizon=600, cadence=10, settle=10, native_image_rotation_degrees=180,
                     noise="72000000 + suite_index*10000 + task_id*100 + initial_state_id + control_step"),
        smoke=16, labels="unsafe=1, safe=0", state_audit="change raw 8D state before normalization and tokenization")
    assert len(rows) == 457
    immutable_json(RUN / "protocol.json", protocol)
    write_json(RUN / "reporting/delivery_mode.json", {"mode": "foreground"})
    for condition in set(CONDITIONS + STATE_CONDITIONS):
        folder = RUN / "conditions" / condition
        (folder / "features").mkdir(parents=True, exist_ok=True)
        target = folder / "manifest.json"
        if not target.exists():
            target.symlink_to(manifest)
        assert target.read_bytes() == manifest.read_bytes()


def load_policy(name):
    protocol = prior.read(RUN / "protocol.json")
    if name in COMMON:
        return prior.load_model(name, protocol)
    from openpi.training import config

    assert name == STATE_BASE
    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, discrete_state_input=True, action_horizon=50))
    with patch.object(config, "get_config", return_value=cfg):
        return prior.load_model("base", protocol)


def state_spans(values, instructions, processor):
    """Align numeric state values (and separating spaces) to the actual SentencePiece token IDs."""
    spans = []
    for value, prompt in zip(values, instructions, strict=True):
        digits = np.digitize(value["state"][:8], bins=np.linspace(-1, 1, 257)[:-1]) - 1
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        prefix = f"Task: {cleaned}, State:"
        numeric = " " + " ".join(map(str, digits))
        source = prefix + numeric + ";\nAction: "
        pieces = processor.encode_as_immutable_proto(source).pieces
        valid = value["tokenized_prompt_mask"].sum()
        ids = [processor.bos_id()] + [p.id for p in pieces]
        assert np.array_equal(ids, value["tokenized_prompt"][:valid]) and valid < len(value["tokenized_prompt"])
        mask = np.zeros_like(value["tokenized_prompt_mask"], dtype=bool)
        for i, piece in enumerate(pieces, start=1):
            if piece.end > len(prefix) and piece.begin < len(prefix) + len(numeric):
                assert piece.begin >= len(prefix) and piece.end <= len(prefix) + len(numeric)
                mask[i] = True
        assert mask.any() and np.all(mask <= value["tokenized_prompt_mask"])
        spans.append(mask)
    return np.stack(spans)


def forward(net, batch, noise, condition, *, capture=False, state_positions=None):
    """Use the policy's original sampler; never change suffix self-attention, weights or input noise."""
    import torch
    from extract_representation_linear import masked_mean

    embed, denoise, mask4 = net.embed_prefix, net.denoise_step, net._prepare_attention_masks_4d
    scope, modality = ("intact", None) if condition == "intact" else condition.split("_")
    out, seen = {}, {"flow": 0, "ae_masks": 0}
    horizon = noise.shape[1]
    assert horizon == net.config.action_horizon and net.pi05

    def prefix(images, image_masks, tokens, token_masks):
        embs, valid, att = embed(images, image_masks, tokens, token_masks)
        off = valid.shape[1] - tokens.shape[1]
        assert len(images) == 3 and off == 768
        removed = torch.zeros_like(valid)
        if modality == "image":
            removed[:, :off] = valid[:, :off]
        elif modality == "text":
            removed[:, off:] = token_masks
        elif modality == "state":
            removed[:, off:] = torch.as_tensor(state_positions, device=valid.device)
        else:
            assert modality is None
        pool = valid & ~removed if scope == "global" else valid
        if condition == "global_state":
            # Numeric strings have variable token lengths. Give the surviving prompt tokens
            # identical physical slots as well as identical RoPE IDs on every state perturbation.
            keep = pool[:, off:]
            order = torch.argsort((~keep).int(), dim=1, stable=True)
            kept = keep.gather(1, order)
            text = embs[:, off:].gather(1, order[:, :, None].expand(-1, -1, embs.shape[-1]))
            embs = torch.cat([embs[:, :off], text.masked_fill(~kept[:, :, None], 0)], dim=1)
            pool = torch.cat([pool[:, :off], kept], dim=1)
            assert not att.any()
        seen.update(removed=removed, pool=pool, prefix=valid.shape[1])
        return embs, pool, att

    def ae_step(*args):
        seen["flow"] += 1
        return denoise(*args)

    def attention(mask):
        if scope == "ae" and seen["flow"]:
            prefix_len = seen["prefix"]
            assert mask.shape[1:] == (horizon, prefix_len + horizon)
            mask = mask.clone()
            mask[:, :, :prefix_len] &= ~seen["removed"][:, None, :]
            seen["ae_masks"] += 1
        return mask4(mask)

    def vlm_hook(module, args, output):
        states = output[0] if isinstance(output, (tuple, list)) else output
        out[VLM] = masked_mean(states, seen["pool"])

    def ae_hook(module, args, output):
        if seen["flow"] == 10:
            states = output[0] if isinstance(output, (tuple, list)) else output
            assert states.shape[1] == horizon
            out[AE] = states.float().mean(dim=1)

    with torch.inference_mode(), ExitStack() as stack:
        stack.enter_context(patch.object(net, "embed_prefix", new=prefix))
        stack.enter_context(patch.object(net, "denoise_step", new=ae_step))
        if scope == "ae":
            stack.enter_context(patch.object(net, "_prepare_attention_masks_4d", new=attention))
        if capture:
            family = net.paligemma_with_expert
            stack.callback(family.paligemma.language_model.layers[17].register_forward_hook(vlm_hook).remove)
            stack.callback(family.gemma_expert.model.layers[17].register_forward_hook(ae_hook).remove)
        chunk = net.sample_actions("cuda", batch, noise=noise, num_steps=10)
    assert seen["flow"] == 10 and seen["ae_masks"] == (10 if scope == "ae" else 0)
    if capture:
        assert set(out) == set(SITES)
    return chunk, {k: v.cpu().numpy() for k, v in out.items()}


def physical(policy, values, chunks):
    return np.stack([policy._output_transform(dict(state=v["state"].copy(), actions=x.copy()))["actions"]
                     for v, x in zip(values, chunks.float().cpu().numpy(), strict=True)])


def perturbed_inputs(policy, rows):
    from openpi_client import image_tools
    from motor_lora import tensor_batch

    values = []
    for row in rows:
        with np.load(ROOT / row["observation_path"]) as data:
            state = data["robot_state"].copy() + np.linspace(-.6, .6, 8, dtype=np.float32)
            values.append(policy._input_transform({
                "observation/image": image_tools.resize_with_pad(data["agentview_rgb"], 224, 224),
                "observation/wrist_image": image_tools.resize_with_pad(data["wrist_rgb"], 224, 224),
                "observation/state": state, "prompt": row["instruction"]}))
    return values, tensor_batch(values)[0]


def extract(name, limit=None):
    import torch
    from motor_lora import frozen_parameter_digest
    from openpi.models.tokenizer import PaligemmaTokenizer

    protocol = prior.read(RUN / "protocol.json")
    all_rows = prior.read(REFERENCE / "manifest.json")["rows"]
    out = RUN / "smoke" if limit else RUN
    rows = all_rows[:limit] if limit else all_rows
    policy = load_policy(name)
    net = policy._model
    weight = frozen_parameter_digest(net)
    is_state = name == STATE_BASE
    assert net.config.discrete_state_input == is_state
    horizon = net.config.action_horizon
    processor = PaligemmaTokenizer(200)._tokenizer if is_state else None
    fixed = np.random.default_rng(contrast.SEED).standard_normal((horizon, 32)).astype(np.float32)
    ref_name = "base" if is_state else name
    audit = {"model": name, "rows": len(rows), "weight_digest": weight, "horizon": horizon, "state_input": is_state}
    with ExitStack() as stack:
        ref = stack.enter_context(h5py.File(REFERENCE / "features" / (ref_name + ".h5")))
        assert weight == ref.attrs["model_weight_digest"] and ref.attrs["complete"]
        files = {}
        for cond in conditions(name):
            path = out / "conditions" / cond / "features" / (name + ".h5")
            path.parent.mkdir(parents=True, exist_ok=True)
            f = stack.enter_context(h5py.File(path, "a"))
            sig = contrast.digest([protocol, weight, cond, name, len(rows)])
            if "written" not in f:
                f.attrs.update(signature=sig, complete=False)
                f.create_dataset("written", shape=(len(rows),), dtype="bool")
                for site, width in SITES.items():
                    f.create_dataset(site, shape=(len(rows), width), dtype="float32", compression="lzf")
            assert f.attrs["signature"] == sig
            files[cond] = f
        for start in range(0, len(rows), BATCH):
            group = rows[start:start+BATCH]
            sl = slice(start, start+len(group))
            if not limit and start and all(f["written"][sl].all() for f in files.values()):
                continue
            values, batch, hashes = prior.inputs(policy, group)
            if not is_state:
                assert [h.decode() for h in ref["input_sha256"][sl]] == hashes
            prompts = [r["instruction"] for r in group]
            spans = state_spans(values, prompts, processor) if is_state else None
            noise = torch.as_tensor(np.repeat(fixed[None], len(group), axis=0), device="cuda")
            if start == 0:
                with torch.inference_mode():
                    original = net.sample_actions("cuda", batch, noise=noise.clone(), num_steps=10)
                changed, changed_batch = perturbed_inputs(policy, group)
                assert not torch.equal(changed_batch.state, batch.state)
                assert torch.equal(changed_batch.tokenized_prompt, batch.tokenized_prompt) != is_state
                changed_spans = state_spans(changed, prompts, processor) if is_state else None
                audit["state_test_n"] = len(group)
                audit["state_token_ids_changed"] = is_state
                audit["image_valid_positions"] = sum(int(v) * 256 for v in values[0]["image_mask"].values())
                audit["state_positions_first_sample"] = int(spans[0].sum()) if is_state else 0
            for cond, f in files.items():
                chunk, result = forward(net, batch, noise, cond, capture=True, state_positions=spans)
                if cond == "intact":
                    original_vlm = result[VLM]
                    if start == 0:
                        assert torch.equal(chunk, original)
                if cond.startswith("ae_"):
                    assert np.array_equal(result[VLM], original_vlm)
                for site, feature in result.items():
                    assert np.isfinite(feature).all()
                    f[site][sl] = feature
                    if cond == "intact" and not is_state:
                        # With the archive's batch shape the unmasked representation must reproduce bit for bit.
                        if not limit or len(group) == BATCH:
                            assert np.array_equal(feature, ref[site][sl]), (name, site, start, float(np.abs(feature-ref[site][sl]).max()))
                if start == 0 and cond in ({"intact", "global_state"} if is_state else {"intact"}):
                    other, other_result = forward(net, changed_batch, noise, cond, capture=True, state_positions=changed_spans)
                    diff = {key: float(np.max(np.abs(other_result[key]-result[key]))) for key in SITES}
                    diff["physical_action"] = float(np.max(np.abs(physical(policy, values, chunk)-physical(policy, changed, other))))
                    audit["perturb_"+cond] = diff
                    if not is_state or cond == "global_state":
                        assert all(v == 0 for v in diff.values())
                f.flush()
                f["written"][sl] = True
                f.flush()
        for f in files.values():
            assert f["written"][:].all()
            f.attrs["complete"] = True
    audit["complete"] = True
    write_json(out / "audits" / (name + ".json"), audit)
    print(name, "complete", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("operation", choices=["prepare", "extract"])
    parser.add_argument("--model", choices=COMMON + (STATE_BASE,))
    parser.add_argument("--limit", choices=[16], type=int)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare()
    else:
        extract(args.model, args.limit)
