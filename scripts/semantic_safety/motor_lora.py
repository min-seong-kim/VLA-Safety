"""Strict action-expert-only LoRA and frozen-prefix flow-matching utilities."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
for key, value in {"HF_HOME": ".cache/huggingface", "TORCH_HOME": ".cache/torch",
                   "OPENPI_DATA_HOME": ".cache/openpi"}.items():
    os.environ.setdefault(key, str(ROOT / value))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("WANDB_MODE", "disabled")
EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model.layers."
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, value):
        base = self.base(value)
        delta = F.linear(F.linear(value.float(), self.lora_a), self.lora_b)
        return base + (delta * self.scale).to(base.dtype)


def load_policy(model_name="pi05_libero", device="cuda"):
    from openpi.policies import policy_config
    from openpi.training import config

    cfg = config.get_config(model_name)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    checkpoint = ROOT / "checkpoints" / f"{model_name}_pytorch"
    policy = policy_config.create_trained_policy(cfg, checkpoint, pytorch_device=device)
    policy._model.requires_grad_(False)
    policy._model.eval()
    policy._model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    policy._model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    return policy, cfg


def install_lora(model, rank=16, alpha=32.):
    model.requires_grad_(False)
    selected = [(name, module) for name, module in model.named_modules()
                if name.startswith(EXPERT_PREFIX) and name.rsplit(".", 1)[-1] in TARGETS
                and isinstance(module, nn.Linear)]
    if len(selected) != 18 * 7:
        raise ValueError(f"Expected 126 expert linear modules, found {len(selected)}")
    for name, module in selected:
        parent, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, LoRALinear(module, rank, alpha))
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(name.startswith(EXPERT_PREFIX) and ".lora_" in name for name in trainable)
    return {"rank": rank, "alpha": alpha, "dropout": 0., "target_modules": [name for name, _ in selected],
            "trainable_parameters": sum(p.numel() for p in trainable.values()),
            "trainable_parameter_names": list(trainable),
            "frozen": "all original parameters, including vision, language, norms and action input/output projections"}


def frozen_parameter_digest(model, *, vlm_only=False):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            continue
        if vlm_only and not name.startswith("paligemma_with_expert.paligemma."):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_input(policy, main, wrist, state, instruction, actions=None):
    from openpi_client import image_tools

    value = {
        "observation/image": image_tools.resize_with_pad(np.ascontiguousarray(main[::-1, ::-1]), 224, 224),
        "observation/wrist_image": image_tools.resize_with_pad(np.ascontiguousarray(wrist[::-1, ::-1]), 224, 224),
        "observation/state": np.asarray(state, dtype=np.float32), "prompt": instruction,
    }
    if actions is not None:
        value["actions"] = np.asarray(actions, dtype=np.float32)
    return policy._input_transform(value)


def tensor_batch(items, device="cuda"):
    from openpi.models.model import Observation

    def stack(values):
        first = values[0]
        if isinstance(first, dict):
            return {key: stack([value[key] for value in values]) for key in first}
        array = np.stack(values)
        if array.dtype.kind == "f":
            array = array.astype(np.float32)
        return torch.from_numpy(array).to(device)
    batch = stack(items)
    return Observation.from_dict(batch), batch.get("actions")


def frozen_prefix(model, observation):
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    with torch.no_grad():
        images, image_masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)
        prefix, mask, att = model.embed_prefix(images, image_masks, tokens, token_masks)
        attention = model._prepare_attention_masks_4d(make_att_2d_masks(mask, att))
        positions = torch.cumsum(mask, dim=1) - 1
        (hidden, _), cache = model.paligemma_with_expert.forward(
            attention_mask=attention, position_ids=positions, inputs_embeds=[prefix, None], use_cache=True)
    return state, mask, cache, hidden


def flow_loss(model, observation, actions, *, noise=None, time=None):
    if noise is None:
        noise = model.sample_noise(actions.shape, actions.device)
    if time is None:
        time = model.sample_time(actions.shape[0], actions.device)
    state, mask, cache, _ = frozen_prefix(model, observation)
    weight = time[:, None, None]
    noisy_actions = weight * noise + (1 - weight) * actions
    predicted = model.denoise_step(state, mask, cache, noisy_actions, time)
    return F.mse_loss(predicted.float(), (noise - actions).float())


def save_adapter(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save({"state_dict": weights, "metadata": metadata}, temporary)
    temporary.replace(path)


def load_adapter(model, path):
    package = torch.load(path, map_location="cpu", weights_only=False)
    meta = package["metadata"]
    architecture = install_lora(model, rank=meta["lora"]["rank"], alpha=meta["lora"]["alpha"])
    named = dict(model.named_parameters())
    assert set(package["state_dict"]) == set(architecture["trainable_parameter_names"])
    with torch.no_grad():
        for name, value in package["state_dict"].items():
            named[name].copy_(value)
    model.eval()
    return meta
