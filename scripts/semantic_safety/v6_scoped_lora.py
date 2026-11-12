"""Scope-controlled LoRA for the vision-language backbone and action expert."""

from contextlib import nullcontext
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from motor_lora import LoRALinear as BaseLoRALinear


class LoRALinear(BaseLoRALinear):
    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


VLM = "paligemma_with_expert.paligemma."
AE = "paligemma_with_expert.gemma_expert.model."
ACTION_PROJECTIONS = {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}


def module_scope(name):
    if name.startswith(VLM) and ".lm_head" not in name and ".head." not in name:
        return "vlm"
    if name.startswith(AE) or name in ACTION_PROJECTIONS:
        return "ae"
    return None


def install(model, scope, rank=16, alpha=32.):
    assert scope in {"vlm", "ae", "all"}
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    model.requires_grad_(False)
    selected = [(name, module) for name, module in model.named_modules()
                if isinstance(module, nn.Linear) and module_scope(name) is not None
                and (scope == "all" or module_scope(name) == scope)]
    assert selected
    for name, module in selected:
        parent, _, leaf = name.rpartition(".")
        device=module.weight.device.index
        with torch.random.fork_rng(devices=[] if device is None else [device]):
            torch.manual_seed(int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
            adapter=LoRALinear(module,rank,alpha)
        setattr(model.get_submodule(parent) if parent else model, leaf, adapter)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all(".lora_" in n for n in names)
    return dict(scope=scope, rank=rank, alpha=alpha, dropout=0.,initialization="module-name SHA256 seed; shared modules identical across scopes",
                target_modules=[n for n, _ in selected], trainable_parameter_names=names,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                module_counts={s:sum(module_scope(n)==s for n,_ in selected) for s in ["vlm","ae"]},
                frozen="all original parameters; only selected LoRA matrices are trainable")


def prefix(model, observation, *, gradients):
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    with nullcontext() if gradients else torch.no_grad():
        images, image_masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)
        embeddings, mask, att = model.embed_prefix(images, image_masks, tokens, token_masks)
        attention = model._prepare_attention_masks_4d(make_att_2d_masks(mask, att))
        positions = torch.cumsum(mask, dim=1)-1
        (hidden, _), cache = model.paligemma_with_expert.forward(
            attention_mask=attention, position_ids=positions, inputs_embeds=[embeddings,None],use_cache=True)
    return state, mask, cache, hidden


def loss_matrix(model, observation, actions, scope, *, noise=None, time=None):
    # Prefix cannot attend to actions. Its differentiable KV cache gives the same
    # causal flow-matching computation while avoiding joint quadratic attention.
    if noise is None: noise = model.sample_noise(actions.shape, actions.device)
    if time is None: time = model.sample_time(actions.shape[0], actions.device)
    state, mask, cache, _ = prefix(model, observation, gradients=scope!="ae")
    weight=time[:,None,None]
    prediction=model.denoise_step(state, mask, cache, weight*noise+(1-weight)*actions, time)
    return F.mse_loss(prediction.float(),(noise-actions).float(),reduction="none")


def load(model, path):
    package=torch.load(path,map_location="cpu",weights_only=False)
    meta=package["metadata"]; cfg=meta["lora"]
    architecture=install(model,cfg["scope"],cfg["rank"],cfg["alpha"])
    assert set(architecture["trainable_parameter_names"])==set(package["state_dict"])
    named=dict(model.named_parameters())
    with torch.no_grad():
        for name,value in package["state_dict"].items():named[name].copy_(value)
    model.eval()
    return meta
