"""Preserve official normalization while making dataset-specific image orientation explicit."""

import dataclasses
from unittest.mock import patch

import numpy as np

from rollout_common import ROOT


def make_upright_input(policy,main,wrist,state,instruction,actions=None):
    from openpi_client import image_tools
    value={"observation/image":image_tools.resize_with_pad(np.ascontiguousarray(main[::-1]),224,224),
           "observation/wrist_image":image_tools.resize_with_pad(np.ascontiguousarray(wrist[::-1]),224,224),
           "observation/state":np.asarray(state,dtype=np.float32),"prompt":instruction}
    if actions is not None:value["actions"]=np.asarray(actions,dtype=np.float32)
    return policy._input_transform(value)


def load_policy(model_name="pi05_libero",device="cuda"):
    import torch
    torch.set_num_threads(8)
    from openpi.training import config
    from openpi.policies import policy_config
    from safetensors import safe_open
    import safetensors.torch
    from transformers.modeling_utils import no_init_weights

    cfg=config.get_config(model_name)
    cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
    original=safetensors.torch.load_model
    def restore_declared_ties(instance,filename,*args,**kwargs):
        with safe_open(filename,framework="pt",device="cpu") as source:aliases=source.metadata() or {}
        for alias,retained in aliases.items():
            if not alias.endswith(".weight") or not retained.endswith(".weight"):continue
            parent,leaf=alias.rsplit(".",1);other,other_leaf=retained.rsplit(".",1)
            setattr(instance.get_submodule(parent),leaf,getattr(instance.get_submodule(other),other_leaf))
        return original(instance,filename,*args,**kwargs)
    with no_init_weights(),patch("safetensors.torch.load_model",new=restore_declared_ties):
        policy=policy_config.create_trained_policy(cfg,ROOT/"checkpoints"/(model_name+"_pytorch"),pytorch_device=device)
    policy._model.requires_grad_(False);policy._model.eval()
    policy._model.paligemma_with_expert.paligemma.language_model.config._attn_implementation="eager"
    policy._model.paligemma_with_expert.gemma_expert.model.config._attn_implementation="eager"
    return policy,cfg
