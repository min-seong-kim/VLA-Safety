"""Batched pi05_libero inference with explicitly activated scope-controlled adapters."""

import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from motor_lora import load_policy, make_input, tensor_batch
from v6_scoped_lora import install
from semantic_pairs_v6 import sha256

AUTHKEY = b"pi05-v6-baseline-local"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--socket",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    policy,cfg = load_policy("pi05_libero")
    model=policy._model
    architecture=install(model,"all")
    named=dict(model.named_parameters());model.requires_grad_(False)
    adapter_names=architecture["trainable_parameter_names"]
    metadata=dict(model="pi05_libero",checkpoint="checkpoints/pi05_libero_pytorch",
                  weights="official JAX checkpoint converted with official OpenPI converter",
                  adapter=None,additional_training=False,quantized=False,torch_compile=False,
                  action_horizon=cfg.model.action_horizon,action_dim_internal=cfg.model.action_dim,
                  flow_denoising_steps=10,uses_main_image=True,uses_wrist_image=True,
                  state_in_prompt=cfg.model.discrete_state_input,state_in_action_expert=not cfg.model.pi05,
                  image_orientation="raw[::-1, ::-1], resize_with_pad(224, 224)",
                  torch_version=torch.__version__,parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                  feature_extraction=False)
    reference_audit=None
    if args.socket.exists(): raise FileExistsError(args.socket)
    stop=False
    with Listener(str(args.socket),family="AF_UNIX",authkey=AUTHKEY) as listener:
        print("LISTENING",json.dumps(metadata),flush=True)
        while not stop:
            with listener.accept() as connection:
                connection.send({"ready":True,"metadata":metadata})
                while True:
                    try: packet=connection.recv()
                    except EOFError: break
                    try:
                        if packet["op"] in {"disconnect","shutdown"}:
                            stop=packet["op"]=="shutdown";connection.send({"closed":True});break
                        if packet["op"]=="activate":
                            path=Path(packet["adapter"]) if packet.get("adapter") else None
                            with torch.no_grad():
                                for name in adapter_names:named[name].zero_()
                                if path:
                                    package=torch.load(path,map_location="cpu",weights_only=False)
                                    config=package["metadata"]["lora"]
                                    assert config["rank"]==architecture["rank"] and config["alpha"]==architecture["alpha"]
                                    assert set(package["state_dict"])==set(config["trainable_parameter_names"])
                                    assert set(package["state_dict"]).issubset(adapter_names)
                                    for name,value in package["state_dict"].items():named[name].copy_(value)
                            metadata.update(adapter=str(path) if path else None,additional_training=path is not None,
                                            adapter_sha256=sha256(path) if path else None,
                                            scope=package["metadata"]["lora"]["scope"] if path else "baseline",
                                            training_step=package["metadata"]["step"] if path else 0)
                            reference_audit=None
                            connection.send(dict(ready=True,metadata=metadata));continue
                        started=time.monotonic()
                        items=[make_input(policy,p["observation"]["image"],p["observation"]["wrist_image"],
                                          p["observation"]["state"],p["instruction"]) for p in packet["items"]]
                        assert all(np.asarray(i["tokenized_prompt_mask"]).sum()<len(i["tokenized_prompt_mask"]) for i in items)
                        noise=np.stack([np.random.default_rng(p["seed"]).standard_normal(
                            (cfg.model.action_horizon,cfg.model.action_dim)).astype(np.float32) for p in packet["items"]])
                        if reference_audit is None:
                            from openpi_client import image_tools
                            first=packet["items"][0];obs=first["observation"]
                            inputs={"observation/image":image_tools.resize_with_pad(np.ascontiguousarray(obs["image"][::-1,::-1]),224,224),
                                    "observation/wrist_image":image_tools.resize_with_pad(np.ascontiguousarray(obs["wrist_image"][::-1,::-1]),224,224),
                                    "observation/state":obs["state"],"prompt":first["instruction"]}
                            one,_=tensor_batch(items[:1])
                            with torch.inference_mode():
                                reference=policy.infer(inputs,noise=noise[0])["actions"]
                                sampled=model.sample_actions("cuda",one,noise=torch.from_numpy(noise[:1]).cuda(),num_steps=10).cpu().numpy()[0]
                            direct=policy._output_transform({"state":items[0]["state"],"actions":sampled})["actions"]
                            error=float(np.max(np.abs(reference-direct)))
                            if error>1e-5:raise ValueError(f"Official policy.infer differs from direct inference: {error}")
                            reference_audit=dict(official_policy_infer_max_abs_difference=error,passed=True,noise_seed=first["seed"])
                            metadata["reference_audit"]=reference_audit
                        batch,_=tensor_batch(items)
                        with torch.inference_mode():
                            chunks=model.sample_actions("cuda",batch,noise=torch.from_numpy(noise).cuda(),num_steps=10).cpu().numpy()
                        actions=[np.asarray(policy._output_transform({"state":item["state"],"actions":chunk})["actions"],dtype=np.float32)
                                 for item,chunk in zip(items,chunks)]
                        assert all(np.isfinite(a).all() and a.shape==(cfg.model.action_horizon,7) for a in actions)
                        connection.send({"actions":actions,"inference_seconds":time.monotonic()-started,"reference_audit":reference_audit})
                    except Exception:
                        connection.send({"error":traceback.format_exc()})
    args.socket.unlink(missing_ok=True)


if __name__=="__main__":
    main()
