"""Local scoped-adapter inference, preserving explicit custom/native RGB conventions."""

import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from benign_policy_v1 import load_policy, make_upright_input
from motor_lora import make_input, tensor_batch
from v6_scoped_lora import install
from v6_lora_server import AUTHKEY
from benign_adaptation_v1 import sha


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--socket",type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(8)
    policy,cfg=load_policy();model=policy._model
    architecture=install(model,"all");named=dict(model.named_parameters());model.requires_grad_(False)
    names=architecture["trainable_parameter_names"]
    metadata=dict(model="pi05_libero",adapter=None,adapter_sha256=None,scope="baseline",training_step=0,
                  action_horizon=10,denoising_steps=10,
                  image_orientation="custom raw vertical flip; native raw 180 rotation; explicit per packet")
    audits={};stop=False
    if args.socket.exists():raise FileExistsError(args.socket)
    with Listener(str(args.socket),family="AF_UNIX",authkey=AUTHKEY) as listener:
        print("LISTENING",flush=True)
        while not stop:
            with listener.accept() as connection:
                connection.send(dict(ready=True,metadata=metadata))
                while True:
                    try:packet=connection.recv()
                    except EOFError:break
                    try:
                        if packet["op"] in {"shutdown","disconnect"}:
                            stop=packet["op"]=="shutdown";connection.send(dict(closed=True));break
                        if packet["op"]=="activate":
                            path=Path(packet["adapter"]) if packet.get("adapter") else None
                            with torch.no_grad():
                                for name in names:named[name].zero_()
                                if path:
                                    package=torch.load(path,map_location="cpu",weights_only=False)
                                    assert set(package["state_dict"])==set(package["metadata"]["lora"]["trainable_parameter_names"])
                                    assert set(package["state_dict"])<=set(names)
                                    for name,value in package["state_dict"].items():named[name].copy_(value)
                            metadata.update(adapter=str(path) if path else None,adapter_sha256=sha(path) if path else None,
                                scope=package["metadata"]["lora"]["scope"] if path else "baseline",
                                training_step=package["metadata"]["step"] if path else 0)
                            audits={};connection.send(dict(ready=True,metadata=metadata));continue
                        started=time.monotonic();items=[]
                        for item in packet["items"]:
                            obs=item["observation"]
                            builder=make_upright_input if item["image_orientation"]=="upright" else make_input
                            items.append(builder(policy,obs["image"],obs["wrist_image"],obs["state"],item["instruction"]))
                        assert all(np.asarray(v["tokenized_prompt_mask"]).sum()<200 for v in items)
                        noise=np.stack([np.random.default_rng(p["seed"]).standard_normal((10,32)).astype(np.float32) for p in packet["items"]])
                        for i,entry in enumerate(packet["items"]):
                            orientation=entry["image_orientation"]
                            if orientation in audits:continue
                            from openpi_client import image_tools
                            obs=entry["observation"];native=orientation=="native180"
                            main=obs["image"][::-1,::-1] if native else obs["image"][::-1]
                            wrist=obs["wrist_image"][::-1,::-1] if native else obs["wrist_image"][::-1]
                            inputs={"observation/image":image_tools.resize_with_pad(np.ascontiguousarray(main),224,224),
                                    "observation/wrist_image":image_tools.resize_with_pad(np.ascontiguousarray(wrist),224,224),
                                    "observation/state":obs["state"],"prompt":entry["instruction"]}
                            one,_=tensor_batch(items[i:i+1])
                            with torch.inference_mode():
                                reference=policy.infer(inputs,noise=noise[i])["actions"]
                                sampled=model.sample_actions("cuda",one,noise=torch.from_numpy(noise[i:i+1]).cuda(),num_steps=10).cpu().numpy()[0]
                            direct=policy._output_transform({"state":items[i]["state"],"actions":sampled})["actions"]
                            error=float(np.max(np.abs(reference-direct)));assert error<=1e-5,error
                            audits[orientation]=dict(passed=True,official_policy_infer_max_abs_difference=error)
                        batch,_=tensor_batch(items)
                        with torch.inference_mode():chunks=model.sample_actions("cuda",batch,noise=torch.from_numpy(noise).cuda(),num_steps=10).cpu().numpy()
                        actions=[np.asarray(policy._output_transform({"state":v["state"],"actions":chunk})["actions"],dtype=np.float32) for v,chunk in zip(items,chunks)]
                        assert all(a.shape==(10,7) and np.isfinite(a).all() for a in actions)
                        connection.send(dict(actions=actions,inference_seconds=time.monotonic()-started,
                                             reference_audit=dict(passed=True,orientation_audits=audits)))
                    except Exception:connection.send(dict(error=traceback.format_exc()))
    args.socket.unlink(missing_ok=True)


if __name__=="__main__":main()
