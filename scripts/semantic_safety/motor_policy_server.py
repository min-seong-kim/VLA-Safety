"""Local batched pi0.5 inference for motor evaluation, without probe extraction."""

import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from motor_lora import install_lora, load_policy, make_input, tensor_batch


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    policy, config = load_policy()
    model = policy._model
    lora = install_lora(model)
    adapter = "original"
    if args.socket.exists():
        raise FileExistsError(args.socket)
    with Listener(str(args.socket), family="AF_UNIX", authkey=b"motor-b-local") as listener:
        print("LISTENING", str(args.socket), flush=True)
        with listener.accept() as connection:
            connection.send({"ready": True, "base": "pi05_libero", "action_horizon": 10})
            while True:
                packet = connection.recv()
                try:
                    if packet["op"] == "close":
                        connection.send({"closed": True})
                        break
                    if packet["op"] == "select":
                        adapter = packet["adapter"]
                        named = dict(model.named_parameters())
                        with torch.no_grad():
                            if adapter == "original":
                                for name, value in named.items():
                                    if name.endswith(".lora_b"):
                                        value.zero_()
                            else:
                                package = torch.load(adapter, map_location="cpu", weights_only=False)
                                assert package["metadata"]["lora"]["rank"] == lora["rank"]
                                assert not package["metadata"].get("calibration_smoke_only", False)
                                assert set(package["state_dict"]) == set(lora["trainable_parameter_names"])
                                for name, value in package["state_dict"].items():
                                    named[name].copy_(value)
                        connection.send({"selected": adapter})
                        continue
                    started = time.monotonic()
                    items = [make_input(policy, p["observation"]["image"], p["observation"]["wrist_image"],
                                        p["observation"]["state"], p["instruction"]) for p in packet["items"]]
                    obs, _ = tensor_batch(items)
                    noises = np.stack([np.random.default_rng(p["seed"]).standard_normal((10, 32)).astype(np.float32)
                                       for p in packet["items"]])
                    with torch.inference_mode():
                        chunks = model.sample_actions("cuda", obs, noise=torch.from_numpy(noises).cuda()).cpu().numpy()
                    actions = [policy._output_transform({"state": item["state"], "actions": chunk})["actions"]
                               for item, chunk in zip(items, chunks)]
                    connection.send({"actions": actions, "adapter": adapter,
                                     "inference_seconds": time.monotonic() - started})
                except Exception:
                    connection.send({"error": traceback.format_exc()})
    args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
