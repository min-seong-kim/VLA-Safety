"""Local policy worker for official checkpoints with read-only feature instrumentation.

The simulator runs in its own environment. This worker uses an authenticated
local Unix socket and writes representations at every policy/control step.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT / ".cache/openpi"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch
from PIL import Image

from rollout_features import ActivationCapture, TimestepFeatureWriter


def inference_namespaces(root):
    # Avoid eager imports of training/data-loading modules. Inference modules are unmodified upstream files.
    for name in ["prismatic", "prismatic.extern", "prismatic.extern.hf", "prismatic.training",
                 "prismatic.vla", "prismatic.models"]:
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split(".")))]
        module.__package__ = name
        sys.modules[name] = module
    sys.path.insert(0, str(root))


def tensor_numpy(value):
    return value.detach().float().cpu().numpy()


class InstrumentedPolicy:
    def __init__(self, name):
        self.name = name
        self.capture = ActivationCapture()
        self.writer = None
        self.inputs = {}
        self.runtime_arrays = {}
        self.metadata = {"model": name, "control_replan_steps": 1, "only_chunk_index_executed": 0,
                         "full_token_features": "initial policy timestep only",
                         "pooled_features": "every policy/control timestep, all instrumented layers",
                         "weights_quantized": False, "torch_version": torch.__version__,
                         "feature_dtype": "float16; pooling accumulates in float32",
                         "input_schema_version": 2,
                         "kv_features": "K/V linear projections before RoPE and attention caching",
                         "preprocessing_orientation": "official LIBERO observation[::-1, ::-1]"}
        if name.startswith("pi"):
            self.load_pi()
        else:
            self.load_vla()
        self.capture.vision_hooks(self.model)
        self.metadata["vision_module_names"] = self.capture.vision_module_names

    def load_pi(self):
        from openpi.training import config
        from openpi.policies import policy_config

        cfg = config.get_config(self.name)
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
        self.config = cfg
        checkpoint = ROOT / "checkpoints" / (self.name + "_pytorch")
        self.policy = policy_config.create_trained_policy(cfg, checkpoint, pytorch_device="cuda")
        self.model = self.policy._model
        llm = self.model.paligemma_with_expert.paligemma.language_model
        self.capture.layer_hooks(llm.layers, "vlm_hidden", projections=True)
        expert = self.model.paligemma_with_expert.gemma_expert.model
        self.capture.layer_hooks(expert.layers, "action_expert_hidden", "expert", projections=True)
        old_transform = self.policy._input_transform

        def transform(value):
            result = old_transform(value)
            self.inputs = result
            mask = np.asarray(result["tokenized_prompt_mask"])
            count = int(mask.sum())
            if count >= len(mask):
                raise ValueError("Prompt fills the token budget; shorten context before evaluation to rule out truncation")
            patch_count = 256
            text_start = 3 * patch_count
            self.capture.regions["vlm"] = [(0, patch_count), (patch_count, 2 * patch_count),
                                           (text_start, text_start + count),
                                           (text_start + count - 1, text_start + count)]
            self.capture.region_names["vlm"] = ["main_image", "wrist_image", "prompt", "last_prompt"]
            action_start = 0 if cfg.model.pi05 else 1
            self.capture.regions["expert"] = [(action_start, action_start + cfg.model.action_horizon),
                                              (action_start, action_start + 1)]
            self.capture.region_names["expert"] = ["all_action_tokens", "first_action_token"]
            return result

        self.policy._input_transform = transform
        self.metadata.update({"checkpoint": str(checkpoint.relative_to(ROOT)), "config": self.name,
                              "action_horizon": cfg.model.action_horizon, "action_dim_internal": cfg.model.action_dim,
                              "flow_denoising_steps": 10, "vlm_attention": "bidirectional observation prefix",
                              "state_in_prompt": cfg.model.discrete_state_input,
                              "state_in_action_expert": not cfg.model.pi05,
                              "model_architecture_source": "official OpenPI; official JAX-to-PyTorch conversion",
                              "torch_compile": False})

    def load_vla(self):
        oft = "oft" in self.name
        source = ROOT / "third_party" / ("openvla_oft" if oft else "openvla")
        inference_namespaces(source)
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from transformers import AutoTokenizer

        checkpoint = ROOT / "checkpoints" / self.name
        cfg = OpenVLAConfig.from_pretrained(checkpoint)
        extra = {} if oft else {"attn_implementation": "flash_attention_2"}
        self.model = OpenVLAForActionPrediction.from_pretrained(
            checkpoint, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **extra).to("cuda").eval()
        self.processor = PrismaticProcessor(PrismaticImageProcessor.from_pretrained(checkpoint),
                                             AutoTokenizer.from_pretrained(checkpoint))
        self.model.norm_stats = json.loads((checkpoint / "dataset_statistics.json").read_text())
        def embedding_input(module, args):
            if self.capture.enabled:
                index = sum(k.startswith("input/embedding_token_ids/") for k in self.runtime_arrays)
                self.runtime_arrays[f"input/embedding_token_ids/call_{index:02d}"] = args[0].detach().cpu().numpy()
        self.model.get_input_embeddings().register_forward_pre_hook(embedding_input)
        def llm_input(module, args, kwargs):
            if self.capture.enabled and kwargs.get("attention_mask") is not None:
                index = sum(k.startswith("input/llm_attention_mask/") for k in self.runtime_arrays)
                self.runtime_arrays[f"input/llm_attention_mask/call_{index:02d}"] = kwargs["attention_mask"].detach().cpu().numpy()
        self.model.language_model.register_forward_pre_hook(llm_input, with_kwargs=True)
        keys = [k for k in self.model.norm_stats if "libero_spatial" in k]
        if len(keys) != 1:
            raise ValueError(f"Ambiguous normalization key: {keys}")
        self.unnorm_key = keys[0]
        self.capture.layer_hooks(self.model.language_model.model.layers, "vlm_hidden", projections=True)
        self.proprio_projector = self.action_head = None
        if oft:
            from prismatic.models.action_heads import L1RegressionActionHead
            from prismatic.models.projectors import ProprioProjector

            self.model.vision_backbone.set_num_images_in_input(2)
            self.proprio_projector = ProprioProjector(llm_dim=self.model.llm_dim, proprio_dim=8)
            self.action_head = L1RegressionActionHead(input_dim=self.model.llm_dim,
                                                       hidden_dim=self.model.llm_dim, action_dim=7)
            for module, prefix in [(self.proprio_projector, "proprio_projector"), (self.action_head, "action_head")]:
                files = sorted(checkpoint.glob(prefix + "--*.pt"))
                if len(files) != 1:
                    raise ValueError(f"Expected one {prefix} checkpoint")
                weights = torch.load(files[0], map_location="cpu", weights_only=True)
                module.load_state_dict({k.removeprefix("module."): v for k, v in weights.items()}, strict=True)
                module.to(device="cuda", dtype=torch.bfloat16).eval()
            self.capture.layer_hooks(self.action_head.model.mlp_resnet_blocks, "action_head_hidden", "head")
            self.capture.region_names["head"] = ["action_chunk_mean"]
        else:
            original_generate = self.model.generate

            def generate(*args, **kwargs):
                result = original_generate(*args, **kwargs)
                self.generated_ids = result.detach().cpu().numpy()
                return result

            self.model.generate = generate
        self.metadata.update({"checkpoint": str(checkpoint.relative_to(ROOT)),
                              "normalization_key": self.unnorm_key, "center_crop_area": 0.9,
                              "action_horizon": 8 if oft else 1, "uses_wrist_image": oft, "uses_proprio": oft,
                              "vlm_attention": "bidirectional (official OFT fork)" if oft else "causal",
                              "attention_implementation": self.model.language_model.config._attn_implementation,
                              "has_separate_action_expert": False,
                              "action_head_type": "L1 MLP" if oft else "autoregressive action tokens"})

    @staticmethod
    def vla_image(raw):
        import tensorflow as tf

        image = np.ascontiguousarray(raw[::-1, ::-1])
        image = tf.image.encode_jpeg(image)
        image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
        image = tf.image.resize(image, (224, 224), method="lanczos3", antialias=True)
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
        image = tf.image.convert_image_dtype(image, tf.float32)
        side = np.sqrt(0.9)
        offset = (1 - side) / 2
        image = tf.image.crop_and_resize(image[None], [[offset, offset, offset + side, offset + side]], [0], (224, 224))[0]
        image = tf.image.convert_image_dtype(tf.clip_by_value(image, 0, 1), tf.uint8, saturate=True).numpy()
        return Image.fromarray(image).convert("RGB")

    @torch.inference_mode()
    def predict(self, packet, *, capture=True):
        self.capture.reset(full_tokens=packet["step"] == 0 and capture)
        self.capture.enabled = capture
        self.runtime_arrays = {}
        obs = packet["observation"]
        instruction = packet["instruction"]
        seed = int(packet["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        started = time.monotonic()
        arrays = {}
        if self.name.startswith("pi"):
            from openpi_client import image_tools

            image = image_tools.resize_with_pad(np.ascontiguousarray(obs["image"][::-1, ::-1]), 224, 224)
            wrist = image_tools.resize_with_pad(np.ascontiguousarray(obs["wrist_image"][::-1, ::-1]), 224, 224)
            noise = np.random.default_rng(seed).standard_normal(
                (self.config.model.action_horizon, self.config.model.action_dim)).astype(np.float32)
            output = self.policy.infer({"observation/image": image, "observation/wrist_image": wrist,
                                       "observation/state": obs["state"], "prompt": instruction}, noise=noise)
            chunk = np.asarray(output["actions"], dtype=np.float32)
            command = chunk[0].copy()
            arrays.update({"input/token_ids": self.inputs["tokenized_prompt"],
                           "input/token_mask": self.inputs["tokenized_prompt_mask"],
                           "input/normalized_state": self.inputs["state"], "input/flow_noise": noise,
                           "input/main_rgb_224": np.asarray(image), "input/wrist_rgb_224": np.asarray(wrist)})
            arrays["input/image_mask"] = np.asarray(list(self.inputs["image_mask"].values()), dtype=bool)
        else:
            oft = "oft" in self.name
            image = self.vla_image(obs["image"])
            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
            inputs = self.processor(prompt, image).to("cuda", dtype=torch.bfloat16)
            if oft:
                wrist = self.vla_image(obs["wrist_image"])
                wrist_inputs = self.processor(prompt, wrist).to("cuda", dtype=torch.bfloat16)
                inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)
                arrays["input/wrist_rgb_224"] = np.asarray(wrist)
            n = inputs["input_ids"].shape[1] + int(inputs["input_ids"][0, -1] != 29871)
            patches = self.model.vision_backbone.featurizer.patch_embed.num_patches
            visual_count = patches * (2 if oft else 1)
            text_start = 1 + visual_count + int(oft)
            prompt_end = text_start + n - 1
            regions = [(1, 1 + patches), (1 + patches, 1 + 2 * patches) if oft else (0, 0),
                       (text_start, prompt_end), (prompt_end - 1, prompt_end)]
            names = ["main_image", "wrist_image" if oft else "wrist_absent", "prompt_without_bos", "last_prompt"]
            if oft:
                regions.extend([(1 + visual_count, 2 + visual_count), (prompt_end - 1, prompt_end - 1 + 56)])
                names.extend(["proprio", "action_readout_positions"])
            self.capture.regions["vlm"] = regions
            self.capture.region_names["vlm"] = names
            arrays.update({"input/token_ids": inputs["input_ids"].cpu().numpy()[0],
                           "input/token_mask": inputs["attention_mask"].cpu().numpy()[0],
                           "input/main_rgb_224": np.asarray(image),
                           "input/pixel_values_bf16_bits": inputs["pixel_values"].view(torch.uint16).cpu().numpy()[0]})
            if oft:
                stats = self.model.norm_stats[self.unnorm_key]["proprio"]
                low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
                mask = stats.get("mask", np.ones_like(low, dtype=bool))
                proprio = np.clip(np.where(mask, 2 * (obs["state"] - low) / (high - low + 1e-8) - 1,
                                           obs["state"]), -1, 1)
                chunk, _ = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False,
                                                      proprio=proprio, proprio_projector=self.proprio_projector,
                                                      action_head=self.action_head, use_film=False)
                arrays["input/normalized_state"] = proprio
            else:
                chunk = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)[None]
                arrays["output/generated_token_ids"] = self.generated_ids[0]
            chunk = np.asarray(chunk, dtype=np.float32)
            command = chunk[0].copy()
            command[-1] = -np.sign(2 * command[-1] - 1)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
            raise ValueError(f"Invalid model output: {chunk}")
        elapsed = time.monotonic() - started
        if capture:
            arrays.update(self.runtime_arrays)
            arrays.update(self.capture.extract())
            arrays.update({"output/action_chunk": chunk, "output/submitted_action": command,
                           "input/raw_state": np.asarray(obs["state"]), "input/rng_seed": np.asarray(seed, dtype=np.int64)})
            if self.writer:
                self.writer.append(packet["step"], arrays, {
                    "step": packet["step"], "seed": seed, "inference_seconds": elapsed,
                    "region_names": self.capture.region_names, "region_spans": self.capture.regions,
                    "main_observation_sha256": hashlib.sha256(obs["image"].tobytes()).hexdigest(),
                    "instruction": instruction, "model_input_contains_safety_label": False,
                    "replan_step": True, "executed_chunk_index": 0,
                })
        return {"action": command, "action_chunk": chunk, "inference_seconds": elapsed,
                "feature_keys": sorted(k for k in arrays if k.startswith("pooled/"))}

    def begin(self, path):
        if self.writer:
            self.writer.close()
        self.writer = TimestepFeatureWriter(path, self.metadata)

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--smoke", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    policy = InstrumentedPolicy(args.model)
    print("MODEL_READY", json.dumps(policy.metadata), flush=True)
    if args.smoke:
        payload = dict(np.load(args.smoke / "native_observation.npz"))
        instruction = json.loads((args.smoke / "native_initial.json").read_text())["instruction"]
        output = args.smoke / (args.model + "_features.h5")
        policy.begin(output)
        packet = {"observation": payload, "instruction": instruction, "seed": 73000, "step": 0}
        result = policy.predict(packet)
        plain = policy.predict(packet, capture=False)
        difference = float(np.max(np.abs(result["action_chunk"] - plain["action_chunk"])))
        policy.close()
        report = {"metadata": policy.metadata, "action": result["action"].tolist(),
                  "feature_keys": result["feature_keys"], "hook_action_max_abs_difference": difference,
                  "inference_seconds": result["inference_seconds"]}
        (args.smoke / (args.model + "_smoke.json")).write_text(json.dumps(report, indent=2))
        assert difference < 1e-5, difference
        print("SMOKE_OK", json.dumps(report), flush=True)
        return
    if args.socket is None:
        parser.error("--socket or --smoke required")
    if args.socket.exists():
        raise FileExistsError(args.socket)
    with Listener(str(args.socket), family="AF_UNIX", authkey=b"vla-local-rollout-v2") as listener:
        print("LISTENING", args.socket, flush=True)
        with listener.accept() as connection:
            connection.send({"ready": True, "metadata": policy.metadata})
            while True:
                packet = connection.recv()
                try:
                    if packet["op"] == "close":
                        policy.close()
                        connection.send({"closed": True})
                        break
                    if packet["op"] == "begin":
                        policy.begin(packet["feature_path"])
                        connection.send({"begun": True})
                    elif packet["op"] == "end":
                        policy.close()
                        connection.send({"ended": True})
                    else:
                        connection.send(policy.predict(packet))
                except Exception:
                    connection.send({"error": traceback.format_exc()})
    args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
