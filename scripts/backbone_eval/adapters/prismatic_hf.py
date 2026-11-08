"""Prompt-only readout for the Llama-2 based Prismatic family: OpenVLA-7B, Prismatic prism-dinosiglip-224px+7b,
OpenVLA-OFT (PyTorch; runs inside the site PyTorch container with envs/ctpkgs on PYTHONPATH).

Sequence (HF modeling_prismatic): [BOS | 256 fused DINOv2+SigLIP patch tokens | "In: {body}\\nOut:"]
`h` = hidden_states[-1] (after the final RMSNorm) at the last valid text token (":" of "Out:").
`logits` = language_model.lm_head(h) -- the separate Llama LM head, never embed_tokens. FP32 projection and log_softmax
over the real vocabulary (len(tokenizer) = 32000 Llama-2 tokens + <PAD>); the 63 padding rows of the resized head are
excluded from the normalisation.

Variants
  openvla      : openvla/openvla-7b, causal attention (upstream transformers 4.40.1)
  prismatic    : TRI-ML/prismatic-vlms prism-dinosiglip-224px+7b `latest-checkpoint.pt`, re-mapped into the same HF
                 PrismaticForConditionalGeneration class (identical architecture/config to openvla-7b); tokenizer =
                 openvla-7b's Llama-2 tokenizer + <PAD> (the same construction Prismatic used); causal attention
  openvla_oft  : moojink/openvla-7b-oft-finetuned-libero-spatial (LoRA merged weights), single image, no proprio,
                 no action placeholders, L1/FiLM heads unused; attention = the OFT transformers fork's fully
                 bidirectional Llama (envs/oft_patch/modeling_llama.py injected into sys.modules in THIS process only)
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .base import CandidateCheck, ForwardOut, PrefixReader

ROOT = pathlib.Path(__file__).resolve().parents[3]
OFT_PATCH = ROOT / "envs/oft_patch"
MAX_TEXT_TOKENS = 1790  # llm_max_length 2048 minus 256 patches (BOS counted in the text tokens), small margin
# timm pretrained weights used by Prismatic for its frozen vision towers (timm 0.9.10 default sources)
TIMM_WEIGHTS = {
    "vit_large_patch14_reg4_dinov2.lvd142m": ROOT / "checkpoints/timm/vit_large_patch14_reg4_dinov2.lvd142m.safetensors",
    "vit_so400m_patch14_siglip_224": ROOT / "checkpoints/timm/ViT-SO400M-14-SigLIP.open_clip_pytorch_model.bin",
}


def _inject_oft_llama():
    """Replace transformers.models.llama.modeling_llama by the OFT fork's file BEFORE any Llama model is built."""
    if str(OFT_PATCH) not in sys.path:
        sys.path.insert(0, str(OFT_PATCH))  # provides the `prismatic` constants package OFT's modeling code imports
    import transformers  # noqa: F401  (lazy package; does not import modeling_llama)

    name = "transformers.models.llama.modeling_llama"
    assert name not in sys.modules, "modeling_llama already imported; OFT patch must be injected first"
    spec = importlib.util.spec_from_file_location(name, OFT_PATCH / "modeling_llama.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha1_file(p: pathlib.Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


class PrismaticHFReader(PrefixReader):
    family = "prismatic"

    def __init__(self, name: str, spec: dict, batch_size: int = 8, device: str = "cuda"):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.name, self.batch_size, self.device = name, batch_size, device
        self.kind = spec["kind"]
        ckpt = ROOT / spec["checkpoint"]
        t0 = time.time()
        if self.kind == "hf_oft":
            _inject_oft_llama()
        import transformers
        from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

        tok_dir = ROOT / spec.get("tokenizer_from", spec["checkpoint"])
        self.processor = AutoProcessor.from_pretrained(tok_dir, trust_remote_code=True)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "right"
        if self.kind in ("hf", "hf_oft"):
            model_class = AutoModelForVision2Seq
            load_kwargs = {"trust_remote_code": True}
            if spec.get("code_from"):
                # Official LIBERO releases reference the base model's code; resolve it offline from the local copy.
                from transformers.dynamic_module_utils import get_class_from_dynamic_module

                code_dir = ROOT / spec["code_from"]
                config_class = get_class_from_dynamic_module("configuration_prismatic.OpenVLAConfig", code_dir)
                model_class = get_class_from_dynamic_module("modeling_prismatic.OpenVLAForActionPrediction", code_dir)
                load_kwargs = {"config": config_class.from_pretrained(ckpt)}
            self.model, loading = model_class.from_pretrained(
                ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, output_loading_info=True, **load_kwargs
            )
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                if loading.get(key):
                    raise RuntimeError(f"Incomplete checkpoint load ({key}): {loading[key]}")
            load_info = {"loader": "HF from_pretrained with loading-info verification",
                         "missing_keys": loading.get("missing_keys", []),
                         "unexpected_keys": loading.get("unexpected_keys", []),
                         "code_from": spec.get("code_from", spec["checkpoint"])}
        elif self.kind == "tri_pt":
            self.model, load_info = self._load_tri_prismatic(ckpt, tok_dir, AutoConfig)
        else:
            raise ValueError(self.kind)
        self.model.eval().requires_grad_(False)
        self.model.to(device)
        if hasattr(self.model.vision_backbone, "set_num_images_in_input"):
            self.model.vision_backbone.set_num_images_in_input(1)
        lm = self.model.language_model
        self.lm_head = lm.get_output_embeddings()
        self.embed_tokens = lm.get_input_embeddings()
        self.n_patches = self._num_patches()
        self.vocab_size = len(self.tok)  # real vocabulary: 32000 + <PAD>
        self.hidden_dim = int(lm.config.hidden_size)
        self.max_text_tokens = MAX_TEXT_TOKENS
        self._head_w32 = self.lm_head.weight[: self.vocab_size].float()  # fp32 copy of the real rows
        llama_file = pathlib.Path(inspect.getsourcefile(type(lm.model)))
        self.attention = {
            "hf": "original causal Llama-2 attention (upstream transformers 4.40.1)",
            "tri_pt": "original causal Llama-2 attention (upstream transformers 4.40.1)",
            "hf_oft": "OFT fork: full bidirectional attention over all non-pad tokens (is_causal=False, pad-only mask)",
        }[self.kind]
        self.manifest = {
            "checkpoint": str(ckpt), "kind": self.kind, "dtype": "bfloat16 weights/activations; lm_head projection + log_softmax in fp32",
            "source": spec.get("source"),
            "compute_device": str(self.device), "gpu_name": torch.cuda.get_device_name(self.device),
            "physical_batch_limit": self.batch_size,
            "batch_execution": "singleton forwards; composite calls are split" if self.batch_size == 1 else "native padded batches",
            "head": "language_model.lm_head (separate Llama head; embed_tokens NOT used)",
            "tokenizer": f"{tok_dir} (Llama-2 sentencepiece + <PAD>={self.tok.pad_token_id}; len={len(self.tok)})",
            "vocab_rows_in_head": int(self.lm_head.weight.shape[0]), "real_vocab_used_for_softmax": self.vocab_size,
            "preprocessing": f"HF PrismaticImageProcessor from {tok_dir}: {self.processor.image_processor.image_resize_strategy}, 224px, "
                             f"DINOv2 (ImageNet mean/std) + SigLIP (0.5/0.5) channel-stacked",
            "prompt_format": "BOS [256 patches after BOS] 'In: {body}\\nOut:'  (PurePromptBuilder, rstripped)",
            "candidate_tokenization": "joint: ids(prompt + ' ' + label) must equal ids(prompt) + [one token]",
            "attn_implementation": getattr(lm.config, "_attn_implementation", None),
            "modeling_llama_file": str(llama_file), "modeling_llama_sha1": _sha1_file(llama_file),
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "num_images_in_input": getattr(self.model.vision_backbone, "num_images_in_input", 1), "n_patches": self.n_patches,
            "load_seconds": round(time.time() - t0, 1), **load_info,
        }

    # ------------------------------------------------------------------ loading helpers
    def _num_patches(self) -> int:
        with torch.no_grad():
            px = torch.zeros(1, 6, 224, 224, dtype=torch.bfloat16, device=self.device)
            return int(self.model.vision_backbone(px).shape[1])

    def _load_tri_prismatic(self, ckpt_dir: pathlib.Path, arch_dir: pathlib.Path, AutoConfig):
        """Build the HF Prismatic class from openvla-7b's config/code and load TRI's native state dict (strict)."""
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        tri_cfg = json.loads((ckpt_dir / "config.json").read_text())["model"]
        hf_cfg = AutoConfig.from_pretrained(arch_dir, trust_remote_code=True)
        for k_tri, k_hf in [("vision_backbone_id", "vision_backbone_id"), ("llm_backbone_id", "llm_backbone_id"),
                            ("arch_specifier", "arch_specifier"), ("image_resize_strategy", "image_resize_strategy")]:
            assert tri_cfg[k_tri] == getattr(hf_cfg, k_hf), (k_tri, tri_cfg[k_tri], getattr(hf_cfg, k_hf))
        cls = get_class_from_dynamic_module("modeling_prismatic.PrismaticForConditionalGeneration", arch_dir)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = cls(hf_cfg)
        finally:
            torch.set_default_dtype(prev)
        pt = ckpt_dir / "checkpoints/latest-checkpoint.pt"
        try:
            sd = torch.load(pt, map_location="cpu", weights_only=True)["model"]
        except Exception:  # noqa: BLE001  (older pickles with non-tensor objects)
            sd = torch.load(pt, map_location="cpu", weights_only=False)["model"]
        assert all(k in sd for k in ["projector", "llm_backbone"]), list(sd)
        vision_source = "checkpoint"
        if "vision_backbone" not in sd:
            # Prismatic keeps DINOv2 + SigLIP frozen and does not save them; upstream `PrismaticVLM.from_pretrained`
            # loads only projector + llm_backbone and materialises the vision towers from timm's pretrained weights
            # (timm.create_model(..., pretrained=True, num_classes=0, img_size=224)). We do the same from local files.
            sd["vision_backbone"] = self._timm_pretrained_vision_state(hf_cfg)
            vision_source = "timm pretrained (frozen in Prismatic training): " + ", ".join(TIMM_WEIGHTS[m].name for m in hf_cfg.timm_model_ids)
        proj_map = {"projector.0.weight": "projector.fc1.weight", "projector.0.bias": "projector.fc1.bias",
                    "projector.2.weight": "projector.fc2.weight", "projector.2.bias": "projector.fc2.bias",
                    "projector.4.weight": "projector.fc3.weight", "projector.4.bias": "projector.fc3.bias"}
        new = {}
        for k, v in sd["projector"].items():
            new[proj_map[k]] = v
        for k, v in sd["llm_backbone"].items():
            new[k.replace("llm.", "language_model.", 1)] = v
        for k, v in sd["vision_backbone"].items():
            if k.startswith("dino_featurizer"):
                k = k.replace(".gamma", ".scale_factor").replace("dino_featurizer.", "vision_backbone.featurizer.", 1)
            elif k.startswith("siglip_featurizer"):
                k = k.replace("siglip_featurizer.", "vision_backbone.fused_featurizer.", 1)
            new[k] = v
        res = model.load_state_dict(new, strict=True, assign=True)
        model.to(torch.bfloat16)
        info = {"loader": "TRI latest-checkpoint.pt -> HF key remap (openvla convert_openvla_weights_to_hf logic), strict=True",
                "vision_backbone_source": vision_source,
                "missing_keys": list(res.missing_keys), "unexpected_keys": list(res.unexpected_keys),
                "tri_config": {k: tri_cfg[k] for k in ["model_id", "vision_backbone_id", "llm_backbone_id", "arch_specifier", "image_resize_strategy"]},
                "lm_head_rows": int(new["language_model.lm_head.weight"].shape[0])}
        del sd, new
        return model, info

    @staticmethod
    def _timm_pretrained_vision_state(hf_cfg) -> dict:
        """State dict in Prismatic's `vision_backbone` naming (dino_featurizer.* / siglip_featurizer.*) from timm weights."""
        import timm

        out = {}
        for prefix, model_id, img_size in zip(["dino_featurizer.", "siglip_featurizer."], hf_cfg.timm_model_ids, hf_cfg.image_sizes, strict=True):
            f = TIMM_WEIGHTS[model_id]
            assert f.exists(), f"missing timm weights {f}"
            m = timm.create_model(model_id, pretrained=True, num_classes=0, img_size=img_size, pretrained_cfg_overlay={"file": str(f)})
            for k, v in m.state_dict().items():
                out[prefix + k] = v.to(torch.bfloat16)
            del m
        return out

    # ------------------------------------------------------------------ text
    def wrap(self, body: str) -> str:
        return f"In: {body.strip()}\nOut:"

    def prompt_ids(self, body: str) -> list[int]:
        ids = self.tok(self.wrap(body), add_special_tokens=True).input_ids
        assert ids[0] == self.tok.bos_token_id and ids[-1] != self.tok.eos_token_id
        return ids

    def candidate_checks(self, labels, sample_bodies):
        out = []
        special = {self.tok.bos_token_id, self.tok.eos_token_id, self.tok.pad_token_id, self.tok.unk_token_id}
        for lab in labels:
            ids_set, ok, notes = set(), True, []
            for b in sample_bodies[:5]:
                p = self.prompt_ids(b)
                full = self.tok(self.wrap(b) + " " + lab, add_special_tokens=True).input_ids
                good = full[: len(p)] == p and len(full) == len(p) + 1 and full[-1] not in special
                ok &= good
                ids_set.add(full[-1] if good else None)
            tid = next(iter(ids_set)) if ok and len(ids_set) == 1 else None
            piece = self.tok.convert_ids_to_tokens([tid]) if tid is not None else []
            out.append(CandidateCheck(lab, tid, piece, tid is not None,
                                      f"joint encoding of prompt+' {lab}' over {min(5, len(sample_bodies))} bodies; ids={sorted(map(str, ids_set))}"))
        return out

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, pils, bodies) -> ForwardOut:
        if len(bodies) > self.batch_size:
            pieces = [self.forward(pils[i:i+self.batch_size], bodies[i:i+self.batch_size])
                      for i in range(0, len(bodies), self.batch_size)]
            return ForwardOut(
                np.concatenate([p.h for p in pieces]), np.concatenate([p.logp for p in pieces]),
                np.concatenate([p.n_prompt_tokens for p in pieces]),
                {"physical_batch_limit": self.batch_size, "microbatches": len(pieces),
                 "head_vs_forward_max_abs": max(p.aux["head_vs_forward_max_abs"] for p in pieces)},
            )
        prompts = [self.wrap(b) for b in bodies]
        enc = self.tok(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
        input_ids, attn = enc.input_ids.to(self.device), enc.attention_mask.to(self.device)
        n_text = attn.sum(1)
        if int(n_text.max()) > self.max_text_tokens:
            raise ValueError(f"prompt too long: {int(n_text.max())}")
        px = self.processor.image_processor([im.convert("RGB") for im in pils], return_tensors="pt")["pixel_values"]
        px = px.to(self.device, dtype=torch.bfloat16)
        extra = {}
        if self.kind == "hf_oft":
            # OFT's forward derives action-token masks from `labels`; all-IGNORE labels = "no action tokens anywhere"
            # (pure VLM path: no placeholders, no proprio, no noisy actions). The resulting LM loss is unused.
            extra["labels"] = torch.full_like(input_ids, -100)
        out = self.model(input_ids=input_ids, attention_mask=attn, pixel_values=px, output_hidden_states=True, return_dict=True, **extra)
        pos = n_text + self.n_patches - 1  # BOS, patches, then the rest of the text; last valid text token
        ar = torch.arange(len(bodies), device=self.device)
        h = out.hidden_states[-1][ar, pos].float()  # final-normed
        logits_fwd = out.logits[ar, pos, : self.vocab_size].float()
        logits = F.linear(h, self._head_w32)
        logp = F.log_softmax(logits, dim=-1)
        head_vs_fwd = float((logits - logits_fwd).abs().max())  # bf16 head vs fp32 head on the same h
        return ForwardOut(h.cpu().numpy(), logp.cpu().numpy(), n_text.cpu().numpy(), {"head_vs_forward_max_abs": head_vs_fwd})

    # ------------------------------------------------------------------ validation hooks
    def check_head(self) -> dict:
        w_head, w_emb = self.lm_head.weight, self.embed_tokens.weight
        same_storage = w_head.data_ptr() == w_emb.data_ptr()
        equal = bool(torch.equal(w_head[:1000].float(), w_emb[:1000].float()))
        return {"head_type": "separate_lm_head", "lm_head_shape": list(w_head.shape), "embed_tokens_shape": list(w_emb.shape),
                "shares_storage_with_embed_tokens": same_storage, "equal_to_embed_tokens": equal,
                "real_vocab_used": self.vocab_size, "ok": (not same_storage) and (not equal)}

    @torch.no_grad()
    def check_attention_direction(self) -> dict:
        """Text-only LLM probe: change the last token, see whether earlier final-normed states move."""
        a = self.tok("In: Is there a dog in the image?\nOut:", return_tensors="pt").input_ids.to(self.device)
        b = a.clone()
        b[0, -1] = self.tok(" cat", add_special_tokens=False).input_ids[-1]
        lm = self.model.language_model
        ha = lm(input_ids=a, output_hidden_states=True).hidden_states[-1][0, :-1].float()
        hb = lm(input_ids=b, output_hidden_states=True).hidden_states[-1][0, :-1].float()
        diff = float((ha - hb).abs().max())
        expected = self.kind == "hf_oft"
        return {"test": "perturb last token, compare earlier positions (text-only LLM)", "max_abs_change_earlier_positions": diff,
                "earlier_positions_affected": diff > 1e-3, "expected_bidirectional": expected, "ok": (diff > 1e-3) == expected,
                "modeling_llama_file": self.manifest["modeling_llama_file"], "attn_implementation": self.manifest["attn_implementation"]}

    def param_fingerprint(self) -> str:
        hsh = hashlib.sha1()
        params = list(self.model.parameters())
        for p in params[:: max(1, len(params) // 16)][:16]:
            hsh.update(p.detach().flatten()[:64].float().cpu().numpy().tobytes())
        hsh.update(self.lm_head.weight[:4, :64].float().cpu().numpy().tobytes())
        return hsh.hexdigest()
