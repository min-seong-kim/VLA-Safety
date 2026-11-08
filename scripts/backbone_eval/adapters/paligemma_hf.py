"""Prompt-only readout of the ORIGINAL PaliGemma-3B-pt-224 (HF `google/paligemma-3b-pt-224`) -- the baseline the Pi models
were initialised from. Runs inside the PyTorch container with envs/ctpkgs_pg (transformers 4.44.2) ahead of envs/ctpkgs.

Sequence (HF PaliGemmaProcessor): [256 <image> tokens | BOS | prompt | "\\n"], all bidirectional (prefix only; at inference
HF's PaliGemma mask is fully bidirectional over the input). `h` = hidden_states[-1] (post final RMSNorm) at the trailing
"\\n" token. `logits` = lm_head(h) = h @ E^T (Gemma ties lm_head to the input embedding), fp32, softmax over the real
257,152-token vocabulary (the head has 257,216 rows; the 64 padding rows are excluded).

Variants
  paligemma     : official HF preprocessing (SiglipImageProcessor: plain resize to 224x224 bicubic, mean/std 0.5)
  paligemma_lb  : the same model, but the image is first letterboxed to a 224 square exactly like openpi's resize_with_pad
                  (the Pi models' preprocessing) so the Pi-vs-PaliGemma comparison has matched inputs
"""

from __future__ import annotations

import hashlib
import inspect
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from .base import CandidateCheck, ForwardOut, PrefixReader

ROOT = pathlib.Path(__file__).resolve().parents[3]
MAX_TEXT_TOKENS = 1024  # same bucket ceiling as the openpi adapter (PaliGemma pre-training used short text)


def letterbox_224(im):
    """openpi resize_with_pad geometry with PIL antialiased bilinear (same function as adapters/openpi_prefix.py)."""
    from PIL import Image

    im = im.convert("RGB")
    w, h = im.size
    ratio = max(w / 224, h / 224)
    rh, rw = int(h / ratio), int(w / ratio)
    resized = np.asarray(im.resize((rw, rh), Image.BILINEAR), dtype=np.uint8)
    pad_h0, rem_h = divmod(224 - rh, 2)
    pad_w0, rem_w = divmod(224 - rw, 2)
    arr = np.pad(resized, ((pad_h0, pad_h0 + rem_h), (pad_w0, pad_w0 + rem_w), (0, 0)), constant_values=0)
    return Image.fromarray(arr)


class PaliGemmaHFReader(PrefixReader):
    family = "paligemma"
    attention = "HF PaliGemma inference mask: image+prompt fully bidirectional (prefix-LM), no suffix"

    def __init__(self, name: str, spec: dict, batch_size: int = 16, device: str = "cuda"):
        self.name, self.batch_size, self.device = name, batch_size, device
        self.letterbox = bool(spec.get("letterbox", False))
        ckpt = ROOT / spec["checkpoint"]
        t0 = time.time()
        import transformers
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 41), transformers.__version__
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.tok = self.processor.tokenizer
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.model.eval().requires_grad_(False).to(device)
        lm = self.model.language_model
        self.lm_head = lm.get_output_embeddings()
        self.embed_tokens = lm.get_input_embeddings()
        self.vocab_size = int(self.tok.vocab_size)  # 257152 sentencepiece tokens; excludes the added <image> token and the 64 padding rows
        self.hidden_dim = int(lm.config.hidden_size)
        self.max_text_tokens = MAX_TEXT_TOKENS
        self._head_w32 = self.lm_head.weight[: self.vocab_size].float()
        self.n_image_tokens = int(self.model.config.text_config.num_image_tokens if hasattr(self.model.config.text_config, "num_image_tokens")
                                  else getattr(self.model.config, "num_image_tokens", 256))
        pg_file = pathlib.Path(inspect.getsourcefile(type(self.model)))
        self.manifest = {
            "checkpoint": str(ckpt), "kind": "hf_paligemma", "letterbox": self.letterbox,
            "dtype": "bfloat16 weights/activations; lm_head projection + log_softmax in fp32",
            "head": "language_model.lm_head, tied to embed_tokens (Gemma design; same head as the Pi readout uses)",
            "tokenizer": f"{ckpt} (PaliGemma sentencepiece; len={len(self.tok)}; padding_side={self.tok.padding_side})",
            "vocab_rows_in_head": int(self.lm_head.weight.shape[0]), "real_vocab_used_for_softmax": self.vocab_size,
            "preprocessing": ("openpi-style letterbox to 224 (PIL bilinear, black pad) then " if self.letterbox else "")
                             + f"HF {type(self.processor.image_processor).__name__}: resize {self.processor.image_processor.size}, "
                             f"resample={self.processor.image_processor.resample}, mean/std {self.processor.image_processor.image_mean}/{self.processor.image_processor.image_std}",
            "prompt_format": "[<image>*256] BOS 'answer en {body}' '\\n' (processor appends the newline; newlines inside body kept)",
            "candidate_tokenization": "suffix encoded separately (PaliGemma convention): id = tok(label, add_special_tokens=False), must be 1 token",
            "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            "modeling_file": str(pg_file), "modeling_sha1": hashlib.sha1(pg_file.read_bytes()).hexdigest(),
            "n_image_tokens": self.n_image_tokens, "load_seconds": round(time.time() - t0, 1),
        }

    # ------------------------------------------------------------------ text
    def wrap(self, body: str) -> str:
        return f"answer en {body.strip()}"

    def prompt_ids(self, body: str) -> list[int]:
        # text tokens only (BOS + prompt + "\n"), as the processor builds them; image tokens are not counted here
        return [self.tok.bos_token_id] + self.tok(self.wrap(body), add_special_tokens=False).input_ids + self.tok("\n", add_special_tokens=False).input_ids

    def candidate_checks(self, labels, sample_bodies):
        out = []
        special = {self.tok.bos_token_id, self.tok.eos_token_id, self.tok.pad_token_id, self.tok.unk_token_id}
        for lab in labels:
            ids = self.tok(lab, add_special_tokens=False).input_ids
            ok = len(ids) == 1 and ids[0] not in special and self.tok.decode(ids) == lab
            joint_ok = all(self.tok(self.wrap(b) + "\n" + lab, add_special_tokens=False).input_ids[-1] == ids[0] for b in sample_bodies[:3]) if ok else False
            out.append(CandidateCheck(lab, ids[0] if ok else None, self.tok.convert_ids_to_tokens(ids), ok,
                                      f"separate encoding; joint-encode tail matches={joint_ok}"))
        return out

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, pils, bodies) -> ForwardOut:
        imgs = [letterbox_224(im) if self.letterbox else im.convert("RGB") for im in pils]
        enc = self.processor(text=[self.wrap(b) for b in bodies], images=imgs, return_tensors="pt", padding="longest")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        px = enc["pixel_values"].to(self.device, dtype=torch.bfloat16)
        n_text = (attn.sum(1) - self.n_image_tokens)
        if int(n_text.max()) > self.max_text_tokens:
            raise ValueError(f"prompt too long: {int(n_text.max())}")
        out = self.model(input_ids=input_ids, attention_mask=attn, pixel_values=px, output_hidden_states=True, return_dict=True)
        # last valid position regardless of padding side
        ar_pos = torch.arange(input_ids.shape[1], device=self.device)
        pos = (attn * ar_pos).argmax(1)
        b = torch.arange(len(bodies), device=self.device)
        assert bool((input_ids[b, pos] == self.tok("\n", add_special_tokens=False).input_ids[0]).all()), "last token is not the newline boundary"
        h = out.hidden_states[-1][b, pos].float()
        logits_fwd = out.logits[b, pos, : self.vocab_size].float()
        logits = F.linear(h, self._head_w32)
        logp = F.log_softmax(logits, dim=-1)
        return ForwardOut(h.cpu().numpy(), logp.cpu().numpy(), n_text.cpu().numpy(),
                          {"head_vs_forward_max_abs": float((logits - logits_fwd).abs().max())})

    # ------------------------------------------------------------------ validation hooks
    def check_head(self) -> dict:
        w_head, w_emb = self.lm_head.weight, self.embed_tokens.weight
        return {"head_type": "tied_input_embedding (HF lm_head shares embed_tokens)", "lm_head_shape": list(w_head.shape),
                "shares_storage_with_embed_tokens": w_head.data_ptr() == w_emb.data_ptr(),
                "equal_to_embed_tokens": bool(torch.equal(w_head[:1000].float(), w_emb[:1000].float())), "real_vocab_used": self.vocab_size,
                "ok": bool(torch.equal(w_head[:1000].float(), w_emb[:1000].float()))}

    @torch.no_grad()
    def check_attention_direction(self) -> dict:
        """Change the trailing '\\n' of a real multimodal input and see whether earlier text positions move (prefix-LM -> yes)."""
        from PIL import Image

        img = Image.fromarray(np.full((224, 224, 3), 127, np.uint8))
        enc = self.processor(text=[self.wrap("Is there a dog in the image?\nAnswer yes or no.")], images=[img], return_tensors="pt")
        ids = enc["input_ids"].to(self.device); px = enc["pixel_values"].to(self.device, dtype=torch.bfloat16); attn = enc["attention_mask"].to(self.device)
        ids2 = ids.clone(); ids2[0, -1] = self.tok("?", add_special_tokens=False).input_ids[-1]
        ha = self.model(input_ids=ids, attention_mask=attn, pixel_values=px, output_hidden_states=True).hidden_states[-1][0, self.n_image_tokens:-1].float()
        hb = self.model(input_ids=ids2, attention_mask=attn, pixel_values=px, output_hidden_states=True).hidden_states[-1][0, self.n_image_tokens:-1].float()
        diff = float((ha - hb).abs().max())
        return {"test": "perturb trailing newline, compare earlier prompt positions (multimodal input)", "max_abs_change_earlier_positions": diff,
                "earlier_positions_affected": diff > 1e-3, "expected_bidirectional": True, "ok": diff > 1e-3,
                "attn_implementation": getattr(self.model.config, "_attn_implementation", None)}

    def param_fingerprint(self) -> str:
        hsh = hashlib.sha1()
        params = list(self.model.parameters())
        for p in params[:: max(1, len(params) // 16)][:16]:
            hsh.update(p.detach().flatten()[:64].float().cpu().numpy().tobytes())
        return hsh.hexdigest()
