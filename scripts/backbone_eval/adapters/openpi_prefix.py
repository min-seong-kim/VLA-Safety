"""Prompt-only readout of the PaliGemma backbone inside an openpi pi0 / pi0.5 checkpoint (JAX, openpi/.venv).

Sequence (prefix-LM, as in PaliGemma pre-training and openpi `embed_prefix`):
    [256 SigLIP tokens | BOS "answer en {body}" ... "\\n"]     all bidirectional, no answer tokens
`h` = final-normed hidden state at the trailing "\\n" (the answer-start boundary).
`logits` = h @ E^T with E = the checkpoint's own `PaliGemma.llm.embedder.input_embedding` (Gemma tied embedding;
openpi's `Embedder.decode`). FP32 accumulation, full-vocabulary log_softmax. The action expert is never run.
"""

from __future__ import annotations

import hashlib
import pathlib
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.shared.image_tools import resize_with_pad

from .base import CandidateCheck, ForwardOut, PrefixReader

ROOT = pathlib.Path(__file__).resolve().parents[3]
NUM_IMAGE_TOKENS = 256
PAD_ID = 0
BUCKETS = (64, 128, 192, 256, 384, 512, 768, 1024)


class OpenPiPrefixReader(PrefixReader):
    family = "paligemma"
    attention = "PaliGemma prefix-LM: image+prompt fully bidirectional (openpi make_attn_mask with ar_mask=False), no suffix"

    def __init__(self, name: str, spec: dict, batch_size: int = 16):
        self.name, self.batch_size = name, batch_size
        if jax.default_backend() != "gpu":
            raise RuntimeError("Backbone extraction requires a GPU allocation; refusing CPU fallback")
        ckpt = ROOT / spec["checkpoint"]
        self.tok = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "checkpoints/tokenizer/paligemma_tokenizer.model"))
        self.nl_ids = self.tok.encode("\n")
        assert len(self.nl_ids) == 1, self.nl_ids
        t0 = time.time()
        cfg = pi0_config.Pi0Config(pi05=spec["pi05"], **spec.get("model_config", {}))
        params = _model.restore_params(ckpt / "params", dtype=jnp.bfloat16)
        self.model = cfg.load(params)
        embed = params["PaliGemma"]["llm"]["embedder"]["input_embedding"]
        if isinstance(embed, dict):
            embed = embed["value"]
        self._embed = jnp.asarray(embed, dtype=jnp.bfloat16)  # [V, D]
        del params
        self.vocab_size, self.hidden_dim = int(self._embed.shape[0]), int(self._embed.shape[1])
        self.max_text_tokens = BUCKETS[-1]
        self._graphdef, self._state = nnx.split(self.model)
        self._fwd = jax.jit(self._fwd_impl)
        self._cpu = jax.devices("cpu")[0]
        self.manifest = {
            "checkpoint": str(ckpt), "pi05": spec["pi05"], "dtype": "bfloat16 (logits fp32 accumulation)",
            "source": spec.get("source"),
            "compute_backend": jax.default_backend(), "compute_devices": [str(d) for d in jax.devices()],
            "model_config": {"pi05": cfg.pi05, "action_horizon": cfg.action_horizon,
                             "discrete_state_input": cfg.discrete_state_input},
            "head": "tied input embedding of the checkpoint (openpi Embedder.decode); no separate LM head exists",
            "tokenizer": "checkpoints/tokenizer/paligemma_tokenizer.model (vocab %d)" % self.tok.get_piece_size(),
            "embedding_table_shape": [self.vocab_size, self.hidden_dim],
            "preprocessing": "resize_with_pad(224,224) geometry of openpi.shared.image_tools (aspect-preserving, black letterbox) "
                             "re-implemented with PIL antialiased bilinear on uint8, then [-1,1]",
            "prompt_format": "[image] BOS 'answer en {body}' '\\n'  (newlines inside body kept)",
            "candidate_tokenization": "suffix encoded separately (big_vision/openpi convention): id = sp.encode(label)[0], must be 1 piece",
            "load_seconds": round(time.time() - t0, 1),
        }

    # ------------------------------------------------------------------ text
    def wrap(self, body: str) -> str:
        return f"answer en {body.strip()}"

    def prompt_ids(self, body: str) -> list[int]:
        return self.tok.encode(self.wrap(body), add_bos=True) + self.nl_ids

    def candidate_checks(self, labels, sample_bodies):
        out = []
        for lab in labels:
            ids = self.tok.encode(lab)
            pieces = [self.tok.id_to_piece(i) for i in ids]
            ok = len(ids) == 1 and self.tok.decode(ids) == lab and ids[0] not in (PAD_ID, self.tok.bos_id(), self.tok.eos_id(), self.tok.unk_id())
            # cross-check: joint encoding of "<prompt>\n<label>" must end with the same single piece
            joint_ok = all(self.tok.encode(self.wrap(b) + "\n" + lab)[-1] == ids[0] for b in sample_bodies[:3]) if ok else False
            out.append(CandidateCheck(lab, ids[0] if ok else None, pieces, ok,
                                      f"separate-encode piece={pieces}; joint-encode tail matches={joint_ok}"))
        return out

    # ------------------------------------------------------------------ forward
    def _fwd_impl(self, state, embed, images, tokens, tok_mask, last_idx):
        m = nnx.merge(self._graphdef, state)
        b = images.shape[0]
        img_tokens, _ = m.PaliGemma.img(images, train=False)
        txt_tokens = m.PaliGemma.llm(tokens, method="embed")
        x = jnp.concatenate([img_tokens, txt_tokens], axis=1)
        in_mask = jnp.concatenate([jnp.ones((b, NUM_IMAGE_TOKENS), bool), tok_mask], axis=1)
        ar = jnp.zeros_like(in_mask)  # everything is prefix -> bidirectional
        attn_mask = make_attn_mask(in_mask, ar)
        positions = jnp.cumsum(in_mask, axis=1) - 1
        (out, _), _ = m.PaliGemma.llm([x, None], mask=attn_mask, positions=positions)  # final RMSNorm applied inside
        h = out[jnp.arange(b), NUM_IMAGE_TOKENS + last_idx]  # [b, D] bf16, last prompt token ("\n")
        logits = jnp.einsum("bd,vd->bv", h, embed, preferred_element_type=jnp.float32)
        logp = jax.nn.log_softmax(logits, axis=-1)
        return h.astype(jnp.float32), logp, out

    @staticmethod
    def resize_with_pad_pil(im, height: int = 224, width: int = 224) -> np.ndarray:
        """Same geometry as openpi.shared.image_tools.resize_with_pad (tf.image.resize_with_pad semantics): keep aspect
        ratio, int() the resized size, centre with the remainder on the bottom/right, pad black. Resampling uses PIL's
        antialiased bilinear (triangle) filter, the same kernel family as jax.image.resize(method="linear", antialias=True);
        only float rounding differs. (A per-image-shape XLA/CPU compile crashed on datasets with many distinct image sizes.)"""
        from PIL import Image

        im = im.convert("RGB")
        w, h = im.size
        ratio = max(w / width, h / height)
        rh, rw = int(h / ratio), int(w / ratio)
        resized = np.asarray(im.resize((rw, rh), Image.BILINEAR), dtype=np.uint8)
        pad_h0, rem_h = divmod(height - rh, 2)
        pad_w0, rem_w = divmod(width - rw, 2)
        return np.pad(resized, ((pad_h0, pad_h0 + rem_h), (pad_w0, pad_w0 + rem_w), (0, 0)), constant_values=0)

    def preprocess_images(self, pils) -> np.ndarray:
        arrs = [self.resize_with_pad_pil(im) for im in pils]
        return np.stack(arrs).astype(np.float32) / 255.0 * 2.0 - 1.0

    def forward(self, pils, bodies) -> ForwardOut:
        n = len(bodies)
        ids = [self.prompt_ids(b) for b in bodies]
        lens = np.array([len(i) for i in ids])
        if lens.max() > BUCKETS[-1]:
            raise ValueError(f"prompt too long: {lens.max()} > {BUCKETS[-1]}")
        L = next(bk for bk in BUCKETS if bk >= lens.max())
        B = self.batch_size
        assert n <= B
        tokens = np.full((B, L), PAD_ID, np.int32)
        mask = np.zeros((B, L), bool)
        for i, seq in enumerate(ids):
            tokens[i, : len(seq)] = seq
            mask[i, : len(seq)] = True
        last_idx = np.zeros(B, np.int32)
        last_idx[:n] = lens - 1
        images = np.zeros((B, 224, 224, 3), np.float32)
        images[:n] = self.preprocess_images(pils)
        if n < B:  # pad the batch with copies of the last example (fixed shapes for jit); sliced off below
            tokens[n:], mask[n:], images[n:], last_idx[n:] = tokens[n - 1], mask[n - 1], images[n - 1], last_idx[n - 1]
        h, logp, _ = self._fwd(self._state, self._embed, jnp.asarray(images), jnp.asarray(tokens), jnp.asarray(mask),
                               jnp.asarray(last_idx))
        return ForwardOut(np.asarray(h)[:n], np.asarray(logp)[:n], lens, {"bucket": L})

    # ------------------------------------------------------------------ validation hooks
    def check_head(self) -> dict:
        e = np.asarray(self._embed[:, :8].astype(jnp.float32))
        return {"head_type": "tied_input_embedding", "embedding_shape": list(self._embed.shape),
                "embedding_sample_abs_mean": float(np.abs(e).mean()), "note": "openpi Embedder.decode(x) = x @ E^T; identical to this readout"}

    def check_attention_direction(self) -> dict:
        """Change the LAST prompt token and look whether an EARLIER position's hidden state moves (prefix is bidirectional)."""
        body = "Is there a dog in the image?\nAnswer yes or no."
        from PIL import Image

        img = Image.fromarray(np.full((224, 224, 3), 127, np.uint8))
        ids = self.prompt_ids(body)
        ids2 = ids[:-1] + [self.tok.encode("?")[-1]]  # replace the trailing "\n"
        B, L = self.batch_size, next(bk for bk in BUCKETS if bk >= len(ids))
        outs = []
        for seq in (ids, ids2):
            tokens = np.full((B, L), PAD_ID, np.int32); tokens[:, : len(seq)] = seq
            mask = np.zeros((B, L), bool); mask[:, : len(seq)] = True
            imgs = np.repeat(self.preprocess_images([img]), B, axis=0)
            _, _, out = self._fwd(self._state, self._embed, jnp.asarray(imgs), jnp.asarray(tokens), jnp.asarray(mask),
                                  jnp.asarray(np.full(B, len(seq) - 1, np.int32)))
            outs.append(np.asarray(out[0, NUM_IMAGE_TOKENS : NUM_IMAGE_TOKENS + len(seq) - 1].astype(jnp.float32)))
        diff = float(np.abs(outs[0] - outs[1]).max())
        return {"test": "perturb last prompt token, compare earlier prompt positions", "max_abs_change_earlier_positions": diff,
                "earlier_positions_affected": diff > 1e-3, "expected": "True (prefix-LM bidirectional)"}

    def param_fingerprint(self) -> str:
        leaves = jax.tree.leaves(self._state)
        hsh = hashlib.sha1()
        for leaf in leaves[::max(1, len(leaves) // 16)][:16]:
            if hasattr(leaf, "shape") and leaf.size:
                hsh.update(np.asarray(jnp.ravel(leaf)[:64].astype(jnp.float32)).tobytes())
        hsh.update(np.asarray(self._embed[:4, :64].astype(jnp.float32)).tobytes())
        return hsh.hexdigest()
