"""Expose the PaliGemma VLM backbone inside an openpi pi0 / pi0.5 checkpoint as a text scorer / generator.

The action expert is loaded but never run: only the image tower, the token embedder and the first
(PaliGemma) expert of the mixture transformer are used.  Next-token logits come from the tied
embedding table (Gemma has no separate LM head), exactly as in big_vision / pi0-FAST.

Sequence layout per example (prefix-LM, same as PaliGemma training and openpi's `embed_prefix`):

    [256 image tokens | BOS prompt... "\\n" | answer tokens ...]
     bidirectional      bidirectional        causal

Logits at position i predict token i+1, so the "\\n" token predicts the first answer token.
"""

from __future__ import annotations

import dataclasses
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask

NUM_IMAGE_TOKENS = 256  # SigLIP So400m/14 at 224x224
PAD_ID = 0
EOS_ID = 1


@dataclasses.dataclass
class Encoded:
    tokens: np.ndarray  # int32 [b, L]
    mask: np.ndarray  # bool [b, L], True on real tokens
    ar_mask: np.ndarray  # bool [b, L], True on causal (answer) tokens
    prefix_len: np.ndarray  # int [b]
    suffix_len: np.ndarray  # int [b]


class BackboneLM:
    def __init__(
        self,
        checkpoint_dir: str | pathlib.Path,
        *,
        pi05: bool,
        tokenizer_path: str | pathlib.Path,
        max_len: int = 192,
    ):
        self.max_len = max_len
        self.tok = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
        cfg = pi0_config.Pi0Config(pi05=pi05)
        params = _model.restore_params(pathlib.Path(checkpoint_dir) / "params", dtype=jnp.bfloat16)
        self.model = cfg.load(params)
        embed = params["PaliGemma"]["llm"]["embedder"]["input_embedding"]
        if isinstance(embed, dict):  # some checkpoints wrap leaves as {"value": ...}
            embed = embed["value"]
        self._embed = jnp.asarray(embed, dtype=jnp.bfloat16)  # [V, D]
        del params

        graphdef, state = nnx.split(self.model)
        self._state = state

        def fwd(state, embed, images, tokens, tok_mask, ar_mask, targets):
            m = nnx.merge(graphdef, state)
            b = images.shape[0]
            img_tokens, _ = m.PaliGemma.img(images, train=False)  # [b, 256, D]
            txt_tokens = m.PaliGemma.llm(tokens, method="embed")  # [b, L, D]
            x = jnp.concatenate([img_tokens, txt_tokens], axis=1)
            in_mask = jnp.concatenate([jnp.ones((b, NUM_IMAGE_TOKENS), bool), tok_mask], axis=1)
            ar = jnp.concatenate([jnp.zeros((b, NUM_IMAGE_TOKENS), bool), ar_mask], axis=1)
            attn_mask = make_attn_mask(in_mask, ar)
            positions = jnp.cumsum(in_mask, axis=1) - 1
            (out, _), _ = m.PaliGemma.llm([x, None], mask=attn_mask, positions=positions)
            h = out[:, NUM_IMAGE_TOKENS:, :]  # text positions only, [b, L, D]
            logits = jnp.einsum("bld,vd->blv", h, embed, preferred_element_type=jnp.float32)
            logp = jax.nn.log_softmax(logits, axis=-1)
            target_logp = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]  # [b, L]
            argmax = jnp.argmax(logits, axis=-1)  # [b, L]
            return target_logp, argmax

        self._fwd = jax.jit(fwd)

    # ---------------------------------------------------------------- tokenization
    def _prefix_ids(self, prompt: str) -> list[int]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        return self.tok.encode(cleaned, add_bos=True) + self.tok.encode("\n")

    def encode(self, prompts: list[str], answers: list[str] | None = None, *, add_eos: bool = True) -> Encoded:
        b = len(prompts)
        tokens = np.full((b, self.max_len), PAD_ID, np.int32)
        mask = np.zeros((b, self.max_len), bool)
        ar = np.zeros((b, self.max_len), bool)
        plen = np.zeros(b, np.int32)
        slen = np.zeros(b, np.int32)
        for i, p in enumerate(prompts):
            pre = self._prefix_ids(p)
            suf = []
            if answers is not None:
                suf = self.tok.encode(answers[i].strip()) + ([EOS_ID] if add_eos else [])
            if len(pre) + len(suf) > self.max_len:
                # Never truncate the answer: shorten the prompt body, keeping BOS and the trailing "\n".
                keep = max(self.max_len - len(suf) - 2, 0)
                pre = [pre[0]] + pre[1:-1][:keep] + [pre[-1]]
            seq = (pre + suf)[: self.max_len]
            n_pre = min(len(pre), self.max_len)
            tokens[i, : len(seq)] = seq
            mask[i, : len(seq)] = True
            ar[i, n_pre : len(seq)] = True
            plen[i] = n_pre
            slen[i] = len(seq) - n_pre
        return Encoded(tokens, mask, ar, plen, slen)

    # ---------------------------------------------------------------- forward
    def _run(self, images: np.ndarray, enc: Encoded):
        images = np.asarray(images)
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0 * 2.0 - 1.0
        assert images.shape[1:] == (224, 224, 3), images.shape
        targets = np.roll(enc.tokens, -1, axis=1)  # token at i+1 is the target for position i
        logp, argmax = self._fwd(
            self._state, self._embed, jnp.asarray(images), jnp.asarray(enc.tokens),
            jnp.asarray(enc.mask), jnp.asarray(enc.ar_mask), jnp.asarray(targets),
        )
        return np.asarray(logp), np.asarray(argmax)

    def score(self, images: np.ndarray, prompts: list[str], answers: list[str], *, add_eos: bool = True):
        """Log-likelihood of each answer given image+prompt. Returns (sum_logp [b], n_tokens [b])."""
        enc = self.encode(prompts, answers, add_eos=add_eos)
        logp, _ = self._run(images, enc)
        sums = np.zeros(len(prompts), np.float64)
        for i in range(len(prompts)):
            lo, hi = enc.prefix_len[i] - 1, enc.prefix_len[i] + enc.suffix_len[i] - 1
            sums[i] = logp[i, lo:hi].sum()
        return sums, enc.suffix_len.astype(np.int64)

    def generate(self, images: np.ndarray, prompts: list[str], *, max_new_tokens: int = 32) -> list[str]:
        """Greedy decoding by re-running the full (fixed-shape) forward pass each step."""
        enc = self.encode(prompts, None)
        b = len(prompts)
        done = np.zeros(b, bool)
        cur = enc.prefix_len.copy()
        for _ in range(max_new_tokens):
            _, argmax = self._run(images, enc)
            for i in range(b):
                if done[i] or cur[i] >= self.max_len:
                    done[i] = True
                    continue
                nxt = int(argmax[i, cur[i] - 1])
                enc.tokens[i, cur[i]] = nxt
                enc.mask[i, cur[i]] = True
                enc.ar_mask[i, cur[i]] = True
                cur[i] += 1
                if nxt == EOS_ID:
                    done[i] = True
            if done.all():
                break
        outs = []
        for i in range(b):
            ids = [int(t) for t in enc.tokens[i, enc.prefix_len[i] : cur[i]] if t not in (EOS_ID, PAD_ID)]
            outs.append(self.tok.decode(ids).strip())
        return outs
