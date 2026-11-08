# VLA-Safety — VLM backbone evaluation of π0 / π0.5

Goal: measure how much of the PaliGemma-3B vision-language ability survives inside the VLA
checkpoints `pi0_base`, `pi0_libero`, `pi05_base`, `pi05_libero` (openpi).

Follow `../HANGANG_AGENT_GUIDE.md` for every GPU job (partition `natStr`, submit from this dir).

## Layout
- `openpi/` — upstream clone; `openpi/.venv` is the project-local uv environment (JAX 0.5.3 + CUDA, aarch64).
- `tools/uv` — project-local uv binary. `scripts/setup_env.sh` rebuilds the venv.
- `checkpoints/<name>/` — downloaded with `scripts/download_ckpt.sh` (plain HTTPS from the public GCS bucket).
- `checkpoints/tokenizer/paligemma_tokenizer.model` — PaliGemma sentencepiece tokenizer.
- `data/<bench>/` — benchmarks as 224×224 resize-with-pad PNGs + `items.jsonl` (`scripts/prepare_data.py`).
- `scripts/vlm_backbone.py` — `BackboneLM`: runs only SigLIP + PaliGemma expert of the pi0 mixture
  transformer and reads next-token logits off the tied embedding table. Provides `score()` (log-likelihood of
  an answer) and `generate()` (greedy).
- `scripts/eval_backbone.py` — benchmark driver (POPE, MME, ScienceQA-IMG, VQAv2 subset, COCO captions).
- `jobs/eval_backbone.sbatch` — 1 node / 4 GPUs, one checkpoint per GPU.
- `runs/<name>/summary.json` + per-item jsonl; `runs/submitted-jobs.tsv` records every Slurm job we submitted.

## Why likelihood-based metrics
Action training (flow-matching loss only for π0; π0.5 base additionally saw text/FAST-token targets) can
break free-form decoding. Comparing log p(answer | image, prompt) across candidates needs no coherent
generation, so it measures what the backbone still *knows*; generation metrics are reported alongside to
show how much is still *expressible*.

## Commands
```bash
sbatch --export=ALL,CKPTS="pi0_base pi0_libero pi05_base pi05_libero",EXTRA="--limit 16" jobs/eval_backbone.sbatch   # smoke
sbatch --export=ALL,CKPTS="pi0_base pi0_libero pi05_base pi05_libero",EXTRA="--templates paligemma plain" jobs/eval_backbone.sbatch
```
