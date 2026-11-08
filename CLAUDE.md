# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Measures how much PaliGemma-3B vision-language ability survives inside the openpi VLA checkpoints
`pi0_base`, `pi0_libero`, `pi05_base`, `pi05_libero`. The top-level directory is **not** a git repo;
`openpi/` is an unmodified upstream clone (Physical-Intelligence/openpi) that also hosts the project venv.
Findings so far are written up (in Korean) in `doc/VLM_backbone_eval_report.md`; `runs/summary_table.md`
is the current results table. The owner writes docs in Korean, code comments in English.

## Cluster rules (KISTI Hangang, aarch64 GH200)

Full guide: `../HANGANG_AGENT_GUIDE.md`. Non-negotiables that affect every task here:

- GPU/compute work goes through Slurm on partition **`natStr`** only; never switch partitions. Every
  `#SBATCH` needs `--comment="field=nlp;appl=pytorch"`. Login node (`glogin*`) is for editing, syntax
  checks, light CPU scripts (`summarize.py`), and `sbatch`.
- Submit from this directory (`sbatch` scripts `cd "$SLURM_SUBMIT_DIR"` and refuse to run outside
  `/scratch/$USER/`). Append every submission to `runs/submitted-jobs.tsv`
  (`date -Is <TAB> jobid <TAB> path <TAB> label`).
- The Linux account is shared. Never `scancel -u`, `pkill -u`, or touch jobs you did not record. Cancel
  only by exact Job ID after checking `scontrol show job`.
- No global changes: no `~/.bashrc`, `conda init`, or global `pip install`. All caches live under
  `.cache/` (set by `scripts/hangang-env.sh`). Nothing GPU-related is installed system-wide.
- Progression: `--limit 16` smoke run → full run. Don't resubmit an unchanged failing job.
- `/scratch` is purged after 15 days without access and is not backed up.

## Commands

Project Python (JAX 0.5.3 CUDA, Python 3.11, works on login node for CPU-only imports):

```bash
PY=openpi/.venv/bin/python
```

```bash
bash scripts/setup_env.sh                                      # rebuild openpi/.venv with tools/uv (only if broken)
bash scripts/download_ckpt.sh pi0_base pi05_libero             # HTTPS from public GCS bucket, resumable
$PY scripts/prepare_data.py [--benches pope mme] [--limit N]   # HF -> data/<bench>/{items.jsonl,images/}
$PY -m py_compile scripts/*.py && bash -n jobs/*.sbatch        # pre-submit syntax check (no tests exist)

# smoke (use --out_dir so partial results don't clobber runs/<name>/summary.json; see Gotchas)
sbatch --export=ALL,CKPTS="pi05_base",EXTRA="--limit 16 --benches pope --out_dir runs/smoke_pi05_base" jobs/eval_backbone.sbatch
# full run, all four checkpoints, one per GPU
sbatch --export=ALL,CKPTS="pi0_base pi0_libero pi05_base pi05_libero",EXTRA="--templates paligemma plain --batch 32" jobs/eval_backbone.sbatch
sbatch jobs/gpu_probe.sbatch                                   # 10-min JAX-sees-4-GPUs check

squeue -u "$USER" -o '%.18i %.24j %.2t %.10M %R'               # then read ONLY logs for your job id
tail -n 40 logs/eval-<name>-<jobid>.log                        # per-checkpoint stdout/stderr
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed
$PY scripts/summarize.py [name ...] > runs/summary_table.md    # aggregate runs/*/summary.json
```

`eval_backbone.py` flags: `--name`, `--ckpt_dir`, `--pi05` (required for pi05_* checkpoints; the sbatch
sets it from the name prefix), `--benches`, `--templates {paligemma,plain,paligemma_strip}`, `--limit`,
`--batch`, `--no_gen`, `--out_dir`. A full 4-checkpoint run with two templates took ~40 min.

## Architecture

**`scripts/vlm_backbone.py` – `BackboneLM`.** Loads a pi0/pi0.5 checkpoint through openpi's own
`Pi0Config.load`, then runs only SigLIP (`PaliGemma.img`) and the *first* expert of the mixture
transformer (`PaliGemma.llm([x, None], ...)`); the action expert is loaded but never executed. Gemma has
no LM head, so next-token logits are `hidden @ embedder.input_embedding.T` (tied embeddings), computed in
fp32 inside a single jitted forward with fixed shape `[b, 192]`. Sequence layout mirrors PaliGemma
prefix-LM / openpi `embed_prefix`: 256 image tokens + `BOS prompt "\n"` (bidirectional) then answer tokens
(causal). Position i predicts token i+1, so the `"\n"` slot predicts the first answer token.
`score()` returns summed log p(answer) and token count; `generate()` is greedy and re-runs the full forward
every step (simple, slow, fixed shape). Prompt truncation keeps BOS and trailing `"\n"` and never truncates
the answer.

**`scripts/eval_backbone.py` – driver.** `BENCH_TASK` maps bench → runner (`yesno`: POPE, MME;
`mc`: ScienceQA; `vqa`: VQAv2; `caption`: COCO). Every runner emits two metric families because action
training can destroy free-form decoding: *likelihood* (compare log p over candidates, or NLL/token of the
reference; needs no coherent generation) and *generation* (greedy decode + standard metric). Yes/no is
scored both with and without EOS appended. `TEMPLATES` holds prompt formats; `paligemma_strip` removes
MME's trailing "Please answer yes or no." Output per run: `runs/<name>/summary.json` keyed
`"<bench>/<template>"` plus `runs/<name>/<bench>.<template>.jsonl` with per-item log-probs and
generations.

**`scripts/prepare_data.py`.** Streams `lmms-lab/*` datasets from HF and writes 224×224 letterboxed PNGs
matching openpi `resize_with_pad`, so images get exactly the VLA's training preprocessing. Default
subsets: COCO val2017 first 2000, VQAv2 val first 3000, others full. Changing resize or subsets
invalidates comparisons with existing `runs/`.

**`jobs/eval_backbone.sbatch`.** One node, 4 GPUs, `srun --ntasks=N --gpus-per-task=1`; each task picks
`CKPTS[$SLURM_PROCID]` and writes `logs/eval-<name>-<jobid>.log`. Deliberately no `--kill-on-bad-exit`
so one checkpoint failing doesn't cancel the others. `CKPTS` and `EXTRA` come via `--export`.

## Gotchas

- `summary.json` merging: without `--limit`, a run loads the existing `runs/<name>/summary.json` and merges
  new `bench/template` keys, so partial re-runs (one bench, one template) are safe. **With `--limit`, it
  starts from empty results and still writes `summary.json`**, overwriting full results. Always pass
  `--out_dir` for smoke runs.
- `--pi05` selects `Pi0Config(pi05=True)` and must match the checkpoint family. When invoking
  `eval_backbone.py` by hand, pass it for `pi05_*`; the sbatch derives it from the name prefix.
- `checkpoints/paligemma_pt_224/` is empty: the original PaliGemma baseline could not be downloaded (GCS
  403, HF gated). Any "how much was lost" claim lacks this reference until it is obtained.
- `checkpoints/tokenizer/paligemma_tokenizer.model` is the sentencepiece model; `PAD_ID=0`, `EOS_ID=1`.
- `scripts/` imports `openpi.*` from the venv's editable install; `openpi/` is upstream at the pinned
  commit with no local patches. Prefer adding code under `scripts/` over editing `openpi/`.
- `logs/*.log.seen` are earlier mid-run snapshots of the matching `.log`, kept while watching jobs; the
  `.log` is authoritative.

## Utility/safety single-token + probe experiment (added 2026-11-05)

`scripts/backbone_eval/` implements `../VLA_VLM_Utility_Safety_Experiment_Plan.md`: `prepare_eval_data.py` (manifests in
`data_eval/<ds>/{items,images}.parquet`), `adapters/` (`openpi_prefix.py` for pi0/pi05 in the openpi venv; `prismatic_hf.py` for
openvla / prismatic / openvla_oft inside the site PyTorch container with `envs/ctpkgs` on PYTHONPATH), `extract_and_score.py`
(validation + features + one-token likelihood), `train_linear_probe.py`, `summarize_eval.py`, `diagnostics.py`. Job:
`jobs/backbone_eval.sbatch` (one lane per GPU, smoke gate first; `ONLY=... EXTRA=--overwrite` re-runs). Results and write-ups:
`runs/backbone_eval_v1/`, `doc/VLA_VLM_Utility_Safety_Results.md`, `doc/VLA_VLM_Utility_Safety_Experiment_Log.md`.

Hard constraint: the group inode quota on /scratch is nearly full (`lfs quota -g bs0229 /scratch`). Never create new venvs or
many small files; use the container, `--target` package dirs, parquet/npz/tar. Large downloads go through `ssh dm01`.
OFT needs `envs/oft_patch/modeling_llama.py` injected (bidirectional attention). PaliGemma (HF, needs `envs/ctpkgs_pg` = transformers 4.44.2
ahead of `envs/ctpkgs`) and VLGuard were added once an HF token existed (`~/.cache/huggingface/token`, read by `scripts/dm_fetch.sh`);
LlavaGuard still needs the dataset's access form approved (403 otherwise). Never write the token into code, logs or docs.
