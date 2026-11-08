# Repository Guidelines

## Project Structure & Module Organization

This project evaluates PaliGemma vision-language capabilities retained in π0/π0.5 checkpoints.

- `scripts/`: backbone extraction, evaluation, data preparation, and summarization.
- `jobs/`: Slurm evaluation and GPU-probe scripts.
- `openpi/`: upstream clone and Python environment (`.venv/`); prefer project changes under `scripts/`.
- `data/<bench>/`: normalized images and `items.jsonl`; `checkpoints/`: weights and tokenizer.
- `runs/`: per-item JSONL, summaries, and submission records; `logs/`: job output; `doc/`: procedures and reports.

## Build, Test, and Development Commands

Run from this directory using `openpi/.venv/bin/python`. Rebuild a broken environment with `bash scripts/setup_env.sh`.

```bash
# Prepare selected benchmarks.
openpi/.venv/bin/python scripts/prepare_data.py --benches pope mme
# Check Python and Slurm syntax.
openpi/.venv/bin/python -m py_compile scripts/*.py
for file in jobs/*.sbatch; do bash -n "$file"; done
# Smoke evaluation with isolated output.
sbatch --export=ALL,CKPTS="pi05_base",EXTRA="--limit 16 --benches pope --out_dir runs/smoke_pi05_base" jobs/eval_backbone.sbatch
# Aggregate completed evaluations.
openpi/.venv/bin/python scripts/summarize.py > runs/summary_table.md
```

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, `PascalCase` classes, and uppercase constants. Keep code comments in English; existing research reports use Korean. Upstream `openpi/` configures Ruff linting/formatting with a 120-character line limit and pre-commit hooks; no root-level formatter is configured.

## Testing Guidelines

Project scripts have no dedicated automated suite or coverage threshold. Run syntax checks, then a `--limit 16` smoke evaluation before full GPU runs. Always supply a separate `--out_dir` for limited runs: they can overwrite existing summaries. Preserve preprocessing and benchmark subsets for comparable results. Upstream tests use pytest and `*_test.py` names; run selected tests from `openpi/` with `.venv/bin/python -m pytest <path>`.

## Commit & Pull Request Guidelines

The root has no Git history. Upstream commits use short descriptive subjects, sometimes scoped prefixes such as `docs(droid):`, and PR numbers. Use concise imperative subjects. PR descriptions should explain changes, affected checkpoints/benchmarks, validation commands, and result differences; link relevant issues and reports.

## Cluster & Configuration Rules

Before executing code in this environment, read and follow [HANGANG_AGENT_GUIDE.md](/scratch/t512a07/minseong/HANGANG_AGENT_GUIDE.md). Apply its procedures to environment setup, data processing, testing, evaluation, training, and Slurm job management. Submit through Slurm on `natStr` with the required job comment; keep compute off login nodes. Record every submission in `runs/submitted-jobs.tsv`. Keep environments and caches local. The account is shared: inspect and cancel only your recorded job IDs.
