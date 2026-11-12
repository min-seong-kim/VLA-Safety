# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Research code on safety-relevant knowledge inside VLA models (openpi `pi0`/`pi05`, OpenVLA, OpenVLA-OFT).
Two bodies of work share the repo:

1. **Semantic safety rollouts (active, `scripts/semantic_safety/`)** – does a VLA represent that an
   instruction is unsafe, pass that to the action expert (AE), and use it? LIBERO-based safe/unsafe twin
   scenes, per-timestep feature recording for four checkpoints, and an AE-only LoRA pilot ("experiment B").
2. **VLM backbone evaluation (earlier, `scripts/*.py`, `scripts/backbone_eval/`)** – how much PaliGemma
   VL ability survives in pi0/pi0.5 checkpoints. Ran on the KISTI Hangang cluster; see "Legacy" below.

Read `doc/Semantic_Safety_Research_Handoff.md` first: it is the authoritative state of work (what was run,
what was *not*, interpretation limits, proposed next steps). `doc/Semantic_Safety_B_Transfer_Diagnosis.md`
continues it: why the experiment-B adapter does not transfer to the safety scenes (it is tied to the
narrow training layout). Docs are written in Korean, code comments in
English.

## Environment (this machine, not Hangang)

- Root is `/NHNHOME/26msit001_B/BASE/edge_ai_lab/minseong/VLA-Safety`: x86_64, one local NVIDIA B200,
  **no Slurm, no `/scratch`**. GPU work runs as local processes. `AGENTS.md`, `README.md` and every
  `jobs/*.sbatch` still describe the old aarch64 Hangang/Slurm setup (they hard-check a `/scratch/...`
  root and will not run here); `../HANGANG_AGENT_GUIDE.md` does not exist on this machine.
- The server is shared. Only stop processes whose PID this project recorded (`runs/*/local_jobs.jsonl`,
  `runs/semantic_rollouts_v3/launch.json`), and do not reuse PIDs recorded by an earlier session.
- The filesystem is ~97% full and `runs/` is already ~340 GB. Check `df -h .` before anything that writes
  trajectories/features; do not create new venvs casually.
- No global installs or shell-profile edits; caches go under `.cache/` (scripts set `HF_HOME`,
  `TORCH_HOME`, `OPENPI_DATA_HOME` themselves). Never write the HF token into code, logs or docs.

- **Headless rendering needs the user-level NVIDIA EGL vendor.** The system only ships Mesa, so any
  simulator process fails with `Cannot initialize a EGL device display` unless it is started with
  `__EGL_VENDOR_LIBRARY_FILENAMES=$E/10_nvidia.json LD_LIBRARY_PATH=$E` where
  `E=/NHNHOME/26msit001_B/BASE/edge_ai_lab/minseong/.persist/nvidia-egl`. (An `EGLError` printed from
  `GLContext.__del__` at exit is harmless.)
- Long runs: start them detached (`setsid nohup ... &`) and record the PID; see
  `runs/motor_b_transfer_exec_ablation_v1/launch.sh` for a working server + evaluator launch.

Each role has its own interpreter — using the wrong one is the most common failure:

| Role | Interpreter | Notes |
|---|---|---|
| pi0/pi0.5 inference, LoRA training | `openpi/.venv/bin/python` | py3.11, torch 2.7.1+cu128; includes openpi's patched Transformers files |
| Simulator, teacher, evaluation client, audits, probe export, unit tests | `envs/libero-render/bin/python` | py3.12, MuJoCo 3.2.2 (EGL) |
| OpenVLA policy worker | `envs/openvla-policy/bin/python` | |
| OpenVLA-OFT policy worker | `envs/openvla-oft-policy/bin/python` | |
| Public-dataset inspection / HTML screenshots | `envs/safety-inspection/`, `envs/html-preview/` | |

## Git

Remote `origin` = github.com/min-seong-kim/VLA-Safety. `.gitignore` excludes `data/`, `runs/`,
`checkpoints/`, `envs/`, `openpi/`, logs and weights — but **not** `third_party/` (upstream clones of
libero, openvla, openvla_oft, plus `robosuite_vanilla`) or the root paper PDFs, so never `git add -A`;
stage explicit paths. Most of `scripts/semantic_safety/` and the three `doc/Semantic_Safety_*.md` are
currently untracked. `openpi/` (commit `215abfb`) and the `third_party/` clones have no local patches;
add code under `scripts/` instead of editing them.

## Commands

```bash
SIM=envs/libero-render/bin/python
PI=openpi/.venv/bin/python
S=scripts/semantic_safety

# Unit tests (CPU; must run from inside the directory with the SIM interpreter — modules import each other flat)
(cd $S && ../../$SIM -m unittest test_semantic_safety)
(cd $S && ../../$SIM -m unittest test_semantic_safety.SemanticSafetyChecks.test_lda_dual_matches_primal_regularized_covariance)
$SIM -m py_compile $S/*.py                         # syntax check; there is no linter config at the root

# Recorded rollouts of an official checkpoint (spawns the matching policy worker itself)
$SIM $S/run_recorded_rollouts.py --model pi05_libero --suite native --limit 16 --out-dir runs/<new_dir>
$SIM $S/run_recorded_rollouts.py --model pi05_libero --suite safety --horizon 160 --out-dir runs/<new_dir>
#   models: pi0_libero | pi05_libero | openvla_libero_spatial | openvla_oft_libero_spatial

# Export a probe matrix from recorded features (no probe is trained by this)
$SIM $S/export_probe_matrix.py --model pi05_libero --feature vlm_hidden --region last_prompt \
     --layer -1 --timesteps initial --fold 0 --out runs/<name>.npz

# Experiment B pieces (pi0.5 only): policy server in one process, simulator client in another
$PI  $S/train_motor_lora.py --manifest data/motor_skill_b_v1/demonstrations/manifest.jsonl --out runs/<new_dir>/training
$PI  $S/motor_policy_server.py --socket /tmp/<name>.sock &
$SIM $S/evaluate_motor_policy.py --socket /tmp/<name>.sock --design data/motor_skill_b_v1/demonstrations/frozen_design.json \
     --adapter <original|path/to/adapter_step_XXXXX.pt> --kind <motor|native|transfer|training_replay> \
     --split <validation|test> --out runs/<new_dir>/<eval_name>
#   --horizon N / --chunk-steps N override the per-kind defaults (motor 360/10, native 220/10, transfer 160/1);
#   the same keys ("horizon", "chunk_steps") work per entry in a --commands JSON list.
#   --commands entries also accept "instruction_style": "motor_template" (transfer) and
#   "layout_style": "train_distribution" (transfer) | "safety_layout" (motor); see doc/Semantic_Safety_B_Transfer_Diagnosis.md.
```

`run_motor_b_workflow.py` and `supervise_semantic_rollouts.py` are the orchestrators that produced
`runs/motor_lora_b_v1/` and `runs/semantic_rollouts_v3/`; their output paths are hard-coded to those
directories. Treat them as a record of how the runs were made, not something to re-run.

## Architecture (semantic safety)

**Two-process rollouts.** The simulator and the policy never share an interpreter. A driver in the SIM env
(`run_recorded_rollouts.py`, `evaluate_motor_policy.py`) steps LIBERO and talks over an authenticated
`multiprocessing.connection` Unix socket to a policy process in the model's own env
(`rollout_policy_worker.py` for all four checkpoints with feature hooks; `motor_policy_server.py` for
batched pi0.5 ± LoRA without feature extraction). `rollout_common.py` is the shared bootstrap: it puts
`third_party/libero` + `third_party/robosuite_vanilla` on `sys.path`, writes
`envs/libero-rollout-config/config.yaml`, forces EGL, and defines the 8-D state vector and observation
payload. Scripts import each other as flat modules (`from rollout_common import ...`), so they are run by
path, not with `-m`.

**Data lineage** (each stage has a different meaning; do not merge counts across them):

| Stage | Location | What it is |
|---|---|---|
| Public sources | `data/safety_sources/` | Downloaded safety datasets (some only sampled); reference / external eval |
| v1 | `data/libero_semantic_explicit_v1/` | 6 situations rendered from LIBERO-Safety scenes; pipeline check only |
| v2 | `data/semantic_safety_catalog_v2/` | **Specification only** (36 families, 108 pair candidates, 0 new images), from `synthesis_v2_scenarios.py` |
| v3 | `data/semantic_rollout_v3/` | 36 executable scenes × safe/unsafe instruction on identical initial state (`rollout_safety_env.py`) |
| v3 rollouts | `runs/semantic_rollouts_v3/<model>/episodes/<family>__<safe\|unsafe>__seed0/` | `trajectory.h5`, `features.h5`, `events.jsonl`, videos |
| B demos | `data/motor_skill_b_v1/demonstrations/` | 610 neutral pick-and-place demos from a privileged-state scripted FSM teacher (`collect_motor_demonstrations.py`, env in `motor_skill_env.py`) |
| B results | `runs/motor_lora_b_v1/` | `results.json`, `per_asset.csv`, `report.html`, per-episode recordings |

**Recording alignment.** `observation[t] → model input/features[t] → action[t] → observation[t+1], events[t]`.
Pooled features in `features.h5` have axes `[control_timestep, forward_call, layer, batch_or_camera, region,
hidden_dim]`; `forward_call` is *not* time (10 denoising calls for the pi AE, per-camera calls for vision).
Token spans are in each step's `step_metadata`; full token tensors exist only for t=0. Raw RGB is stored in
simulator orientation and flipped on both axes in model preprocessing. Schema details:
`doc/Semantic_Safety_Rollouts_v3.md`.

**Experiment B** (`motor_lora.py`, `train_motor_lora.py`). Base `checkpoints/pi05_libero_pytorch`; every
original parameter frozen; LoRA (rank 16) only on the 18 action-expert layers' q/k/v/o and MLP projections.
VLM prefix is computed under `no_grad` and its KV cache reused for the flow-matching loss.
`audit_motor_lora.py` / `training/freeze_audit.json` verify frozen-weight hashes and bitwise-identical VLM
hidden states on a fixed input.

## Gotchas

- **Skip-if-exists:** `run_recorded_rollouts.py` and `evaluate_motor_policy.py` skip any episode whose
  `summary.json` already exists. Always give new experiments a fresh output directory; reusing one silently
  reports old results. Never overwrite `runs/semantic_rollouts_v3/` or `runs/motor_lora_b_v1/`.
- **Settings that differ between result sets** (never mix their numbers): v3 rollouts replan every step
  (`replan_steps=1`, horizon 160, pad-only grasp / 2 cm lift); B motor/native evals execute 10-action
  chunks (horizon 360 / 220, full-finger grasp / 4 cm lift). Neither reproduces official benchmark settings.
- `known_allergen_serving` is physically unstable (object drifts with no policy action). Use the
  `*_passive_stability_filtered` probe matrices/index (35 families, 70 samples per model at t=0).
- Probe/analysis splits are by family/mechanism (`data/semantic_rollout_v3/probe_splits.json`): both labels,
  all models, all timesteps of a family stay in one partition. Timesteps are not independent samples.
- `null` in success/harm fields means "physics does not support judging this", not failure or refusal;
  `harm_observed` is `null` everywhere. Risk metrics are geometric proxies only.
- `pi05_libero` has `discrete_state_input=False`: state is recorded but is not a model input.
- OpenVLA has no wrist camera; its wrist-region features are zero placeholders.
- The B test split is new layouts of *trained* objects, not held-out objects, and has already been looked
  at — do not tune against it.
- Not done yet (do not report as done): representation linear probes, LDA/t-SNE on VLA features,
  special-token interventions, experiment C, reproducibility re-run of B.

## Legacy: VLM backbone evaluation (Hangang)

Code is present but its data, checkpoints (`pi0_base`, `pi05_base`, tokenizer), `runs/backbone_eval_*`,
`tools/uv` and the Slurm cluster are not on this machine; results live in `doc/VLM_backbone_eval_report.md`
and `doc/VLA_VLM_*.md`.

- `scripts/vlm_backbone.py` (`BackboneLM`): loads a pi0/pi0.5 JAX checkpoint via `Pi0Config.load`, runs only
  SigLIP + the first (PaliGemma) expert, and gets next-token logits as `hidden @ input_embedding.T` (tied
  embeddings, no LM head) in one jitted fixed-shape `[b, 192]` forward. Layout: 256 image tokens +
  `BOS prompt "\n"` (bidirectional) then answer tokens (causal). `--pi05` must match the checkpoint family.
- `scripts/eval_backbone.py`: POPE/MME/ScienceQA/VQAv2/COCO with paired *likelihood* and *generation*
  metrics. With `--limit` it starts from empty results and still overwrites `runs/<name>/summary.json`, so
  smoke runs need `--out_dir`.
- `scripts/backbone_eval/`: single-token likelihood + linear-probe study across pi0/pi05 (`adapters/openpi_prefix.py`)
  and OpenVLA/Prismatic/OFT (`adapters/prismatic_hf.py`, run inside the site PyTorch container on Hangang);
  plan in `VLA_VLM_Utility_Safety_Experiment_Plan.md`.
- If this is ever run on Hangang again: partition `natStr` only, `--comment="field=nlp;appl=pytorch"`,
  submit from the project dir, log every submission in `runs/submitted-jobs.tsv`, cancel only by exact job ID.
