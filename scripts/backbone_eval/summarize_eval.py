"""Aggregate validation reports, probe metrics and likelihood results into summary.csv / summary.md / validation_report.md."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA_EVAL, DATASETS, RUN_DIR  # noqa: E402
from adapters import MODEL_SPECS  # noqa: E402

MODELS = ["pi0", "pi05", "paligemma", "paligemma_lb", "openvla", "openvla_oft", "prismatic"]
CKPT = {"pi0": "pi0_base (openpi)", "pi05": "pi05_base (openpi)", "paligemma": "google/paligemma-3b-pt-224 (HF preprocessing)", "paligemma_lb": "google/paligemma-3b-pt-224 (openpi letterbox)",
        "openvla": "openvla/openvla-7b", "openvla_oft": "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "prismatic": "TRI-ML/prismatic-vlms prism-dinosiglip-224px+7b"}


def f3(x):
    return "-" if x is None or x == "" or (isinstance(x, float) and x != x) else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=str(RUN_DIR))
    ap.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=MODELS)
    args = ap.parse_args()
    models = args.models
    run = pathlib.Path(args.run_dir)
    data_manifest = json.loads((DATA_EVAL / "data_manifest.json").read_text())
    models_manifest = json.loads((run / "models_manifest.json").read_text()) if (run / "models_manifest.json").exists() else {}
    validation = {m: json.loads(p.read_text()) for m in models for p in [run / "validation" / f"{m}.json"] if p.exists()}

    rows = []
    for m in models:
        for ds, (dname, ttype) in DATASETS.items():
            row = {"model": m, "checkpoint": CKPT.get(m, MODEL_SPECS[m].get("source", MODEL_SPECS[m]["checkpoint"])), "dataset_variant": dname, "split_type": data_manifest.get(ds, {}).get("split_type", "-"),
                   "attention": models_manifest.get(m, {}).get("attention", "-"), "N_train": "", "N_dev": "", "N_test": "",
                   "ll_acc": "", "probe_acc": "", "chance": "", "majority_acc": "", "ll_bal_acc": "", "probe_bal_acc": "",
                   "ll_macro_f1": "", "probe_macro_f1": "", "ll_recall_yes": "", "probe_recall_yes": "", "ll_fpr_no": "", "probe_fpr_no": "",
                   "imgshuf_ll_acc": "", "imgshuf_probe_acc": "", "probe_C": "", "status": ""}
            if ds in data_manifest.get("_blocked", {}):
                row["status"] = "blocked: " + data_manifest["_blocked"][ds]
            else:
                mp = run / "probes" / m / ds / "metrics.json"
                if not mp.exists():
                    v = validation.get(m, {}).get("datasets", {}).get(ds)
                    row["status"] = "failed/missing: " + (json.dumps(v)[:120] if v else "no extraction") if m in validation else "not run"
                else:
                    r = json.loads(mp.read_text())
                    row.update(N_train=r["n_train"], N_dev=r["n_dev"], N_test=r["n_test"], status=r["status"], probe_C=r.get("selected_C", ""))
                    if r["status"] == "ok":
                        tp, tl, mb = r["test_probe"], r["test_likelihood"], r["test_majority_baseline"]
                        row.update(probe_acc=tp["acc"], ll_acc=tl.get("acc"), majority_acc=mb["acc"],
                                   chance=tp.get("chance", 0.5) if ttype == "mc" else 0.5)
                        if ttype == "binary":
                            row.update(ll_bal_acc=tl.get("balanced_acc"), probe_bal_acc=tp.get("balanced_acc"), ll_macro_f1=tl.get("macro_f1"),
                                       probe_macro_f1=tp.get("macro_f1"), ll_recall_yes=tl.get("recall_yes"), probe_recall_yes=tp.get("recall_yes"),
                                       ll_fpr_no=tl.get("fpr_no"), probe_fpr_no=tp.get("fpr_no"))
                        if "imgshuf" in r:
                            row.update(imgshuf_ll_acc=r["imgshuf"].get("ll_acc"), imgshuf_probe_acc=r["imgshuf"]["probe_acc"])
                        if tl.get("status") == "unsupported":
                            row["status"] = "ok (likelihood unsupported)"
            rows.append(row)
    with (run / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------ markdown
    md = [f"# VLA VLM backbone utility/safety evaluation — results ({run.name})\n",
          "Likelihood = first answer token read with the model's own LM head (argmax over the candidate tokens, no training). "
          "Probe = StandardScaler + L2 logistic regression on the final-normed last-prompt-token vector, trained on the dataset's train "
          "split, C chosen on dev. Both are evaluated on exactly the same held-out test ids. `chance` = mean 1/K; `majority` = train "
          "majority class on test.\n"]
    md.append("## Main table\n")
    md.append("| model | dataset | split | attention | N_train | N_dev | N_test | ll_acc | probe_acc | chance | majority | status |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['model']} | {r['dataset_variant']} | {r['split_type']} | {str(r['attention'])[:40]} | {r['N_train']} | {r['N_dev']} | "
                  f"{r['N_test']} | {f3(r['ll_acc'])} | {f3(r['probe_acc'])} | {f3(r['chance'])} | {f3(r['majority_acc'])} | {r['status']} |")
    md.append("\n## Binary datasets: balanced accuracy / macro-F1 / yes-recall / no-FPR\n")
    md.append("| model | dataset | ll bal-acc | probe bal-acc | ll macro-F1 | probe macro-F1 | ll recall(yes) | probe recall(yes) | ll FPR(no) | probe FPR(no) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["status"].startswith("ok") and DATASETS[[k for k, v in DATASETS.items() if v[0] == r["dataset_variant"]][0]][1] == "binary":
            md.append(f"| {r['model']} | {r['dataset_variant']} | {f3(r['ll_bal_acc'])} | {f3(r['probe_bal_acc'])} | {f3(r['ll_macro_f1'])} | "
                      f"{f3(r['probe_macro_f1'])} | {f3(r['ll_recall_yes'])} | {f3(r['probe_recall_yes'])} | {f3(r['ll_fpr_no'])} | {f3(r['probe_fpr_no'])} |")
    md.append("\n## Image-shuffle control (<=100 test items, images deranged across image groups; diagnostic only)\n")
    md.append("| model | dataset | ll_acc normal→shuffled | probe_acc normal→shuffled |")
    md.append("|---|---|---|---|")
    for m in models:
        for ds, (dname, _) in DATASETS.items():
            mp = run / "probes" / m / ds / "metrics.json"
            if mp.exists():
                r = json.loads(mp.read_text())
                if "imgshuf" in r:
                    s = r["imgshuf"]
                    md.append(f"| {m} | {dname} | {f3(s.get('ll_acc_same_items_normal'))} → {f3(s.get('ll_acc'))} | "
                              f"{f3(s['probe_acc_same_items_normal'])} → {f3(s['probe_acc'])} |")
    # per-category (MME) and POPE subsets from predictions
    md.append("\n## Data\n")
    for ds, info in data_manifest.items():
        if ds.startswith("_"):
            continue
        md.append(f"- **{info['name']}** ({info['split_type']}): n={info['n']}, images={info['n_images']}, label dist={info['label_dist']}, "
                  f"K dist={info['n_choices_dist']}, exclusions={info['exclusions']}, dropped by cap={info['dropped_by_cap']}")
        for n in info["notes"]:
            md.append(f"    - {n}")
    for ds, why in data_manifest.get("_blocked", {}).items():
        md.append(f"- **{DATASETS[ds][0]}**: blocked — {why}")
    md.append("\n## Interpretation guardrails\n")
    md += ["- Likelihood = how well the existing language head reads a one-token answer; probe = how well a supervised linear readout "
           "recovers the label from the last representation. Neither is the VLA's zero-shot safety judgement or behaviour.",
           "- Low likelihood + high probe: a linearly readable signal exists outside the existing readout. Not proof that semantic alignment "
           "damage is the only cause. Both low: this input / last position / linear readout fails; says nothing about middle layers.",
           "- SIUO-MCQA = choosing the safest response (exploratory, n_test≈33). POPE/MME splits are research_split (probe trained on part "
           "of the official test pool) — not official zero-shot benchmark numbers; MME is not the official total score.",
           "- VQAv2-MC-adapted and COCO-CaptionChoice-adapted are single-token multiple-choice re-formulations, not VQA accuracy / CIDEr.",
           "- Pi0/Pi0.5 (openpi letterbox preprocessing) vs OpenVLA/Prismatic/OFT (resize-naive, Llama-2) differ in backbone, "
           "preprocessing, tokenizer and training stage; cross-family gaps are not a causal effect of action training alone.",
           "- paligemma = HF resize-to-224 preprocessing; paligemma_lb = the same weights with the Pi models' letterbox preprocessing (matched-input comparison)."]
    (run / "summary.md").write_text("\n".join(md) + "\n")

    # ------------------------------------------------------------------ validation report
    vr = [f"# Validation report ({run.name})\n", "Low accuracy is not an implementation error; checks below verify the pipeline, not the scores.\n"]
    for m, v in validation.items():
        d = v.get("describe", {})
        vr.append(f"## {m}\n")
        vr.append(f"- checkpoint: `{d.get('checkpoint')}` · attention: {d.get('attention')} · head: {d.get('head')}")
        vr.append(f"- dtype: {d.get('dtype')} · tokenizer: {d.get('tokenizer')} · preprocessing: {d.get('preprocessing')}")
        if "physical_batch_limit" in d:
            vr.append(f"- execution: physical_batch_limit={d['physical_batch_limit']} · {d.get('batch_execution')}")
        vr.append(f"- prompt: `{d.get('prompt_format')}` · candidates: {d.get('candidate_tokenization')}")
        if "modeling_llama_sha1" in d:
            vr.append(f"- modeling_llama.py: `{d['modeling_llama_file']}` sha1={d['modeling_llama_sha1']} · attn_impl={d.get('attn_implementation')} "
                      f"· transformers {d.get('transformers_version')} · num_images_in_input={d.get('num_images_in_input')}")
        if d.get("missing_keys") not in (None, "n/a (HF strict load)"):
            vr.append(f"- loader: {d.get('loader')} · missing={d.get('missing_keys')} unexpected={d.get('unexpected_keys')}")
        vr.append(f"- head check: {json.dumps(v.get('head'), default=str)}")
        vr.append(f"- attention direction: {json.dumps(v.get('attention_direction'), default=str)}")
        vr.append(f"- parameters unchanged over the run (fingerprint): {v.get('params_unchanged')} · status={v.get('status')} · {v.get('seconds')}s")
        for ds, dv in v.get("datasets", {}).items():
            if "candidates" not in dv:
                vr.append(f"- {ds}: {dv}")
                continue
            cands = {k: (c["token_id"], c["pieces"], c["ok"]) for k, c in dv["candidates"].items()}
            vr.append(f"- **{ds}**: likelihood={dv['likelihood_status']} · candidates={cands}")
            vr.append(f"    - prompt tokens {dv['prompt_tokens']} · excluded too long={dv['n_excluded_too_long']} · "
                      f"padding/batch invariance={dv.get('padding_invariance', {}).get('ok')} "
                      f"(h rel diff {f3(dv.get('padding_invariance', {}).get('h_max_abs_diff_rel'))}, cand logp diff "
                      f"{f3(dv.get('padding_invariance', {}).get('cand_logp_max_abs_diff'))}) · label independence={dv.get('label_independence')}")
            if "execution_scope" in dv.get("padding_invariance", {}):
                vr.append(f"    - batch check scope: {dv['padding_invariance']['execution_scope']}")
            vr.append(f"    - splits: {json.dumps(dv['splits'], default=str)}")
    vr.append("\n## Data checks\n")
    for ds, info in data_manifest.items():
        if not ds.startswith("_"):
            vr.append(f"- {info['name']}: image groups disjoint across splits (asserted at build); n={info['n']}; label dist={info['label_dist']}; "
                      f"exclusions={info['exclusions']}; 10 rendered examples in data_eval/{ds}/examples.md")
    vr.append("\n## Freeze\n- Probes are trained with scikit-learn on cached .npz features; no model parameter, optimizer or gradient is involved. "
              "Scaler and C are chosen on train/dev only. Parameter fingerprints before/after extraction are listed per model above.")
    (run / "validation_report.md").write_text("\n".join(vr) + "\n")
    print(f"wrote {run / 'summary.csv'}, summary.md, validation_report.md")


if __name__ == "__main__":
    main()
