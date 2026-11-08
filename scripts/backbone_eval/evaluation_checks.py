"""Dependency-free validation gates shared by extraction and its small regression checks."""


def extraction_failures(report, datasets, splits, require_shuffle=True):
    failures = []
    if report.get("head", {}).get("ok") is False:
        failures.append("head verification failed")
    attn = report.get("attention_direction", {})
    if attn.get("ok") is False or ("expected" in attn and not attn.get("earlier_positions_affected")):
        failures.append("attention direction failed")
    if report.get("params_unchanged") is not True:
        failures.append("parameter fingerprint changed")
    for dataset in datasets:
        data = report.get("datasets", {}).get(dataset, {})
        if data.get("likelihood_status") != "ok":
            failures.append(f"{dataset}: candidate token validation failed")
        if data.get("padding_invariance", {}).get("ok") is not True:
            failures.append(f"{dataset}: padding/batch check failed")
        if data.get("label_independence", {}).get("same_prompt_ids_after_label_change") is not True:
            failures.append(f"{dataset}: label independence failed")
        required = list(splits) + (["test_imgshuf"] if require_shuffle and "test" in splits else [])
        for split in required:
            result = data.get("splits", {}).get(split, {})
            if result.get("status") not in ("ok", "kept_existing") or result.get("n", 0) <= 0:
                failures.append(f"{dataset}/{split}: missing or failed extraction")
    return failures
