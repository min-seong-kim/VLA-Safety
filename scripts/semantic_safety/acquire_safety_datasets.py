"""Catalog and acquire public safety research data without executing remote code.

The catalog pins HF revisions and GCS generations. Downloads are atomic, bounded
by an explicit plan, and SHA-256 recorded. Original archives are kept intact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data/safety_sources"
HF_REPOS = {
    "embhazard": "EMBGuard/EMBHazard",
    "embguardtest_v3": "EMBGuard/EMBGuardTest_v3",
    "embguardtest_legacy": "EMBGuard/EMBGuardTest",
    "embguard_heldout": "EMBGuard/heldout_set",
    "roboshackles": "YZW00/RoboShackles",
    "libero_safety": "LIBERO-Safety/libero_safety",
    "libero_safety_assets": "LIBERO-Safety/libero_safety_assets",
    "hazardarena": "HazardArena-anon/HazardArena-anonymous",
    "hazardarena_assets": "HazardArena-anon/hazardarena-assets",
    "hazardarena_pi0_real": "hazardarena-anonymous/HazardArena_pi0_real",
}
CODE_REPOS = {
    "hazardarena_code": "HazardArena-Team/HazardArena",
    "libero_safety_code": "LIBERO-SAFETY/LIBERO-Safety",
    "embguard_code": "dongwxxkchoi/EMBGuard",
    "asimov_code": "asimov-benchmark/code",
}
HEADERS = {"User-Agent": "VLA-Safety-dataset-inspection/1.0"}


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def get_json(url, **kwargs):
    response = requests.get(url, headers=HEADERS, timeout=90, **kwargs)
    response.raise_for_status()
    return response.json()


def catalog_hf(key, repo, out):
    info = get_json(f"https://huggingface.co/api/datasets/{repo}", params={"blobs": "true"})
    save_json(out / "catalog" / f"{key}.json", info)
    files = []
    for item in info.get("siblings", []):
        name = item["rfilename"]
        files.append({
            "path": name,
            "size": item.get("size", item.get("lfs", {}).get("size")),
            "sha256": item.get("lfs", {}).get("sha256"),
            "url": f"https://huggingface.co/datasets/{repo}/resolve/{info['sha']}/{quote(name)}?download=true",
        })
    return {"key": key, "repo": repo, "revision": info["sha"], "files": files,
            "gated": info.get("gated"), "card": info.get("cardData", {}), "kind": "huggingface"}


def catalog_code(key, repo, out):
    info = get_json(f"https://api.github.com/repos/{repo}")
    commit = get_json(f"https://api.github.com/repos/{repo}/commits/{info['default_branch']}")["sha"]
    save_json(out / "catalog" / f"{key}.json", {"repository": info, "commit": commit})
    return {"key": key, "repo": repo, "revision": commit, "kind": "github",
            "files": [{"path": "source.tar.gz", "size": None,
                       "url": f"https://codeload.github.com/{repo}/tar.gz/{commit}"}]}


def catalog_gcs(out):
    objects = []
    params = {"prefix": "robotics/asimov", "maxResults": 1000}
    while True:
        page = get_json("https://storage.googleapis.com/storage/v1/b/gresearch/o", params=params)
        objects.extend(page.get("items", []))
        if not page.get("nextPageToken"):
            break
        params["pageToken"] = page["nextPageToken"]
    save_json(out / "catalog/asimov_gcs.json", objects)
    datasets = {}
    for item in objects:
        parts = item["name"].split("/")
        if len(parts) < 4 or not item.get("size") or int(item["size"]) == 0:
            continue
        key = parts[1]
        group = datasets.setdefault(key, {"key": key, "kind": "gcs", "files": []})
        group["files"].append({"path": "/".join(parts[2:]), "size": int(item["size"]),
                               "generation": item["generation"], "md5": item.get("md5Hash"),
                               "url": "https://storage.googleapis.com/gresearch/" + quote(item["name"])
                               + "?generation=" + item["generation"]})
    return list(datasets.values())


def catalog(out):
    records, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(catalog_hf, key, repo, out): key for key, repo in HF_REPOS.items()}
        tasks.update({pool.submit(catalog_code, key, repo, out): key for key, repo in CODE_REPOS.items()})
        tasks[pool.submit(catalog_gcs, out)] = "asimov_gcs"
        for future in as_completed(tasks):
            key = tasks[future]
            try:
                value = future.result()
                values = value if isinstance(value, list) else [value]
                records.extend(values)
                for entry in values:
                    sizes = [f["size"] for f in entry["files"] if f["size"] is not None]
                    print(f"CATALOG {entry['key']} files={len(entry['files'])} bytes={sum(sizes)}", flush=True)
            except Exception as error:
                errors.append({"key": key, "error": str(error)})
                print(f"ERROR {key}: {error}", flush=True)
    url = "https://purseclab.github.io/benchmark-for-robotics-security/croissant.json"
    try:
        value = get_json(url)
        save_json(out / "catalog/robojailbench_croissant.json", value)
        files = []
        for item in value.get("distribution", []):
            if str(item.get("contentUrl", "")).endswith(".zip"):
                files.append({"path": item["contentUrl"].rsplit("/", 1)[-1], "size": None,
                              "url": item["contentUrl"], "sha256": item.get("sha256")})
        records.append({"key": "robojailbench", "kind": "archives", "source": url, "files": files})
        print(f"CATALOG robojailbench files={len(files)}", flush=True)
    except Exception as error:
        errors.append({"key": "robojailbench", "error": str(error)})
    save_json(out / "catalog/index.json", {"date": now(), "datasets": sorted(records, key=lambda x: x["key"]),
                                           "errors": errors})


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download(item, out, cap):
    destination = (out / "raw" / item["dataset"] / item["path"]).resolve()
    if not destination.is_relative_to((out / "raw").resolve()):
        raise ValueError("Unsafe destination path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get("size")
    if destination.exists():
        if expected is not None and destination.stat().st_size != expected:
            raise ValueError(f"Existing file size mismatch: {destination}")
        sha = digest(destination)
        if item.get("sha256") and sha != item["sha256"]:
            raise ValueError(f"Existing file hash mismatch: {destination}")
        return {**item, "local_path": str(destination.relative_to(ROOT)),
                "bytes": destination.stat().st_size, "sha256": sha, "status": "verified_existing"}
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        try:
            hasher, count = hashlib.sha256(), 0
            with requests.get(item["url"], headers=HEADERS, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(4 * 1024 * 1024):
                        count += len(block)
                        if count > cap:
                            raise ValueError(f"Single file exceeds cap: {item['path']}")
                        stream.write(block)
                        hasher.update(block)
            if expected is not None and count != expected:
                raise ValueError(f"Size mismatch {count} != {expected}")
            sha = hasher.hexdigest()
            if item.get("sha256") and sha != item["sha256"]:
                raise ValueError("SHA256 mismatch")
            temporary.replace(destination)
            return {**item, "local_path": str(destination.relative_to(ROOT)), "bytes": count,
                    "sha256": sha, "status": "downloaded"}
        except (requests.RequestException, OSError) as error:
            if attempt == 2:
                raise error
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def fetch(out, plan_path, cap, workers):
    plan = json.loads(plan_path.read_text())
    records, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(download, item, out, cap): item for item in plan["files"]}
        for future in as_completed(tasks):
            item = tasks[future]
            try:
                record = future.result()
                records.append(record)
                print(f"OK {item['dataset']}/{item['path']} {record['bytes']}", flush=True)
            except Exception as error:
                errors.append({**item, "error": str(error)})
                print(f"ERROR {item['dataset']}/{item['path']}: {error}", flush=True)
            save_json(out / f"{plan_path.stem}_results.json", {"date": now(), "files": records, "errors": errors})
    print(f"DONE files={len(records)} bytes={sum(r['bytes'] for r in records)} errors={len(errors)}", flush=True)
    if errors:
        raise SystemExit(1)


def make_plan(out):
    """Download complete compact sets and explicit samples of large training sets."""
    index = json.loads((out / "catalog/index.json").read_text())
    selected, scopes = [], {}
    full = {"embguardtest_v3", "embguardtest_legacy", "robojailbench", "roboshackles", "hazardarena"}
    for dataset in index["datasets"]:
        key = dataset["key"]
        files = dataset["files"]
        if key in full or dataset["kind"] == "github" or key.startswith("asimov_multimodal_") or key.startswith("asimov_v2_"):
            keep = files
            scope = "complete public release at recorded revision"
        elif key == "embhazard":
            shards = [f for f in files if f["path"].endswith(".parquet")]
            chosen = {shards[i]["path"] for i in (0, 15, 30, 45)}
            keep = [f for f in files if f["path"] in chosen or not f["path"].endswith(".parquet")]
            scope = "four distributed original shards (0, 15, 30, 45); not the full training set"
        elif key == "embguard_heldout":
            keep = [f for f in files if not f["path"].startswith(("data/safe-", "data/unsafe-"))]
            scope = "complete four scenario splits; duplicate legacy safe/unsafe splits excluded"
        elif key == "libero_safety":
            keep = [f for f in files if f["path"].startswith("meta/") or f["path"] == ".gitattributes"]
            scope = "complete metadata; episode samples selected separately by semantic task"
        elif key in {"hazardarena_assets", "libero_safety_assets", "hazardarena_pi0_real"}:
            keep = [f for f in files if f["path"] in {"README.md", ".gitattributes"}]
            scope = "catalog and available card only; bulky auxiliary release not downloaded"
        else:
            continue
        scopes[key] = {"selection": scope, "selected_files": len(keep), "available_files": len(files),
                       "selected_known_bytes": sum(f.get("size") or 0 for f in keep)}
        selected.extend({**item, "dataset": key} for item in keep)
    # Fetch small metadata and code first, then compact datasets and larger shards.
    selected.sort(key=lambda f: f.get("size") or 0)
    path = out / "download_plan.json"
    save_json(path, {"created_at": now(), "scopes": scopes, "files": selected})
    print(json.dumps(scopes, indent=2))
    print(f"PLAN {path} files={len(selected)} known_bytes={sum(f.get('size') or 0 for f in selected)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["catalog", "plan", "fetch"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--file-cap-gb", type=float, default=12)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "catalog":
        catalog(args.out)
    elif args.command == "plan":
        make_plan(args.out)
    else:
        if args.plan is None:
            parser.error("fetch requires --plan")
        fetch(args.out, args.plan, int(args.file_cap_gb * 1e9), args.workers)


if __name__ == "__main__":
    main()
