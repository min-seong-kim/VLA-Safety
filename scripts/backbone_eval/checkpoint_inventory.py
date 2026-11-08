"""Record checkpoint provenance and verify expected local file sizes before GPU submission.

Only small metadata is fetched here. Model weights are transferred on a data mover.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/backbone_eval_v2_checkpoints"
HF = ["openvla-7b-finetuned-libero-spatial", "openvla-7b-oft-libero-goal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-only", action="store_true")
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / "checkpoint_sources.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    if not args.local_only:
        for model in ["pi0_libero", "pi05_libero"]:
            prefix = f"checkpoints/{model}/"
            url = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?" + urllib.parse.urlencode({
                "prefix": prefix, "maxResults": 1000,
                "fields": "items(name,size,generation,md5Hash,updated),nextPageToken",
            })
            with urllib.request.urlopen(url, timeout=30) as response:
                remote = json.load(response)
            if remote.get("nextPageToken"):
                raise ValueError("Unexpected paginated GCS checkpoint inventory")
            files = remote.get("items", [])
            if not files:
                raise ValueError(f"Empty checkpoint inventory: {model}")
            for record in files:
                local = ROOT / record["name"]
                if not local.is_file() or local.stat().st_size != int(record["size"]):
                    raise ValueError(f"Missing/size mismatch: {local}")
            info[model] = {"source": f"gs://openpi-assets/{prefix}", "objects": files,
                           "verification": "all listed object sizes match; large weight content not rehashed",
                           "n_files": len(files), "bytes": sum(int(r["size"]) for r in files)}
    if not args.gcs_only:
        for model in HF:
            files = []
            for line in (ROOT / "checkpoints/hf" / f"{model}.filelist.tsv").read_text().splitlines():
                name, size, url = line.split("\t")
                if name == ".gitattributes" or name.startswith("lora_adapter/") or name.endswith(".pt"):
                    continue
                local = ROOT / "checkpoints/hf" / model / name
                if not local.is_file() or local.stat().st_size != int(size):
                    raise ValueError(f"Missing/size mismatch: {local}")
                record = {"file": name, "bytes": int(size), "url": url}
                if int(size) < 5_000_000:
                    record["sha256"] = hashlib.sha256(local.read_bytes()).hexdigest()
                files.append(record)
            info[model] = {"files": files, "n_files": len(files), "bytes": sum(r["bytes"] for r in files),
                           "verification": "revision-pinned URLs and exact sizes; SHA256 of small metadata/code files"}
    info["_verified_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.write_text(json.dumps(info, indent=2) + "\n")
    for model, record in info.items():
        if not model.startswith("_"):
            print(f"{model}: {record['n_files']} files, {record['bytes']/1e9:.2f} GB, sizes verified")


if __name__ == "__main__":
    main()
