"""Download pinned official LIBERO checkpoints using recorded publisher metadata."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data/model_sources"


def inventory():
    files = []
    for name in ["pi0_libero", "pi05_libero"]:
        data = json.loads((CATALOG / (name + ".gcs.json")).read_text())
        assert not data.get("nextPageToken"), "Incomplete GCS inventory"
        for f in data["items"]:
            relative = f["name"].removeprefix("checkpoints/" + name + "/")
            files.append({"checkpoint": name, "path": relative, "size": int(f["size"]),
                          "md5": f.get("md5Hash"), "generation": f["generation"],
                          "url": "https://storage.googleapis.com/openpi-assets/" + quote(f["name"], safe="/")
                          + "?generation=" + f["generation"]})
    for local, source in [
        ("openvla_libero_spatial", "openvla-7b-finetuned-libero-spatial"),
        ("openvla_oft_libero_spatial", "openvla-7b-oft-finetuned-libero-spatial"),
    ]:
        data = json.loads((CATALOG / (source + ".hf.json")).read_text())
        for f in data["siblings"]:
            files.append({"checkpoint": local, "path": f["rfilename"], "size": f["size"],
                          "sha256": f.get("lfs", {}).get("sha256"), "revision": data["sha"],
                          "url": "https://huggingface.co/" + data["id"] + "/resolve/" + data["sha"]
                          + "/" + quote(f["rfilename"], safe="/") + "?download=true"})
    return files


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode()


def valid(spec, path):
    if not path.exists() or path.stat().st_size != spec["size"]:
        return None
    sha, md5 = hashes(path)
    if spec.get("sha256") and spec["sha256"] != sha:
        return None
    if spec.get("md5") and spec["md5"] != md5:
        return None
    return {**spec, "download_sha256": sha, "local_path": str(path.relative_to(ROOT))}


def fetch(spec):
    path = ROOT / "checkpoints" / spec["checkpoint"] / spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    result = valid(spec, path)
    if result:
        return result
    temp = path.with_name(path.name + ".part")
    for attempt in range(4):
        try:
            url = spec["url"] + ("&refresh=" + str(time.time_ns()) if attempt else "")
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temp.open("wb") as handle:
                    for block in response.iter_content(8 * 1024 * 1024):
                        handle.write(block)
            result = valid(spec, temp)
            if result is None:
                raise ValueError("Publisher size or hash mismatch: " + spec["path"])
            temp.replace(path)
            result["local_path"] = str(path.relative_to(ROOT))
            print(f"OK {spec['checkpoint']}/{spec['path']} {spec['size'] / 1e6:.1f} MB", flush=True)
            return result
        except Exception as error:
            print(f"RETRY {spec['checkpoint']}/{spec['path']}: {error}", flush=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    files = inventory()
    (CATALOG / "rollout_download_plan.json").write_text(json.dumps(files, indent=2) + "\n")
    print(f"FILES {len(files)} TOTAL_GB {sum(f['size'] for f in files) / 1e9:.3f}", flush=True)
    complete, errors = [], []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch, f): f for f in files}
        for future in as_completed(futures):
            try:
                complete.append(future.result())
            except Exception as error:
                errors.append({"file": futures[future], "error": str(error)})
            (CATALOG / "rollout_download_results.json").write_text(json.dumps(
                {"complete": sorted(complete, key=lambda f: (f["checkpoint"], f["path"])), "errors": errors},
                indent=2) + "\n")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
