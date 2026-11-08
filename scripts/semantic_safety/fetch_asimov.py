"""Fetch the original ASIMOV robot-scene subset on a HANGANG data mover.

Only public data under one fixed Google Research prefix are downloaded. No
credentials or external evaluation services are used. Existing files are checked
and never overwritten. Run with --metadata-only for a small schema inspection.
"""

import argparse
import hashlib
import json
import pathlib
import socket
import urllib.parse
import urllib.request


PREFIX = "robotics/asimov_multimodal_auto_val/0.1.0/"
BUCKET = "gresearch"
BASE = f"https://storage.googleapis.com/{BUCKET}/"
ALLOWED_METADATA = ("dataset_info.json", "features.json")


def read_url(url: str, limit: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "VLA-Safety-research/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Response exceeds the {limit}-byte limit: {url}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args()
    host = socket.gethostname().split(".")[0]
    if not args.metadata_only and host not in {"dm01", "dm02", "dm03", "dm04"}:
        raise RuntimeError("Dataset transfers must run on dm01-dm04")

    for name in ALLOWED_METADATA:
        raw = read_url(BASE + PREFIX + name, 2 * 1024 * 1024)
        obj = json.loads(raw)
        print(json.dumps({"file": name, "data": obj}, ensure_ascii=False), flush=True)
    if args.metadata_only:
        return

    listing_url = (
        f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?"
        + urllib.parse.urlencode({"prefix": PREFIX, "maxResults": 32})
    )
    listing = json.loads(read_url(listing_url, 2 * 1024 * 1024))
    if listing.get("nextPageToken"):
        raise ValueError("Unexpectedly large listing; refusing an unbounded download")
    files = listing.get("items", [])
    selected = []
    for item in files:
        key = item["name"]
        if not key.startswith(PREFIX):
            raise ValueError("Listing escaped the fixed dataset prefix")
        relative = key[len(PREFIX) :]
        if "/" in relative or relative in {"", ".", ".."}:
            continue
        if relative in ALLOWED_METADATA or ".tfrecord-" in relative:
            selected.append((relative, int(item["size"]), BASE + urllib.parse.quote(key, safe="/")))
    if not any(".tfrecord-" in name for name, _, _ in selected):
        raise ValueError("The fixed dataset prefix contains no TFRecord shards")
    total = sum(size for _, size, _ in selected)
    if total > args.max_bytes:
        raise ValueError(f"Download size {total} exceeds --max-bytes={args.max_bytes}")
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected_size, url in selected:
        destination = args.out / name
        if destination.exists():
            if destination.stat().st_size != expected_size:
                raise FileExistsError(f"Existing file has the wrong size; preserve and inspect it: {destination}")
            raw = destination.read_bytes()
        else:
            raw = read_url(url, expected_size)
            if len(raw) != expected_size:
                raise ValueError(f"Wrong size for {name}: {len(raw)} != {expected_size}")
            with destination.open("xb") as stream:
                stream.write(raw)
        records.append({"file": name, "url": url, "bytes": expected_size,
                        "sha256": hashlib.sha256(raw).hexdigest()})
        print(f"VERIFIED {name} {expected_size} bytes", flush=True)
    manifest = {"source": "Original Google Research TFDS release", "bucket_prefix": BASE + PREFIX,
                "subset": "asimov_multimodal_auto_val", "host": host, "files": records,
                "scope": "RoboVQA-derived robot-scene semantic safety; no generic chat or text-only subsets"}
    manifest_path = args.out / "download_manifest.json"
    if not manifest_path.exists():
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2)
    print(f"FETCH_COMPLETE {args.out} {total} bytes", flush=True)


if __name__ == "__main__":
    main()
