"""Fetch selected official RoboCasa assets without downloading the full ZIP archive."""

import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

from rollout_common import ROOT, write_json

DATA = ROOT / "data/robocasa_libero_import_v1"
REPOSITORY = "robocasa/robocasa-assets"


class RangeArchive(io.RawIOBase):
    """Seekable, bounded HTTP range reader with an eight-block cache."""

    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.cache = OrderedDict()
        self.downloaded_bytes = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, count=-1):
        end = self.size if count < 0 else min(self.size, self.position + count)
        pieces = []
        block_size = 1024 * 1024
        while self.position < end:
            start = self.position // block_size * block_size
            stop = min(self.size, start + block_size)
            if start not in self.cache:
                url = self.url + f"?range_start={start}&range_end={stop - 1}"
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop - 1}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected = f"bytes {start}-{stop - 1}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Server did not honor the requested byte range")
                    payload = response.read(stop - start + 1)
                assert len(payload) == stop - start
                self.cache[start] = payload
                self.downloaded_bytes += len(payload)
                if len(self.cache) > 8:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(start)
            take = min(end, stop) - self.position
            pieces.append(self.cache[start][self.position - start:self.position - start + take])
            self.position += take
        return b"".join(pieces)


def open_archive(pack):
    metadata_path = DATA / "upstream/huggingface_source.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPOSITORY}", timeout=30) as response:
            metadata = json.load(response)
        write_json(metadata_path, {"repo": REPOSITORY, "sha": metadata["sha"]})
    revision = metadata["sha"]
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{pack}.zip"
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])
    reader = RangeArchive(url, size)
    return reader, zipfile.ZipFile(reader), revision


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pack", choices=["objaverse", "aigen_objs"], default="objaverse")
    parser.add_argument("--models", nargs="+", help="Exact model directory names relative to the pack")
    args = parser.parse_args()
    reader, archive, revision = open_archive(args.pack)
    with archive:
        members = archive.infolist()
        index = [{"name": info.filename, "bytes": info.file_size, "compressed": info.compress_size}
                 for info in members]
        write_json(DATA / "upstream" / f"{args.pack}_index.json", index)
        if not args.models:
            for info in members:
                parts = PurePosixPath(info.filename).parts
                if info.filename.endswith("/model.xml") and any(
                        category in parts for category in ("bread", "knife", "spray")):
                    print(info.filename)
        else:
            for model in args.models:
                prefix = args.pack + "/" + model.strip("/") + "/"
                selected = [info for info in members if info.filename.startswith(prefix) and not info.is_dir()]
                assert selected and any(info.filename == prefix + "model.xml" for info in selected), prefix
                provenance = []
                for info in selected:
                    path = PurePosixPath(info.filename)
                    assert not path.is_absolute() and ".." not in path.parts
                    assert info.file_size < 100 * 1024 * 1024, info.filename
                    target = DATA / "original" / path
                    payload = archive.read(info)
                    if target.exists():
                        assert target.read_bytes() == payload, f"Original asset changed: {target}"
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(payload)
                    provenance.append({"path": info.filename, "bytes": len(payload),
                                       "sha256": hashlib.sha256(payload).hexdigest()})
                write_json(DATA / "original" / args.pack / model / "import_provenance.json", {
                    "source": REPOSITORY, "revision": revision, "archive": args.pack + ".zip",
                    "license_as_stated_by_robocasa_readme": "CC BY 4.0",
                    "files": provenance})
                print(json.dumps({"model": model, "files": len(provenance),
                                  "extracted_bytes": sum(row["bytes"] for row in provenance)}), flush=True)
    print(json.dumps({"archive_bytes": reader.size, "transferred_bytes": reader.downloaded_bytes}), flush=True)


if __name__ == "__main__":
    main()
