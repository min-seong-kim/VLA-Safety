"""Fetch many small images into ONE tar file (Lustre inode friendly). Runs on the data-mover node with system python3.

Usage: python3 dm_fetch_images.py <needed.tsv: id<TAB>url> <out.tar> [threads=6]
Resumable: ids already present in the tar are skipped. Each image is verified to start with a JPEG/PNG signature.
"""
import concurrent.futures
import io
import os
import sys
import tarfile
import threading
import time
import urllib.request

needed, out_path = sys.argv[1], sys.argv[2]
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 6
todo = [line.rstrip("\n").split("\t") for line in open(needed) if line.strip()]
done = set()
if os.path.exists(out_path):
    with tarfile.open(out_path) as t:
        done = {m.name for m in t.getmembers()}
todo = [(i, u) for i, u in todo if "%s.jpg" % i not in done]
print("to fetch:", len(todo), "already:", len(done), flush=True)
lock = threading.Lock()
tar = tarfile.open(out_path, "a")
failed = []


def fetch(item):
    img_id, url = item
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if not (data[:2] == b"\xff\xd8" or data[:8] == b"\x89PNG\r\n\x1a\n"):
                raise ValueError("not an image")
            info = tarfile.TarInfo("%s.jpg" % img_id)
            info.size = len(data)
            info.mtime = int(time.time())
            with lock:
                tar.addfile(info, io.BytesIO(data))
            return True
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 * (attempt + 1))
    with lock:
        failed.append((img_id, url, str(err)))
    return False


n_ok = 0
with concurrent.futures.ThreadPoolExecutor(threads) as ex:
    for k, ok in enumerate(ex.map(fetch, todo), 1):
        n_ok += ok
        if k % 200 == 0:
            print("progress", k, "/", len(todo), "ok", n_ok, flush=True)
tar.close()
print("fetched", n_ok, "failed", len(failed))
for f in failed[:20]:
    print("FAILED", *f)
print("IMAGE_FETCH_COMPLETE" if not failed else "IMAGE_FETCH_PARTIAL")
