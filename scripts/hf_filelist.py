"""Write a TSV (relative_path, size, url) of every file in a HF repo (or a sub-folder), for curl-based download."""
import json
import pathlib
import sys
import urllib.parse
import urllib.request

repo, kind, out = sys.argv[1], sys.argv[2], sys.argv[3]  # kind: model | dataset
prefix = sys.argv[4] if len(sys.argv) > 4 else ""
api = f"https://huggingface.co/api/{'models' if kind == 'model' else 'datasets'}/{repo}?blobs=true"
req = urllib.request.Request(api)
tok = pathlib.Path.home() / ".cache/huggingface/token"
if tok.is_file():
    req.add_header("Authorization", "Bearer " + tok.read_text().strip())
with urllib.request.urlopen(req, timeout=60) as r:
    info = json.load(r)
base = f"https://huggingface.co/{'' if kind == 'model' else 'datasets/'}{repo}/resolve/{info['sha']}/"
n = 0
with open(out, "w") as f:
    for s in info["siblings"]:
        name = s["rfilename"]
        if prefix and not name.startswith(prefix):
            continue
        f.write(f"{name}\t{s.get('size', 0)}\t{base}{urllib.parse.quote(name)}\n")
        n += 1
print(f"{repo}@{info['sha'][:8]}: {n} files -> {out}")
