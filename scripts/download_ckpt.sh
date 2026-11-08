#!/bin/bash
# Download public openpi checkpoints from GCS over HTTPS (no gsutil needed).
# Usage: download_ckpt.sh <name> [<name>...]   e.g. pi0_base pi05_base
set -eo pipefail
ROOT=/scratch/t512a07/minseong/VLA-Safety
for name in "$@"; do
  dest="$ROOT/checkpoints/$name"
  mkdir -p "$dest"
  curl -sS -m 60 "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/$name/&fields=items(name,size)" \
  | python3 -c 'import sys,json; [print(i["name"], i["size"]) for i in json.load(sys.stdin)["items"]]' \
  | while read -r obj size; do
      rel="${obj#checkpoints/$name/}"
      out="$dest/$rel"
      if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then continue; fi
      mkdir -p "$(dirname "$out")"
      enc=$(python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$obj")
      curl -sS -L --retry 5 --retry-delay 5 -o "$out.part" "https://storage.googleapis.com/openpi-assets/$enc"
      mv "$out.part" "$out"
      echo "$name $rel $size"
    done
  echo "DONE $name"
done
