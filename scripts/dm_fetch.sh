#!/bin/bash
# Run ON dm01 (data mover). Usage: dm_fetch.sh <filelist.tsv> <dest_dir> [strip_prefix]
# filelist: <relpath>\t<size>\t<url>. Skips files already complete (same size). Sequential, resumable.
# If ~/.cache/huggingface/token exists it is sent as a Bearer header to huggingface.co URLs (gated repos). The token is
# never echoed; curl reads the header from a temporary config file.
set -eo pipefail
LIST="$1"; DEST="$2"; STRIP="${3:-}"
mkdir -p "$DEST"
HDR=""
if [ -s "$HOME/.cache/huggingface/token" ]; then
  HDR=$(mktemp); chmod 600 "$HDR"
  printf 'header = "Authorization: Bearer %s"\n' "$(cat "$HOME/.cache/huggingface/token")" > "$HDR"
  trap 'rm -f "$HDR"' EXIT
fi
while IFS=$'\t' read -r rel size url; do
  [ -z "$rel" ] && continue
  rel="${rel#$STRIP}"
  out="$DEST/$rel"
  if [ -f "$out" ] && [ "$size" != "0" ] && [ "$(stat -c %s "$out")" = "$size" ]; then echo "skip $rel"; continue; fi
  mkdir -p "$(dirname "$out")"
  cfg=()
  case "$url" in https://huggingface.co/*) [ -n "$HDR" ] && cfg=(-K "$HDR") ;; esac
  curl -sS -L --retry 8 --retry-delay 10 --retry-all-errors -C - "${cfg[@]}" -o "$out.part" "$url"
  mv "$out.part" "$out"
  echo "done $rel $(stat -c %s "$out")"
done < "$LIST"
echo "FETCH_COMPLETE $LIST"
