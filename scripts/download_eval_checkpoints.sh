#!/bin/bash
# Download only merged VLM weights and metadata, from pinned public HF file lists, on a data mover.
set -euo pipefail
case "$(hostname -s)" in dm0[1-4]) ;; *) echo 'Run this transfer on dm01-dm04.' >&2; exit 2 ;; esac
cd /scratch/t512a07/minseong/VLA-Safety
for model in openvla-7b-finetuned-libero-spatial openvla-7b-oft-libero-goal; do
    dest="checkpoints/hf/$model"
    mkdir -p "$dest"
    while IFS=$'\t' read -r rel size url; do
        case "$rel" in .gitattributes|lora_adapter/*|*.pt) continue ;; esac
        case "$url" in https://huggingface.co/*) ;; *) exit 2 ;; esac
        case "$rel" in /*|*..*) exit 2 ;; esac
        out="$dest/$rel"
        if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then
            printf 'skip %s/%s\n' "$model" "$rel"
            continue
        fi
        mkdir -p "$(dirname "$out")"
        curl --http1.1 --fail --silent --show-error --location --connect-timeout 30 \
            --speed-limit 1024 --speed-time 120 --max-time 1800 --retry 3 --retry-delay 5 --retry-all-errors \
            --continue-at - --output "$out.part" "$url"
        actual=$(stat -c %s "$out.part")
        if [ "$actual" != "$size" ]; then
            printf 'Size mismatch: %s expected=%s actual=%s\n' "$out" "$size" "$actual" >&2
            exit 1
        fi
        mv "$out.part" "$out"
        printf 'done %s/%s %s bytes\n' "$model" "$rel" "$size"
    done < "checkpoints/hf/$model.filelist.tsv"
done
