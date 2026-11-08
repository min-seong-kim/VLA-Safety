#!/bin/bash
# Bounded second transfer: prefetch OFT's final two shards while the main transfer handles earlier files.
set -euo pipefail
case "$(hostname -s)" in dm0[1-4]) ;; *) exit 2 ;; esac
cd /scratch/t512a07/minseong/VLA-Safety
model=openvla-7b-oft-libero-goal
while IFS=$'\t' read -r rel size url; do
    case "$rel" in model-00003-of-00004.safetensors|model-00004-of-00004.safetensors) ;; *) continue ;; esac
    out="checkpoints/hf/$model/$rel"
    if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then continue; fi
    # A distinct staging path makes overlap with the main resumable transfer harmless.
    curl --fail --silent --show-error --location --connect-timeout 30 --retry 3 --retry-delay 5 \
        --continue-at - --output "$out.prefetch" "$url"
    test "$(stat -c %s "$out.prefetch")" = "$size"
    mv "$out.prefetch" "$out"
    printf 'prefetched %s %s bytes\n' "$rel" "$size"
done < "checkpoints/hf/$model.filelist.tsv"
