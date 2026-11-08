#!/bin/bash
# Build the openpi uv environment inside this project (no global changes).
set -eo pipefail
ROOT=/scratch/t512a07/minseong/VLA-Safety
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv-python"
export UV_PYTHON_PREFERENCE=only-managed
export GIT_LFS_SKIP_SMUDGE=1
cd "$ROOT/openpi"
"$ROOT/tools/uv" python install 3.11
"$ROOT/tools/uv" sync
"$ROOT/tools/uv" pip install -e .
"$ROOT/tools/uv" run python -c 'import jax, torch, platform; print(platform.machine(), jax.__version__, torch.__version__, torch.version.cuda)'
echo SETUP_DONE
