#!/bin/bash
# Build two project-local uv venvs for the Llama-family VLMs (aarch64 + CUDA 12.8 torch wheels).
#   envs/llama : OpenVLA-7B and Prismatic (upstream transformers 4.40.1, causal attention)
#   envs/oft   : OpenVLA-OFT (moojink transformers fork 4.40.1, bidirectional attention) -- isolated process/env
# Nothing global is modified; all caches stay inside the project.
set -eo pipefail
ROOT=/scratch/t512a07/minseong/VLA-Safety
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv-python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_CONCURRENT_DOWNLOADS=4
UV="$ROOT/tools/uv"
TORCH_IDX="https://download.pytorch.org/whl/cu128"
COMMON=(
  "torch==2.7.1" "torchvision==0.22.1"
  "tokenizers==0.19.1" "timm==0.9.10" "accelerate>=0.25.0" "peft==0.11.1" "safetensors" "sentencepiece"
  "numpy<2" "pillow" "scikit-learn" "pyarrow" "einops" "huggingface_hub" "protobuf" "pyyaml"
)
make_env() {  # name, transformers spec
  local name="$1" tf="$2" dir="$ROOT/envs/$name"
  "$UV" venv --python 3.11 "$dir"
  "$UV" pip install --python "$dir/bin/python" --index-url "$TORCH_IDX" --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match "${COMMON[@]}" "$tf"
  "$dir/bin/python" - <<'PY'
import torch, torchvision, transformers, tokenizers, timm, sklearn, numpy, platform
print(platform.machine(), "torch", torch.__version__, torch.version.cuda, "tv", torchvision.__version__,
      "transformers", transformers.__version__, "tokenizers", tokenizers.__version__, "timm", timm.__version__,
      "sklearn", sklearn.__version__, "numpy", numpy.__version__)
PY
}
make_env llama "transformers==4.40.1"
make_env oft "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"
echo ENVS_DONE
