# Source this file inside a Slurm job (sbatch/salloc). It activates the project-local
# openpi uv environment and keeps every cache inside the project directory.
: "${SLURM_SUBMIT_DIR:?Slurm 작업 안에서 불러오세요}"
export PROJECT_ROOT="$SLURM_SUBMIT_DIR"
VENV="$PROJECT_ROOT/openpi/.venv"
test -x "$VENV/bin/python"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OPENPI_DATA_HOME="$PROJECT_ROOT/.cache/openpi"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache/xdg"
export JAX_COMPILATION_CACHE_DIR="$PROJECT_ROOT/.cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"
