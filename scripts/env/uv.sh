# source scripts/env/uv.sh  — uv on OSC without filling home or the shared ess file quota.
# The venv lives on scratch (.venv is a symlink to it); rebuild anytime with: uv sync --group verify --group labelers
export UV_CACHE_DIR=/fs/scratch/PAS2836/mugezhang/uv_cache
export UV_PYTHON_INSTALL_DIR=/fs/ess/PAS2836/mugezhang/envs/uv_python
export UV_PROJECT_ENVIRONMENT=/fs/scratch/PAS2836/mugezhang/envs/chesticpo-venv
