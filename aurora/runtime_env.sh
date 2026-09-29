#!/usr/bin/env bash
# Source this before every BindCraft2 command on Aurora.  It keeps state and
# caches on project storage, never in the user's real home directory.

if [[ $# -ne 1 ]]; then
  echo "Usage: source aurora/runtime_env.sh VENV_PREFIX" >&2
  return 2 2>/dev/null || exit 2
fi

bindcraft_env="$1"
bindcraft_aurora_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bindcraft_env_parent="$(dirname -- "${bindcraft_env}")"
bindcraft_env_name="$(basename -- "${bindcraft_env}")"
bindcraft_cache_root="${BINDCRAFT2_CACHE_ROOT:-${bindcraft_env_parent}/.${bindcraft_env_name}-cache}"
bindcraft_runtime_home="${BINDCRAFT2_RUNTIME_HOME:-${bindcraft_env_parent}/.${bindcraft_env_name}-runtime-home}"

# The shipped Intel JAX plugin is linked against the oneAPI 2025 ABI. Aurora's
# frameworks/2026.1.0 module now selects oneAPI 2026.1, so use the PE 26.26.0
# module rebuilt for the current OS and GPU driver image.
if ! command -v module >/dev/null 2>&1; then
  echo "Aurora's module command is required to load the JAX oneAPI runtime." >&2
  return 2 2>/dev/null || exit 2
fi
set +u
module load "${AURORA_JAX_ONEAPI_MODULE:-oneapi/release/2025.3.1}"
set -u

mkdir -p \
  "${bindcraft_cache_root}/pip" \
  "${bindcraft_cache_root}/uv" \
  "${bindcraft_cache_root}/xdg" \
  "${bindcraft_cache_root}/jax" \
  "${bindcraft_cache_root}/weights" \
  "${bindcraft_cache_root}/matplotlib" \
  "${bindcraft_cache_root}/logs" \
  "${bindcraft_cache_root}/python-userbase" \
  "${bindcraft_cache_root}/tmp" \
  "${bindcraft_runtime_home}"

export BINDCRAFT2_CACHE_ROOT="${bindcraft_cache_root}"
export BINDCRAFT2_RUNTIME_HOME="${bindcraft_runtime_home}"
export HOME="${bindcraft_runtime_home}"
export XDG_CACHE_HOME="${bindcraft_cache_root}/xdg"
export XDG_CONFIG_HOME="${bindcraft_cache_root}/xdg/config"
export XDG_DATA_HOME="${bindcraft_cache_root}/xdg/data"
export PIP_CACHE_DIR="${bindcraft_cache_root}/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export UV_CACHE_DIR="${bindcraft_cache_root}/uv"
export PYTHONNOUSERSITE=1
export PYTHONUSERBASE="${bindcraft_cache_root}/python-userbase"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${bindcraft_aurora_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${bindcraft_cache_root}/tmp"
export MPLCONFIGDIR="${bindcraft_cache_root}/matplotlib"
export HF_HOME="${bindcraft_cache_root}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${bindcraft_cache_root}/torch"
export BINDCRAFT_CACHE_ROOT="${bindcraft_cache_root}"
export BINDCRAFT_WEIGHTS="${bindcraft_cache_root}/weights"
export JAX_COMPILATION_CACHE_DIR="${bindcraft_cache_root}/jax"
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
# The oneAPI plugin wheels install MKL/SYCL libraries in this venv. Expose
# them to dlopen in addition to Aurora's module-provided runtime.
export LD_LIBRARY_PATH="${bindcraft_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Aurora's Level Zero mask is assigned by the existing endpoint.  Preserve it
# unchanged: after masking, the assigned tile is local oneAPI device 0.
export BINDCRAFT_ACCELERATOR=oneapi
export JAX_PLATFORMS=oneapi
export JAX_ONEAPI_VISIBLE_DEVICES="${JAX_ONEAPI_VISIBLE_DEVICES:-0}"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export SYCL_DEVICE_FILTER=level_zero:gpu
export BINDCRAFT_WORKERS_PER_GPU=1
export BINDCRAFT_MAX_WORKERS_PER_GPU=1
