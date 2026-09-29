#!/usr/bin/env bash
# The supported Aurora entry point. It is intentionally a wrapper rather than
# an activation snippet so every invocation receives the no-home cache policy.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: aurora/run_bindcraft.sh VENV_PREFIX COMMAND [ARG ...]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_prefix="$1"
shift

if ! command -v module >/dev/null 2>&1; then
  echo "Aurora's module command is required to load the Level Zero runtime." >&2
  exit 1
fi
set +u
module load "${AURORA_JAX_ONEAPI_MODULE:-oneapi/release/2025.3.1}"
set -u
source "${script_dir}/runtime_env.sh" "${env_prefix}"

case "$1" in
  bindcraft) shift; exec "${env_prefix}/bin/bindcraft" "$@" ;;
  python) shift; exec "${env_prefix}/bin/python" "$@" ;;
  *) exec "$@" ;;
esac
