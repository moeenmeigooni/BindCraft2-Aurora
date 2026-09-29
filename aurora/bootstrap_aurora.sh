#!/usr/bin/env bash
# Build BC2's native Aurora oneAPI virtual environment on a machine whose
# module environment exposes the Aurora Level Zero/oneAPI runtime.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
project_root="${AURORA_PROJECT_ROOT:-/lus/flare/projects/FRAME-IDP/${USER:-${LOGNAME:-user}}}"
env_prefix="${BINDCRAFT2_AURORA_ENV:-${project_root}/envs/bindcraft2-aurora}"
fetch_weights=true

usage() {
  cat <<'EOF'
Usage: aurora/install_aurora.sh [--env PREFIX] [--no-weights]

Creates a Python 3.12+ virtual environment and installs BindCraft2 with its
JAX oneAPI extra.  By default it also downloads the 5.3 GB AlphaFold parameter
archive into project storage.  --no-weights is suitable for the JAX smoke test.
EOF
}

while (($#)); do
  case "$1" in
    --env) env_prefix="$2"; shift 2 ;;
    --no-weights) fetch_weights=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v module >/dev/null 2>&1; then
  echo "Aurora's module command is required to load the Level Zero runtime." >&2
  exit 1
fi
set +u
module load frameworks
set -u

source "${script_dir}/runtime_env.sh" "${env_prefix}"
if [[ -e "${env_prefix}" && ! -x "${env_prefix}/bin/python" ]]; then
  echo "Environment path exists but is not a Python virtual environment: ${env_prefix}" >&2
  exit 1
fi
if [[ ! -x "${env_prefix}/bin/python" ]]; then
  mkdir -p "$(dirname -- "${env_prefix}")"
  if command -v uv >/dev/null 2>&1; then
    uv venv --seed --python '>=3.12' "${env_prefix}"
  else
    python_candidate=""
    for candidate in python3.14 python3.13 python3.12 python3; do
      command -v "${candidate}" >/dev/null 2>&1 || continue
      "${candidate}" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' || continue
      python_candidate="${candidate}"
      break
    done
    if [[ -z "${python_candidate}" ]]; then
      echo "Need uv or a Python 3.12+ interpreter to create ${env_prefix}." >&2
      exit 1
    fi
    "${python_candidate}" -m venv "${env_prefix}"
  fi
fi

"${env_prefix}/bin/python" -m pip install --upgrade pip
# The Intel JAX oneAPI PJRT wheels are published as a pre-release.  Their
# exact version is pinned in pyproject.toml; --pre permits that explicit wheel
# while leaving the rest of the environment resolver-controlled.
"${env_prefix}/bin/python" -m pip install --pre -e "${repo_dir}[oneapi]"
if [[ "${fetch_weights}" == true ]]; then
  "${env_prefix}/bin/bindcraft" fetch-weights
  "${env_prefix}/bin/python" -m bindcraft.selfcheck oneapi
else
  "${env_prefix}/bin/python" -m bindcraft.selfcheck oneapi --shipped-only
fi

echo "Environment: ${env_prefix}"
echo "Cache root: ${BINDCRAFT2_CACHE_ROOT}"
echo "Runtime home: ${BINDCRAFT2_RUNTIME_HOME}"
echo "Next: ${repo_dir}/aurora/run_bindcraft.sh ${env_prefix} python ${repo_dir}/aurora/smoke_xpu.py"
