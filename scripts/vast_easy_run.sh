#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export WANDB_MODE="${WANDB_MODE:-offline}"

PYTHON_BIN="python3"

is_true() {
    local value
    value="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
    [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" || "${value}" == "on" ]]
}

if [[ "$#" -gt 0 ]]; then
    exec "$@"
fi

if ! is_true "${HGSEL_SKIP_INSTALL:-false}"; then
    VENV_DIR="${HGSEL_VENV_DIR:-${ROOT_DIR}/.venv}"
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        python3 -m venv "${VENV_DIR}"
    fi

    PYTHON_BIN="${VENV_DIR}/bin/python"
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install -r requirements.txt
    "${PYTHON_BIN}" -m pip install -e .
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi

if is_true "${HGSEL_REQUIRE_GPU:-false}"; then
    "${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("HGSEL_REQUIRE_GPU is enabled, but CUDA is not available.", file=sys.stderr)
    sys.exit(1)

print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
PY
fi

if is_true "${HGSEL_SETUP_ONLY:-false}"; then
    echo "Setup complete."
    exit 0
fi

TASK="${HGSEL_TASK:-smoke}"

case "${TASK}" in
    smoke|validate)
        "${PYTHON_BIN}" experiments/validate_phase3.py
        ;;
    benchmark)
        "${PYTHON_BIN}" experiments/benchmark_300m.py
        ;;
    train)
        TRAIN_ARGS=()
        if [[ -n "${HGSEL_TRAIN_ARGS:-}" ]]; then
            # shellcheck disable=SC2206
            TRAIN_ARGS=(${HGSEL_TRAIN_ARGS})
        fi
        "${PYTHON_BIN}" experiments/train_300m.py \
            --use-hgsel \
            --device "${HGSEL_DEVICE:-cuda}" \
            "${TRAIN_ARGS[@]}"
        ;;
    shell|bash)
        exec bash
        ;;
    *)
        echo "Unknown HGSEL_TASK='${TASK}'." >&2
        echo "Use one of: smoke, validate, benchmark, train, shell" >&2
        exit 2
        ;;
esac

if is_true "${HGSEL_KEEP_ALIVE:-false}"; then
    echo "Task finished. Keeping container alive (HGSEL_KEEP_ALIVE=true)."
    exec tail -f /dev/null
fi
