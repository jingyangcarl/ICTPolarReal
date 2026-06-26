#!/usr/bin/env bash
set -euo pipefail
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SAMPLE_URL="https://drive.google.com/drive/u/1/folders/1J2lfWe8rO1ZXpbeVW68u2RSqOocCs-S6"
ENV_NAME="${ENV_NAME:-ictpolarreal}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/sample}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
INPUT_NAME="${INPUT_NAME:-static}"
TARGET_NAME="${TARGET_NAME:-albedo}"
MAX_LIGHTS="${MAX_LIGHTS:-8}"
BACKEND="${BACKEND:-auto}"
DEVICE="${DEVICE:-cuda}"
TRAIN_STEPS="${TRAIN_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PRED_ROOT_EXPLICIT=0
if [[ -n "${PRED_ROOT:-}" ]]; then
  PRED_ROOT_EXPLICIT=1
fi
PRED_ROOT="${PRED_ROOT:-${OUTPUT_ROOT}/train_${TARGET_NAME}/predictions}"
EVAL_MODE="${EVAL_MODE:-ictpolarreal}"
EVAL_TASK="${EVAL_TASK:-decomposition}"
EVAL_MANIFEST="${EVAL_MANIFEST:-}"
TORCH_VARIANT="${TORCH_VARIANT:-auto}"
DOWNLOAD_SAMPLE=0
SKIP_SETUP=0
SKIP_PROCESS=0
SKIP_TRAIN=0

usage() {
  cat <<EOF
Usage:
  bash run.sh <command> [options]

Commands:
  setup        Create/activate an environment and install ICTPolarReal.
  check-env    Verify Python package imports and CUDA availability.
  check-data   Validate DATA_ROOT and print Google Drive sample instructions if missing.
  process      Process OLAT cross/parallel images into diffuse/specular material previews.
  train        Run a tiny inverse-decomposition training job.
  evaluate     Evaluate predictions against ICTPolarReal or Objaverse-style samples.
  all          setup -> check-env -> check-data -> process -> train -> evaluate.

Options:
  --data-root PATH          Dataset root. Default: ${DATA_ROOT}
  --output-root PATH        Output root. Default: ${OUTPUT_ROOT}
  --env-name NAME           Conda/micromamba env name. Default: ${ENV_NAME}
  --input NAME              Training input image stem. Default: ${INPUT_NAME}
  --target NAME             Training target image stem. Default: ${TARGET_NAME}
  --max-lights N            Number of OLAT lights to process. Default: ${MAX_LIGHTS}
  --backend auto|cpu|torch  Processing backend. Default: ${BACKEND}
  --device DEVICE           Torch device for processing/training. Default: ${DEVICE}
  --train-steps N           Training smoke-test steps. Default: ${TRAIN_STEPS}
  --batch-size N            Training batch size. Default: ${BATCH_SIZE}
  --pred-root PATH          Prediction root for training/evaluation. Default: ${PRED_ROOT}
  --eval-mode MODE          ictpolarreal or objaverse. Default: ${EVAL_MODE}
  --eval-task TASK          decomposition or relighting. Default: ${EVAL_TASK}
  --eval-manifest PATH      Optional Objaverse/ICTPolarReal evaluation manifest.
  --torch-variant VARIANT   auto, cpu, cu121, cu124, cu126, cu128, or pypi. Default: ${TORCH_VARIANT}
  --download-sample         Try to download the Google Drive sample with gdown. The all command does this automatically.
  --skip-setup              For all: use the current environment.
  --skip-process            For all: skip material preprocessing.
  --skip-train              For all: skip training.

Examples:
  bash run.sh all
  bash run.sh check-data
  bash run.sh process --backend torch --device cuda
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --data-root) DATA_ROOT="$2"; shift 2 ;;
      --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
      --env-name) ENV_NAME="$2"; shift 2 ;;
      --input) INPUT_NAME="$2"; shift 2 ;;
      --target) TARGET_NAME="$2"; shift 2 ;;
      --max-lights) MAX_LIGHTS="$2"; shift 2 ;;
      --backend) BACKEND="$2"; shift 2 ;;
      --device) DEVICE="$2"; shift 2 ;;
      --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
      --batch-size) BATCH_SIZE="$2"; shift 2 ;;
      --pred-root) PRED_ROOT="$2"; PRED_ROOT_EXPLICIT=1; shift 2 ;;
      --eval-mode) EVAL_MODE="$2"; shift 2 ;;
      --eval-task) EVAL_TASK="$2"; shift 2 ;;
      --eval-manifest) EVAL_MANIFEST="$2"; shift 2 ;;
      --torch-variant) TORCH_VARIANT="$2"; shift 2 ;;
      --download-sample) DOWNLOAD_SAMPLE=1; shift ;;
      --skip-setup) SKIP_SETUP=1; shift ;;
      --skip-process) SKIP_PROCESS=1; shift ;;
      --skip-train) SKIP_TRAIN=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown option: $1"; usage; exit 2 ;;
    esac
  done
  if [[ "${PRED_ROOT_EXPLICIT}" != "1" ]]; then
    PRED_ROOT="${OUTPUT_ROOT}/train_${TARGET_NAME}/predictions"
  fi
}

activate_env() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
      conda activate "${ENV_NAME}"
    fi
  elif command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s bash)"
    if micromamba env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
      micromamba activate "${ENV_NAME}"
    fi
  elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
  fi
}

setup_env() {
  cd "${REPO_ROOT}"
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
      conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
    fi
    conda activate "${ENV_NAME}"
  elif command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s bash)"
    if ! micromamba env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
      micromamba create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
    fi
    micromamba activate "${ENV_NAME}"
  else
    python3 -m venv "${REPO_ROOT}/.venv"
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
  fi
  python -m pip install --upgrade pip
  python -m pip install -e ".[dev]" gdown
  install_torch
}

install_torch() {
  local variant="${TORCH_VARIANT}"
  if [[ "${variant}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
      variant="cu126"
    else
      variant="cpu"
    fi
  fi
  if torch_matches_variant "${variant}"; then
    echo "[setup] Existing torch installation matches ${variant}."
    return 0
  fi
  case "${variant}" in
    cpu)
      python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
      ;;
    cu121|cu124|cu126|cu128)
      python -m pip install --force-reinstall torch --index-url "https://download.pytorch.org/whl/${variant}"
      ;;
    pypi)
      python -m pip install --upgrade torch
      ;;
    none)
      echo "[setup] Skipping torch installation because --torch-variant none was set."
      ;;
    *)
      echo "Unknown --torch-variant: ${variant}"
      exit 2
      ;;
  esac
}

torch_matches_variant() {
  local variant="$1"
  python - "$variant" <<'PY'
import sys
variant = sys.argv[1]
try:
    import torch
except ModuleNotFoundError:
    raise SystemExit(1)
cuda = torch.version.cuda
if variant == "cpu":
    raise SystemExit(0 if cuda is None else 1)
if variant.startswith("cu"):
    expected = {"cu121": "12.1", "cu124": "12.4", "cu126": "12.6", "cu128": "12.8"}[variant]
    raise SystemExit(0 if cuda and cuda.startswith(expected) else 1)
if variant == "pypi":
    raise SystemExit(0)
raise SystemExit(1)
PY
}

check_env() {
  cd "${REPO_ROOT}"
  activate_env || true
  python - <<'PY'
import importlib
import sys

print(f"[env] python: {sys.executable}")
for name in ["ictpolarreal", "numpy", "imageio", "PIL", "yaml", "tqdm"]:
    importlib.import_module(name)
    print(f"[env] import ok: {name}")
try:
    import torch
    print(f"[env] torch: {torch.__version__}")
    print(f"[env] cuda_available: {torch.cuda.is_available()}")
except ModuleNotFoundError:
    print("[env] torch: missing; training and torch backend require setup/install")
PY
}

download_sample_if_requested() {
  if [[ "${DOWNLOAD_SAMPLE}" != "1" ]]; then
    return 0
  fi
  cd "${REPO_ROOT}"
  activate_env || true
  mkdir -p "${DATA_ROOT}"
  echo "[data] Downloading sample folder with gdown:"
  echo "       ${SAMPLE_URL}"
  echo "[data] This can take several minutes for the full sample."
  if ! python - "${SAMPLE_URL}" "${DATA_ROOT}" <<'PY'; then
import inspect
import sys

url, output = sys.argv[1:3]
kwargs = {"url": url, "output": output, "quiet": True}
try:
    import gdown

    signature = inspect.signature(gdown.download_folder)
    if "remaining_ok" in signature.parameters:
        kwargs["remaining_ok"] = True
    if "resume" in signature.parameters:
        kwargs["resume"] = True
    downloaded = gdown.download_folder(**kwargs)
except Exception as exc:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    print(f"[data] gdown failed: {message}")
    raise SystemExit(1)
if downloaded is None:
    raise SystemExit(1)
print(f"[data] gdown returned {len(downloaded)} file(s).")
PY
    cat <<EOF
[data] Google Drive download did not complete.
[data] This usually means a file permission, quota, or rate-limit issue in the Drive folder.
[data] Open the folder in a browser, make sure files are shared with anyone who has the link,
[data] then rerun:
[data]   bash run.sh check-data --data-root "${DATA_ROOT}" --download-sample
EOF
    return 3
  fi
}

check_data() {
  cd "${REPO_ROOT}"
  activate_env || true
  if python -m ictpolarreal.data.check --data-root "${DATA_ROOT}" --min-lights 1; then
    return 0
  fi
  if [[ "${DOWNLOAD_SAMPLE}" == "1" ]]; then
    if ! download_sample_if_requested; then
      python -m ictpolarreal.data.check --data-root "${DATA_ROOT}" --min-lights 1 || true
      return 3
    fi
    python -m ictpolarreal.data.check --data-root "${DATA_ROOT}" --min-lights 1
    return 0
  fi
  return 2
}

ensure_data_for_all() {
  if check_data; then
    return 0
  fi
  echo "[data] Data is not ready. Trying to download the sample dataset automatically."
  DOWNLOAD_SAMPLE=1
  if ! download_sample_if_requested; then
    python -m ictpolarreal.data.check --data-root "${DATA_ROOT}" --min-lights 1 || true
    return 3
  fi
  python -m ictpolarreal.data.check --data-root "${DATA_ROOT}" --min-lights 1
}

process_materials() {
  cd "${REPO_ROOT}"
  activate_env || true
  python -m ictpolarreal.processing.prepare_materials \
    --data-root "${DATA_ROOT}" \
    --out-root "${OUTPUT_ROOT}/materials" \
    --max-lights "${MAX_LIGHTS}" \
    --backend "${BACKEND}" \
    --device "${DEVICE}" \
    --preview \
    --save-aggregate
}

train_baseline() {
  cd "${REPO_ROOT}"
  activate_env || true
  python -m ictpolarreal.data.check \
    --data-root "${DATA_ROOT}" \
    --min-lights 1 \
    --require-target "${TARGET_NAME}"
  python -m ictpolarreal.train.inverse \
    --data-root "${DATA_ROOT}" \
    --out-dir "${OUTPUT_ROOT}/train_${TARGET_NAME}" \
    --input "${INPUT_NAME}" \
    --target "${TARGET_NAME}" \
    --max-steps "${TRAIN_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --pred-dir "${PRED_ROOT}"
}

evaluate_predictions() {
  cd "${REPO_ROOT}"
  activate_env || true
  local manifest_args=()
  if [[ -n "${EVAL_MANIFEST}" ]]; then
    manifest_args=(--manifest "${EVAL_MANIFEST}")
  fi
  python -m ictpolarreal.eval.run \
    --dataset-mode "${EVAL_MODE}" \
    --task "${EVAL_TASK}" \
    --gt-root "${DATA_ROOT}" \
    --pred-root "${PRED_ROOT}" \
    --out-dir "${OUTPUT_ROOT}/eval_${EVAL_MODE}_${EVAL_TASK}" \
    --target "${TARGET_NAME}" \
    "${manifest_args[@]}"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 2
  fi
  local command="$1"
  shift
  parse_args "$@"

  case "${command}" in
    setup) setup_env ;;
    check-env) check_env ;;
    check-data) check_data ;;
    process) check_data; process_materials ;;
    train) train_baseline ;;
    evaluate) evaluate_predictions ;;
    all)
      if [[ "${SKIP_SETUP}" != "1" ]]; then setup_env; fi
      check_env
      ensure_data_for_all
      if [[ "${SKIP_PROCESS}" != "1" ]]; then process_materials; fi
      if [[ "${SKIP_TRAIN}" != "1" ]]; then
        train_baseline
        evaluate_predictions
      fi
      ;;
    -h|--help) usage ;;
    *) echo "Unknown command: ${command}"; usage; exit 2 ;;
  esac
}

main "$@"
