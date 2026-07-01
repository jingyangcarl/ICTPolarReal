#!/usr/bin/env bash
set -euo pipefail
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SAMPLE_URL="https://drive.google.com/drive/u/1/folders/1J2lfWe8rO1ZXpbeVW68u2RSqOocCs-S6"
ENV_NAME="${ENV_NAME:-ictpolarreal}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/sample}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
MATERIAL_ROOT_EXPLICIT=0
if [[ -n "${MATERIAL_ROOT:-}" ]]; then
  MATERIAL_ROOT_EXPLICIT=1
fi
MATERIAL_ROOT="${MATERIAL_ROOT:-${OUTPUT_ROOT}/material_acquisition}"
INPUT_NAME="${INPUT_NAME:-polarization}"
TARGET_NAME="${TARGET_NAME:-albedo}"
INPUT_MODE="${INPUT_MODE:-polarization}"
TARGET_MODE="${TARGET_MODE:-image}"
TRAIN_STAGE="${TRAIN_STAGE:-both}"
FORWARD_INPUT="${FORWARD_INPUT:-gbuffer}"
FORWARD_INPUT_MODE="${FORWARD_INPUT_MODE:-gbuffer}"
FORWARD_TARGET="${FORWARD_TARGET:-static}"
FORWARD_TARGET_MODE="${FORWARD_TARGET_MODE:-image}"
MAX_LIGHTS="${MAX_LIGHTS:-346}"
MIN_DECOMP_LIGHTS="${MIN_DECOMP_LIGHTS:-32}"
REQUIRED_DECOMP_LIGHTS="${MIN_DECOMP_LIGHTS}"
LIGHT_START="${LIGHT_START:-0}"
LIGHT_ROOT="${LIGHT_ROOT:-}"
FRAME_LAYOUT="${FRAME_LAYOUT:-auto}"
BACKEND="${BACKEND:-auto}"
DEVICE="${DEVICE:-cuda}"
DECOMP_NOISE="${DECOMP_NOISE:-1.5e-3}"
NORMAL_STEPS="${NORMAL_STEPS:-30}"
SIGMA_STEPS="${SIGMA_STEPS:-50}"
DECOMP_CHUNK_SIZE="${DECOMP_CHUNK_SIZE:-4096}"
TRAIN_STEPS="${TRAIN_STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PRED_ROOT_EXPLICIT=0
if [[ -n "${PRED_ROOT:-}" ]]; then
  PRED_ROOT_EXPLICIT=1
fi
PRED_ROOT="${PRED_ROOT:-${OUTPUT_ROOT}/train_inverse_${TARGET_NAME}/predictions}"
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
  process      Optimize OLAT cross/parallel images into material maps.
  train        Run inverse and/or forward training smoke jobs.
  evaluate     Evaluate predictions against ICTPolarReal or Objaverse-style samples.
  all          setup -> check-env -> check-data -> process -> train -> evaluate.

Options:
  --data-root PATH          Dataset root. Default: ${DATA_ROOT}
  --output-root PATH        Output root. Default: ${OUTPUT_ROOT}
  --material-root PATH      Processed material map root. Default: ${MATERIAL_ROOT}
  --env-name NAME           Conda/micromamba env name. Default: ${ENV_NAME}
  --train-stage STAGE       inverse, forward, or both. Default: ${TRAIN_STAGE}
  --input NAME              Inverse input name. Default: ${INPUT_NAME}
  --target NAME             Training target image stem. Default: ${TARGET_NAME}
  --input-mode MODE         image, polarization, or gbuffer. Default: ${INPUT_MODE}
  --target-mode MODE        image, polarization, or gbuffer. Default: ${TARGET_MODE}
  --forward-input NAME      Forward input name. Default: ${FORWARD_INPUT}
  --forward-input-mode MODE image, polarization, or gbuffer. Default: ${FORWARD_INPUT_MODE}
  --forward-target NAME     Forward training target. Default: ${FORWARD_TARGET}
  --forward-target-mode MODE image, polarization, or gbuffer. Default: ${FORWARD_TARGET_MODE}
  --max-lights N            Number of OLAT lights to process. Default: ${MAX_LIGHTS}
  --min-lights N            Minimum available pairs required for decomposition. Default: ${MIN_DECOMP_LIGHTS}
  --light-start N           First OLAT light id to process. Default: ${LIGHT_START}
  --light-root PATH         Optional LSX calibration folder for light positions.
  --frame-layout LAYOUT     auto, raw, or normalized. Default: ${FRAME_LAYOUT}
  --backend auto|cpu|torch  Processing backend. Default: ${BACKEND}
  --device DEVICE           Torch device for processing/training. Default: ${DEVICE}
  --decomp-noise FLOAT      Decomposition radiance threshold. Default: ${DECOMP_NOISE}
  --normal-steps N          PyTorch normal optimization steps. Default: ${NORMAL_STEPS}
  --sigma-steps N           PyTorch roughness optimization steps. Default: ${SIGMA_STEPS}
  --decomp-chunk-size N     Foreground pixels per optimizer chunk. Default: ${DECOMP_CHUNK_SIZE}
  --train-steps N           Training smoke-test steps. Default: ${TRAIN_STEPS}
  --batch-size N            Training batch size. Default: ${BATCH_SIZE}
  --pred-root PATH          Prediction root for training/evaluation. Default: ${PRED_ROOT}
  --eval-mode MODE          ictpolarreal or objaverse. Default: ${EVAL_MODE}
  --eval-task TASK          decomposition or relighting. Default: ${EVAL_TASK}
  --eval-manifest PATH      Optional Objaverse/ICTPolarReal evaluation manifest.
  --torch-variant VARIANT   auto, cpu, cu121, cu124, cu126, cu128, or pypi. Default: ${TORCH_VARIANT}
  --download-sample         Download one complete sample camera. The all command does this automatically.
  --skip-setup              For all: use the current environment.
  --skip-process            For all: skip material preprocessing.
  --skip-train              For all: skip training.

Examples:
  bash run.sh all
  bash run.sh check-data
  bash run.sh process --backend torch --device cuda
  bash run.sh train --train-stage inverse --input-mode polarization --target albedo
  bash run.sh train --train-stage forward --forward-input-mode gbuffer --forward-target static
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --data-root) DATA_ROOT="$2"; shift 2 ;;
      --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
      --material-root) MATERIAL_ROOT="$2"; MATERIAL_ROOT_EXPLICIT=1; shift 2 ;;
      --env-name) ENV_NAME="$2"; shift 2 ;;
      --train-stage) TRAIN_STAGE="$2"; shift 2 ;;
      --input) INPUT_NAME="$2"; shift 2 ;;
      --target) TARGET_NAME="$2"; shift 2 ;;
      --input-mode) INPUT_MODE="$2"; shift 2 ;;
      --target-mode) TARGET_MODE="$2"; shift 2 ;;
      --forward-input) FORWARD_INPUT="$2"; shift 2 ;;
      --forward-input-mode) FORWARD_INPUT_MODE="$2"; shift 2 ;;
      --forward-target) FORWARD_TARGET="$2"; shift 2 ;;
      --forward-target-mode) FORWARD_TARGET_MODE="$2"; shift 2 ;;
      --max-lights) MAX_LIGHTS="$2"; shift 2 ;;
      --min-lights) MIN_DECOMP_LIGHTS="$2"; shift 2 ;;
      --light-start) LIGHT_START="$2"; shift 2 ;;
      --light-root) LIGHT_ROOT="$2"; shift 2 ;;
      --frame-layout) FRAME_LAYOUT="$2"; shift 2 ;;
      --backend) BACKEND="$2"; shift 2 ;;
      --device) DEVICE="$2"; shift 2 ;;
      --decomp-noise) DECOMP_NOISE="$2"; shift 2 ;;
      --normal-steps) NORMAL_STEPS="$2"; shift 2 ;;
      --sigma-steps) SIGMA_STEPS="$2"; shift 2 ;;
      --decomp-chunk-size) DECOMP_CHUNK_SIZE="$2"; shift 2 ;;
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
    PRED_ROOT="${OUTPUT_ROOT}/train_inverse_${TARGET_NAME}/predictions"
  fi
  if [[ "${MATERIAL_ROOT_EXPLICIT}" != "1" ]]; then
    MATERIAL_ROOT="${OUTPUT_ROOT}/material_acquisition"
  fi
  if (( MAX_LIGHTS > MIN_DECOMP_LIGHTS )); then
    REQUIRED_DECOMP_LIGHTS="${MAX_LIGHTS}"
  else
    REQUIRED_DECOMP_LIGHTS="${MIN_DECOMP_LIGHTS}"
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
  echo "[data] Downloading one complete OLAT camera for material fitting:"
  echo "       ${SAMPLE_URL}"
  if ! python - "${SAMPLE_URL}" "${DATA_ROOT}" "${TARGET_NAME}" "${MAX_LIGHTS}" "${MIN_DECOMP_LIGHTS}" <<'PY'; then
from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

IMAGE_EXTS = (".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def fail(message: str) -> None:
    print(f"[data] {message}")
    raise SystemExit(1)


def list_drive_folder(url: str, output: Path):
    try:
        import gdown
    except Exception as exc:
        fail(f"gdown is required to list the Drive folder: {exc}")

    kwargs = {"url": url, "output": str(output / ".drive-list"), "quiet": True, "skip_download": True}
    signature = inspect.signature(gdown.download_folder)
    if "remaining_ok" in signature.parameters:
        kwargs["remaining_ok"] = True
    if "resume" in signature.parameters:
        kwargs["resume"] = True
    try:
        files = gdown.download_folder(**kwargs)
    except Exception as exc:
        fail(f"gdown folder listing failed: {str(exc).strip().splitlines()[0]}")
    if not files:
        fail("Drive folder listing returned no files.")
    return files


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    return [values[round(index * last / (count - 1))] for index in range(count)]


def find_sample_files(files, target_name: str, requested_lights: int, minimum_lights: int) -> list:
    by_path = {item.path: item for item in files}
    camera_dirs = sorted({"/".join(path.split("/")[:2]) for path in by_path if "/cam" in path and len(path.split("/")) >= 3})
    for camera_dir in camera_dirs:
        static = first_existing(by_path, camera_dir, "static")
        mask = first_existing(by_path, camera_dir, "mask")
        target = first_existing(by_path, camera_dir, target_name)
        cross = light_map(by_path, camera_dir, "cross")
        parallel = light_map(by_path, camera_dir, "parallel")
        paired_lights = sorted(set(cross) & set(parallel))
        raw_layout = len(paired_lights) >= 348 or any(light >= 346 for light in paired_lights)
        valid_lights = [light for light in paired_lights if 2 <= light <= 347] if raw_layout else paired_lights
        if not (static and mask and target and len(valid_lights) >= minimum_lights):
            continue
        valid_lights = evenly_spaced(valid_lights, requested_lights)

        selected = [static, mask, target]
        for stem in ["normal", "normal_w2c", "specular", "sigma", "static_cross", "static_parallel"]:
            item = first_existing(by_path, camera_dir, stem)
            if item is not None:
                selected.append(item)
        selected.extend(cross[light] for light in valid_lights)
        selected.extend(parallel[light] for light in valid_lights)
        return list({item.path: item for item in selected}.values())
    fail(
        f"Could not find a camera with static, mask, {target_name}, "
        f"and at least {minimum_lights} valid OLAT pairs."
    )


def first_existing(by_path: dict, camera_dir: str, stem: str):
    for ext in IMAGE_EXTS:
        item = by_path.get(f"{camera_dir}/{stem}{ext}")
        if item:
            return item
    return None


def light_map(by_path: dict, camera_dir: str, kind: str) -> dict[int, object]:
    prefix = f"{camera_dir}/{kind}/"
    out = {}
    for path, item in by_path.items():
        if not path.startswith(prefix):
            continue
        name = Path(path).stem
        if name.isdigit() and Path(path).suffix.lower() in IMAGE_EXTS:
            out[int(name)] = item
    return out


def download_file(file_id: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    params = urllib.parse.urlencode({"id": file_id, "export": "download", "confirm": "t"})
    url = f"https://drive.usercontent.google.com/download?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type.lower():
            fail(f"Drive returned an HTML page instead of data for {dst.name}.")
        with tempfile.NamedTemporaryFile(delete=False, dir=str(dst.parent), suffix=".part") as tmp:
            tmp_path = Path(tmp.name)
            shutil.copyfileobj(response, tmp)
    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        fail(f"Downloaded empty file for {dst.name}.")
    tmp_path.replace(dst)
    return True


drive_url, output_root, target, requested, minimum = sys.argv[1:6]
root = Path(output_root)
files = list_drive_folder(drive_url, root)
required_files = find_sample_files(files, target, max(int(requested), int(minimum)), int(minimum))
downloaded = 0
print(f"[data] sample files required: {len(required_files)}")
for index, item in enumerate(required_files, start=1):
    downloaded += int(download_file(item.id, root / item.path))
    if index % 25 == 0 or index == len(required_files):
        print(f"[data] prepared {index}/{len(required_files)} files")
print(f"[data] Complete sample camera ready ({downloaded} new files).")
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
  if python -m ictpolarreal.data.check --data-root "${DATA_ROOT}" --min-lights "${REQUIRED_DECOMP_LIGHTS}"; then
    return 0
  fi
  if [[ "${DOWNLOAD_SAMPLE}" == "1" ]]; then
    if ! download_sample_if_requested; then
      python -m ictpolarreal.data.check --data-root "${DATA_ROOT}" --min-lights "${REQUIRED_DECOMP_LIGHTS}" || true
      return 3
    fi
    python -m ictpolarreal.data.check --data-root "${DATA_ROOT}" --min-lights "${REQUIRED_DECOMP_LIGHTS}"
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
    python -m ictpolarreal.data.check --data-root "${DATA_ROOT}" --min-lights "${REQUIRED_DECOMP_LIGHTS}" || true
    return 3
  fi
  python -m ictpolarreal.data.check --data-root "${DATA_ROOT}" --min-lights "${REQUIRED_DECOMP_LIGHTS}"
}

process_materials() {
  cd "${REPO_ROOT}"
  activate_env || true
  local light_root_args=()
  if [[ -n "${LIGHT_ROOT}" ]]; then
    light_root_args=(--light-root "${LIGHT_ROOT}")
  fi
  python -m ictpolarreal.processing.prepare_materials \
    --data-root "${DATA_ROOT}" \
    --out-root "${MATERIAL_ROOT}" \
    --max-lights "${MAX_LIGHTS}" \
    --light-start "${LIGHT_START}" \
    --frame-layout "${FRAME_LAYOUT}" \
    "${light_root_args[@]}" \
    --backend "${BACKEND}" \
    --device "${DEVICE}" \
    --noise "${DECOMP_NOISE}" \
    --normal-steps "${NORMAL_STEPS}" \
    --sigma-steps "${SIGMA_STEPS}" \
    --chunk-size "${DECOMP_CHUNK_SIZE}"
}

train_inverse() {
  cd "${REPO_ROOT}"
  activate_env || true
  if [[ "${TARGET_MODE}" == "image" ]]; then
    python -m ictpolarreal.data.check \
      --data-root "${DATA_ROOT}" \
      --min-lights 1 \
      --require-target "${TARGET_NAME}"
  fi
  python -m ictpolarreal.train.inverse \
    --data-root "${DATA_ROOT}" \
    --out-dir "${OUTPUT_ROOT}/train_inverse_${TARGET_NAME}" \
    --material-root "${MATERIAL_ROOT}" \
    --input "${INPUT_NAME}" \
    --target "${TARGET_NAME}" \
    --input-mode "${INPUT_MODE}" \
    --target-mode "${TARGET_MODE}" \
    --max-steps "${TRAIN_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --pred-dir "${PRED_ROOT}"
}

train_forward() {
  cd "${REPO_ROOT}"
  activate_env || true
  python -m ictpolarreal.data.check \
    --data-root "${DATA_ROOT}" \
    --min-lights 1 \
    --require-target "${FORWARD_TARGET}"
  python -m ictpolarreal.train.forward \
    --data-root "${DATA_ROOT}" \
    --out-dir "${OUTPUT_ROOT}/train_forward_${FORWARD_TARGET}" \
    --material-root "${MATERIAL_ROOT}" \
    --input "${FORWARD_INPUT}" \
    --target "${FORWARD_TARGET}" \
    --input-mode "${FORWARD_INPUT_MODE}" \
    --target-mode "${FORWARD_TARGET_MODE}" \
    --max-steps "${TRAIN_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --pred-dir "${OUTPUT_ROOT}/train_forward_${FORWARD_TARGET}/predictions"
}

train_baseline() {
  case "${TRAIN_STAGE}" in
    inverse) train_inverse ;;
    forward) train_forward ;;
    both) train_inverse; train_forward ;;
    *) echo "Unknown --train-stage: ${TRAIN_STAGE}"; exit 2 ;;
  esac
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
