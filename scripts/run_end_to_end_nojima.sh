#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${GAN_METHOD_B_PYTHON:-/Users/mu-sota/.venvs/gan-method-b/bin/python}"
program="${repo_root}/End-To-End/20260925 U-Net--Faster R-CNN/20260925EndToEnd_Seg204_Det163_nojima_weights.py"
det_weight="${repo_root}/Detection/20250208 Faster R-CNN/TrainingV163_pthfiles/train_30.pth"
seg_weight="${repo_root}/Segmentation/20250108 U-Net/TrainingV204_pthfiles/train_87.pth"
test_csv="${repo_root}/Dataset/20260925 CSV_Data/Detection/Original/test.csv"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python environment not found: ${python_bin}" >&2
  echo "Set GAN_METHOD_B_PYTHON to the Python executable for this project." >&2
  exit 1
fi

for required_file in "${program}" "${det_weight}" "${seg_weight}" "${test_csv}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

"${python_bin}" -c "import cv2, matplotlib, numpy, pandas, PIL, segmentation_models_pytorch, sklearn, torch, torchvision, tqdm"

echo "Preflight check: PASS"
echo "Python: ${python_bin}"
echo "Program: ${program}"
echo "Detection weight: ${det_weight}"
echo "Segmentation weight: ${seg_weight}"

if [[ "${1:-}" == "--check" ]]; then
  exit 0
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="/tmp/end_to_end_nojima_${timestamp}.log"
echo "Log: ${log_file}"

cd "${repo_root}"
if [[ "$(uname -s)" == "Darwin" ]] && command -v caffeinate >/dev/null 2>&1; then
  /usr/bin/time -p caffeinate -i "${python_bin}" -u "${program}" 2>&1 | tee "${log_file}"
else
  /usr/bin/time -p "${python_bin}" -u "${program}" 2>&1 | tee "${log_file}"
fi
