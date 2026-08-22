#!/usr/bin/env bash
# Create conda env "euler" with PyTorch (CUDA) + PyG for lanl_experiments.
set -euo pipefail

ENV_NAME="${EULER_ENV_NAME:-euler}"
TORCH="${EULER_TORCH:-2.5.1}"
CUDA_TAG="${EULER_CUDA:-cu121}"   # cu121 works with recent NVIDIA drivers (4x A6000)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip_only=false
for arg in "$@"; do
  [[ "$arg" == "--pip-only" ]] && pip_only=true
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if [[ "$pip_only" == false ]]; then
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists; activating (use conda env remove -n $ENV_NAME to recreate)."
  else
    echo "Creating conda env '$ENV_NAME' from $ROOT/environment.yml ..."
    conda env create -f "$ROOT/environment.yml" -n "$ENV_NAME"
  fi
fi

conda activate "$ENV_NAME"

echo "Installing PyTorch ${TORCH} (${CUDA_TAG}) ..."
pip install --upgrade pip
pip install "torch==${TORCH}" torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "Installing PyTorch Geometric + extensions ..."
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster \
  -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA_TAG}.html"

echo "Verifying ..."
python - <<'PY'
import torch
import torch_geometric
import pandas, sklearn, joblib, tqdm
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("pyg", torch_geometric.__version__)
print("pandas", pandas.__version__, "sklearn", sklearn.__version__)
PY

echo ""
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "Run Euler:            cd Euler/lanl_experiments && python run.py -w 3 -r TGRU --lr 0.001"
