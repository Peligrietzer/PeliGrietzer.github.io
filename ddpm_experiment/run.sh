#!/bin/bash
# =============================================================================
# Diffusion Manifold Projection Experiment
# =============================================================================
#
# This script runs the full experiment end-to-end.
#
# Requirements:
#   - Python 3.8+
#   - A CUDA-capable GPU (A100/V100/T4 recommended)
#   - ~10 GB disk space for saved images
#
# Usage:
#   bash run.sh              # full experiment (~4-6 hours on A100)
#   bash run.sh --quick      # smoke test (~10 min on GPU, ~1 hr on CPU)
#   bash run.sh --teacher-only  # train teacher only
#
# On Google Colab, you can also run the notebook version directly.
# =============================================================================

set -e

echo "============================================="
echo "DDPM Manifold Projection Experiment"
echo "============================================="

# --- Install dependencies ---
echo ""
echo "Installing dependencies..."
pip install --break-system-packages -q \
    torch torchvision \
    numpy matplotlib tqdm \
    clean-fid scipy \
    2>/dev/null || \
pip install -q \
    torch torchvision \
    numpy matplotlib tqdm \
    clean-fid scipy

# --- Check GPU ---
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected. Training will be very slow.')
    print('Consider using Google Colab (Runtime > Change runtime type > GPU)')
"

# --- Run experiment ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ""
echo "Starting experiment..."
python3 "${SCRIPT_DIR}/ddpm_experiment.py" "$@"

echo ""
echo "============================================="
echo "Done! Results saved in ./results/"
echo "============================================="
