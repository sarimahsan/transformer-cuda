#!/usr/bin/env bash
# profile_ncu.sh: Convenience script to run Nsight Compute on Linux / Colab
set -e

KERNEL_REGEX=${1:-".*(layernorm|tiled_causal|add_bias|gelu|sgemm).*"}
TARGET=${2:-"pure_cuda"}

echo "========================================================================"
echo " Running Nsight Compute Microarchitectural Profiling"
echo " Target: $TARGET | Kernels: $KERNEL_REGEX"
echo "========================================================================"

mkdir -p results/nsight

python3 scripts/profile_ncu.py \
    --kernel_regex "$KERNEL_REGEX" \
    --target "$TARGET" \
    --out_dir results/nsight

echo "Done! NCU report generated under results/nsight/"
