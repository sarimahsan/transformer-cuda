#!/usr/bin/env bash
# profile_nsys.sh: Convenience script to run Nsight Systems profiling on Linux / Colab
set -e

STEPS=${1:-5}
BATCH_SIZE=${2:-32}
SEQ_LEN=${3:-256}
D_MODEL=${4:-256}
NUM_LAYERS=${5:-6}
NUM_HEADS=${6:-8}

echo "========================================================================"
echo " Running Automated Nsight Systems Profiling Suite"
echo " Configuration: B=$BATCH_SIZE, T=$SEQ_LEN, C=$D_MODEL, L=$NUM_LAYERS, H=$NUM_HEADS"
echo " Steps: $STEPS"
echo "========================================================================"

mkdir -p results/nsight

python3 scripts/profile_nsys.py \
    --steps "$STEPS" \
    -b "$BATCH_SIZE" \
    -t "$SEQ_LEN" \
    -d "$D_MODEL" \
    -l "$NUM_LAYERS" \
    -h_heads "$NUM_HEADS" \
    --out_dir results/nsight

echo "Done! Reports generated under results/nsight/"
