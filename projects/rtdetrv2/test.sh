#!/usr/bin/env bash
set -euo pipefail

CWD=$(dirname "$0")

CONFIG=$1
CHECKPOINT=$2
GPUS=${GPUS:-4}
SEED=${SEED:-2025}

mkdir -p logs
torchrun \
    --standalone \
    --nproc-per-node "$GPUS" \
    "$CWD/tools/base/train.py" \
    --config "$CONFIG" \
    --resume "$CHECKPOINT" \
    --print-method rich \
    --seed "$SEED" \
    --test-only \
    "${@:3}" 2>&1 | tee "logs/log_test_$(date +%Y%m%d%H%M%S%N).txt"
