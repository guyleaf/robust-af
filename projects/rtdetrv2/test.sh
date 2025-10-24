#!/usr/bin/env bash
set -eu

CWD=$(dirname "$0")

CONFIG=$1
CHECKPOINT=$2
GPUS=${GPUS:-4}
PORT=${PORT:-9909}
SEED=${SEED:-2025}

torchrun \
    --master-port "$PORT" \
    --nproc-per-node "$GPUS" \
    "$CWD/tools/train.py" \
    --config "$CONFIG" \
    --resume "$CHECKPOINT" \
    --print-method rich \
    --seed "$SEED" \
    --test-only \
    "${@:3}"
