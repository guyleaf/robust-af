#!/usr/bin/env bash
set -eu

CWD=$(dirname "$0")

CONFIG=$1
CHECKPOINT=$2

python "$CWD/base/analyze_model.py" \
    --config-file "$CONFIG" \
    --tasks flop parameter --num-inputs 100 \
    train.init_checkpoint="$CHECKPOINT"
