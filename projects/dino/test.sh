#!/usr/bin/env bash
set -euv

CWD=$(dirname "$0")

CONFIG=$1
CHECKPOINT=$2

python "$CWD/train_net.py" \
    --config-file "$CONFIG" \
    --eval-only \
    "${@:4}" \
    train.init_checkpoint="$CHECKPOINT"
