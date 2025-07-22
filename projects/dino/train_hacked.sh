#!/usr/bin/env bash
set -eu

CWD=$(dirname "$0")

CONFIG=$1
GPUS=${GPUS:-4}

python "$CWD/train_net.py" \
    --config-file "$CONFIG" \
    --num-gpus "$GPUS" \
    "${@:2}"
