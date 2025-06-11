#!/usr/bin/env bash

CWD=$(dirname "$0")
ROOT=$(dirname "$(dirname "$CWD")")

CONFIG=$1
GPUS=${GPUS:-4}

python "$ROOT/tools/train_net.py" \
    --config-file "$CONFIG" \
    --num-gpus "$GPUS" \
    "${@:2}"
