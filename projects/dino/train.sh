#!/usr/bin/env bash
set -eu

CWD=$(dirname "$0")

CONFIG=$1
GPUS=${GPUS:-4}
PORT=${PORT:-25001}

python "$CWD/train_net.py" \
    --config-file "$CONFIG" \
    --num-gpus "$GPUS" \
    --dist-url "tcp://127.0.0.1:$PORT" \
    "${@:2}"
