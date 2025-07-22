#!/usr/bin/env bash
set -eu

CWD=$(dirname "$0")
DETREX_ROOT="$(dirname "$(dirname "$CWD")")/3rdparty/detrex"

CONFIG=$1
GPUS=${GPUS:-4}
PORT=${PORT:-25001}

python "$DETREX_ROOT/tools/train_net.py" \
    --config-file "$CONFIG" \
    --num-gpus "$GPUS" \
    --dist-url "tcp://127.0.0.1:$PORT" \
    "${@:2}"
