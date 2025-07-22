#!/usr/bin/env bash
set -euv

CWD=$(dirname "$0")
DETREX_ROOT="$(dirname "$(dirname "$CWD")")/3rdparty/detrex"

CONFIG=$1
CHECKPOINT=$2

python "$DETREX_ROOT/tools/train_net.py" \
    --config-file "$CONFIG" \
    --eval-only \
    "${@:4}" \
    train.init_checkpoint="$CHECKPOINT"
