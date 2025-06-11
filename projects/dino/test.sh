#!/usr/bin/env bash
set -uv

CWD=$(dirname "$0")
ROOT=$(dirname "$(dirname "$CWD")")

CONFIG=$1
CHECKPOINT=$2
# use different seed to avoid bias.
SEED=${3:-3202}

python "$ROOT/tools/train_net.py" \
    --config-file "$CONFIG" \
    --eval-only \
    "${@:4}" \
    train.init_checkpoint="$CHECKPOINT" train.seed="$SEED"
