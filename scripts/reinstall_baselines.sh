#!/usr/bin/env bash
set -e

cwd=$(dirname "$0")
root=$(dirname "$cwd")

pip install -e "${root}[baseline]"
