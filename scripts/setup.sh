#!/usr/bin/env bash
set -e

cwd=$(dirname "$0")
root=$(dirname "$cwd")

envName="robust_af"

# create conda env
conda env create -y -f "$root/robust-af.yml" -n "$envName"
echo -e "\033[0;32mInstallation Successfully!"
