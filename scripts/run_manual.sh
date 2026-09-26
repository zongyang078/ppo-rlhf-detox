#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python3 src/train_manual_ppo.py --config configs/config.yaml --run-name manual_ppo
