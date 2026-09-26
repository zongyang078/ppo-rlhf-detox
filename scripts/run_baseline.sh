#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python3 src/train_trl_baseline.py --config configs/config.yaml --run-name trl_baseline
