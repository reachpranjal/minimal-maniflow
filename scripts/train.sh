#!/usr/bin/env bash

# Usage:
#   bash scripts/train.sh --zarr dataset/data.zarr --n_epochs 300 --batch 256

set -euo pipefail
cd "$(dirname "$0")/.."
exec python maniflow/training/train.py "$@"
