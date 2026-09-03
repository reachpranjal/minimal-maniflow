#!/usr/bin/env bash

# Usage:
#   scripts/collect_data.sh --n_episodes 500 --out dataset/data.zarr

set -euo pipefail
cd "$(dirname "$0")/.."
exec python maniflow/collect_data.py "$@"
