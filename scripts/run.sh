#!/usr/bin/env bash

# Usage:
#   scripts/run.sh --ckpt checkpoints/model.pt --enable_orbit

set -euo pipefail
cd "$(dirname "$0")/.."
exec python maniflow/sim/run.py "$@"
