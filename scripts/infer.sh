#!/usr/bin/env bash
# Headless evaluation of a trained checkpoint

# Usage:
#   scripts/infer.sh --ckpt checkpoints/model.pt --eval_eps 100 --num_obstacles 3

set -euo pipefail
cd "$(dirname "$0")/.."
exec python maniflow/inference/infer.py "$@"
