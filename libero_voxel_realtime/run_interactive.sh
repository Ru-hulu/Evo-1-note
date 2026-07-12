#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python sim/libero_voxel.py precompute
python sim/interactive_voxel_3d.py
