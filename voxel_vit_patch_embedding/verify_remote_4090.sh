#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <ssh-target> [remote-dir] [python-bin]"
  echo "example: $0 user@host /tmp/voxel_vit_patch_embedding python3"
  exit 2
fi

SSH_TARGET="$1"
REMOTE_DIR="${2:-/tmp/voxel_vit_patch_embedding}"
PYTHON_BIN="${3:-python3}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rsync -az --delete \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  "${LOCAL_DIR}/" "${SSH_TARGET}:${REMOTE_DIR}/"

ssh "${SSH_TARGET}" "cd '${REMOTE_DIR}' && ${PYTHON_BIN} -c 'import torch; print(\"torch\", torch.__version__); print(\"cuda\", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no cuda\")' && ${PYTHON_BIN} smoke_test.py"
