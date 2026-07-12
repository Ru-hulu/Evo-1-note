#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/root/miniconda3/envs/spatialvla/bin/python"
APP_DIR="/root/libero_voxel_realtime"
DISPLAY_VALUE="${DISPLAY:-:1}"

if ! vncserver -list 2>/dev/null | awk '$1 == "1" { found = 1 } END { exit(found ? 0 : 1) }'; then
    if [[ -x /root/start-vnc.sh ]]; then
        /root/start-vnc.sh
    else
        vncserver :1 -localhost yes -geometry 1600x900 -depth 24
    fi
fi

export DISPLAY="$DISPLAY_VALUE"
export QT_X11_NO_MITSHM=1
unset QT_PLUGIN_PATH

cd "$APP_DIR"

if [[ ! -f mounted_panda_collision_voxels_005.npz ]]; then
    "$PYTHON_BIN" sim/libero_voxel.py precompute
fi

exec "$PYTHON_BIN" sim/interactive_voxel_matplotlib.py "$@"
