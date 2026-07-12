from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.workspace import (
    DEFAULT_BASE_POS,
    DEFAULT_WORKSPACE_BOUNDS,
    PRECOMPUTED_1CM_PATH,
    VOXEL_SIZE,
    bounds_to_grid_edges,
    crop_base_points_to_workspace,
    workspace_bounds_from_args,
    world_to_base,
)
try:
    from .libero_voxel import (
        DEFAULT_SUITE,
        DEFAULT_TASK_ID,
        close_runtime,
        prepare_runtime,
        precompute,
        runtime_states_to_voxels,
    )
except ImportError:
    from libero_voxel import (
        DEFAULT_SUITE,
        DEFAULT_TASK_ID,
        close_runtime,
        prepare_runtime,
        precompute,
        runtime_states_to_voxels,
    )


def prepare_bounded_runtime(
    precomputed_path: Path = PRECOMPUTED_1CM_PATH,
    build_cache: bool = False,
    task_suite: Optional[str] = None,
    task_id: Optional[int] = None,
):
    precomputed_path = Path(precomputed_path)
    if not precomputed_path.exists():
        if not build_cache:
            raise FileNotFoundError(
                f"missing 1cm robot voxel cache: {precomputed_path}. "
                "Run `python sim/bounded_robot_voxel.py precompute` first, or pass --build-cache."
            )
        precompute(
            out_path=precomputed_path,
            voxel_size=VOXEL_SIZE,
            task_suite=DEFAULT_SUITE if task_suite is None else task_suite,
            task_id=DEFAULT_TASK_ID if task_id is None else int(task_id),
        )

    runtime = prepare_runtime(precomputed_path, task_suite=task_suite, task_id=task_id)
    pre = runtime[1]
    source_voxel_size = float(np.asarray(pre["voxel_size"]))
    if abs(source_voxel_size - VOXEL_SIZE) > 1e-6:
        close_runtime(runtime)
        raise ValueError(
            f"expected a fixed 1cm cache ({VOXEL_SIZE}), got voxel_size={source_voxel_size} from {precomputed_path}"
        )
    return runtime


def runtime_states_to_bounded_voxels(
    runtime,
    states: Optional[Mapping[str, float] | Sequence[float]] = None,
    workspace_bounds: np.ndarray = DEFAULT_WORKSPACE_BOUNDS,
    base_pos: Sequence[float] = DEFAULT_BASE_POS,
    base_ori: Optional[Sequence[float]] = None,
) -> dict:
    lower_edge, upper_edge = bounds_to_grid_edges(workspace_bounds)
    workspace_bounds = np.stack([lower_edge, upper_edge], axis=1).astype(np.float32) * VOXEL_SIZE

    full = runtime_states_to_voxels(runtime, states)
    # {
    #     "indices": ...,      # voxel index
    #     "points": ...,       # voxel center 坐标
    #     "link_ids": ...,     # 每个 voxel 属于哪个 link / geom
    #     ...
    # }
    points_world = np.asarray(full["points"], dtype=np.float32)
    points_base = world_to_base(points_world, base_pos=base_pos, base_ori=base_ori)
    cropped = crop_base_points_to_workspace(points_base, full["link_ids"], workspace_bounds)

    return {
        "voxel_size": np.asarray(VOXEL_SIZE, dtype=np.float32),
        "axis_order": np.asarray("xyz"),
        "base_pos": np.asarray(base_pos, dtype=np.float32),
        "base_ori": np.asarray([] if base_ori is None else base_ori, dtype=np.float32),
        "workspace_min_xyz": workspace_bounds[:, 0].astype(np.float32),
        "workspace_max_xyz": workspace_bounds[:, 1].astype(np.float32),
        "grid_shape_xyz": cropped["grid_shape_xyz"].astype(np.int32),
        "points_base": cropped["points_base"],
        "occupancy": cropped["occupancy"],
        "link_grid": cropped["link_grid"],
        "source_world_voxel_count": np.asarray(len(points_world), dtype=np.int32),
        "source_joint_names": full["joint_names"],
        "source_geom_names": full["geom_names"],
    }

def parse_states(args) -> Optional[Mapping[str, float] | Sequence[float]]:
    if args.states_json:
        return json.loads(args.states_json)
    return args.states


def save_bounded_voxels(path: str | Path, voxels: dict) -> None:
    np.savez_compressed(path, **voxels)


def demo(args) -> None:
    bounds = workspace_bounds_from_args(args.bounds)
    states = parse_states(args)
    runtime = prepare_bounded_runtime(
        precomputed_path=Path(args.npz),
        build_cache=args.build_cache,
        task_suite=args.task_suite,
        task_id=args.task_id,
    )
    try:
        t0 = time.perf_counter()
        voxels = runtime_states_to_bounded_voxels(
            runtime,
            states=states,
            workspace_bounds=bounds,
            base_pos=args.base_pos,
            base_ori=args.base_ori,
        )
        dt = (time.perf_counter() - t0) * 1000.0

        occupied = int(voxels["occupancy"].sum())
        shape = voxels["grid_shape_xyz"].astype(int)
        capacity = int(np.prod(shape))
        density = occupied / capacity if capacity else 0.0

        print("voxel_size", float(voxels["voxel_size"]))
        print("axis_order", str(voxels["axis_order"]))
        print("base_pos", voxels["base_pos"].round(6).tolist())
        print("workspace_min_xyz", voxels["workspace_min_xyz"].round(6).tolist())
        print("workspace_max_xyz", voxels["workspace_max_xyz"].round(6).tolist())
        print("grid_shape_xyz", shape.tolist())
        print("source_world_voxel_count", int(voxels["source_world_voxel_count"]))
        print("occupied_after_crop", occupied)
        print("grid_capacity", capacity)
        print("occupancy_density", round(density, 6))
        print("state_to_bounded_voxel_ms", round(dt, 3))

        if args.save:
            save_bounded_voxels(args.save, voxels)
            print(f"saved {args.save}")
    finally:
        close_runtime(runtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("precompute")
    p0.add_argument("--out", default=str(PRECOMPUTED_1CM_PATH))
    p0.add_argument("--task-suite", default=DEFAULT_SUITE)
    p0.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)

    p1 = sub.add_parser("demo")
    p1.add_argument("--npz", default=str(PRECOMPUTED_1CM_PATH))
    p1.add_argument("--build-cache", action="store_true")
    p1.add_argument("--task-suite", default=None)
    p1.add_argument("--task-id", type=int, default=None)
    p1.add_argument("--bounds", nargs=6, type=float, default=None, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"))
    p1.add_argument("--base-pos", nargs=3, type=float, default=DEFAULT_BASE_POS.tolist(), metavar=("X", "Y", "Z"))
    p1.add_argument("--base-ori", nargs=4, type=float, default=None, metavar=("QX", "QY", "QZ", "QW"))
    p1.add_argument("--states", nargs="*", type=float, default=None)
    p1.add_argument("--states-json", default=None)
    p1.add_argument("--save", default=None)

    args = parser.parse_args()
    if args.cmd == "precompute":
        precompute(
            out_path=Path(args.out),
            voxel_size=VOXEL_SIZE,
            task_suite=args.task_suite,
            task_id=args.task_id,
        )
    elif args.cmd == "demo":
        demo(args)


if __name__ == "__main__":
    main()
