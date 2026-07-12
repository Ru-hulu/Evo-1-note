from __future__ import annotations

import argparse
import json
import sys
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
    quat_xyzw_to_matrix,
    workspace_bounds_from_args,
)


REQUIRED_KINEMATIC_KEYS = (
    "local_points",
    "geom_slices",
    "geom_names",
    "geom_body_ids",
    "geom_pos",
    "geom_quat",
    "body_parent_ids",
    "body_pos",
    "body_quat",
    "body_jntadr",
    "body_jntnum",
    "all_joint_names",
    "jnt_type",
    "jnt_axis",
    "jnt_pos",
    "jnt_qposadr",
    "joint_names",
    "arm_joint_names",
    "gripper_joint_names",
    "qpos_adrs",
    "init_qpos",
)


MJ_JNT_SLIDE = 2
MJ_JNT_HINGE = 3


def load_standalone_cache(path: str | Path = PRECOMPUTED_1CM_PATH) -> dict:
    data = np.load(path, allow_pickle=True)
    cache = {key: data[key] for key in data.files}
    missing = [key for key in REQUIRED_KINEMATIC_KEYS if key not in cache]
    if missing:
        raise KeyError(
            "The cache does not contain standalone kinematic metadata. "
            f"Missing keys: {missing}. Regenerate it with `sim/bounded_robot_voxel.py precompute`."
        )

    source_voxel_size = float(np.asarray(cache["voxel_size"]))
    if abs(source_voxel_size - VOXEL_SIZE) > 1e-6:
        raise ValueError(f"expected a fixed 1cm cache ({VOXEL_SIZE}), got voxel_size={source_voxel_size}")
    return cache


def quat_wxyz_to_matrix(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def axis_angle_to_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def compose_pose(parent_pos, parent_mat, local_pos, local_mat):
    pos = parent_pos + parent_mat @ np.asarray(local_pos, dtype=np.float64)
    mat = parent_mat @ np.asarray(local_mat, dtype=np.float64)
    return pos, mat


def _state_dict(states: Optional[Mapping[str, float]]) -> dict[str, float]:
    if states is None:
        return {}
    return {str(k): float(v) for k, v in states.items()}


def joint_values_for_state(cache: dict, states: Optional[Mapping[str, float]]) -> np.ndarray:
    all_joint_names = [str(x) for x in cache["all_joint_names"]]
    joint_name_to_id = {name: i for i, name in enumerate(all_joint_names)}
    robot_joint_names = [str(x) for x in cache["joint_names"]]
    init_qpos = np.asarray(cache["init_qpos"], dtype=np.float64)

    values = np.zeros(len(all_joint_names), dtype=np.float64)
    for name, init_value in zip(robot_joint_names, init_qpos):
        if name in joint_name_to_id:
            values[joint_name_to_id[name]] = float(init_value)

    state_by_name = _state_dict(states)
    for name, value in state_by_name.items():
        if name == "gripper_open":
            continue
        if name in joint_name_to_id:
            values[joint_name_to_id[name]] = float(value)

    if "gripper_open" in state_by_name:
        gripper_open = float(state_by_name["gripper_open"])
        gripper_names = [str(x) for x in cache["gripper_joint_names"]]
        if len(gripper_names) >= 1 and gripper_names[0] in joint_name_to_id:
            values[joint_name_to_id[gripper_names[0]]] = gripper_open
        if len(gripper_names) >= 2 and gripper_names[1] in joint_name_to_id:
            values[joint_name_to_id[gripper_names[1]]] = -gripper_open

    return values


def apply_joint_to_body_pose(body_pos, body_mat, joint_type, joint_pos, joint_axis, joint_value):
    if int(joint_type) == MJ_JNT_HINGE:
        rot = axis_angle_to_matrix(joint_axis, joint_value)
        local_pos = np.asarray(joint_pos, dtype=np.float64) - rot @ np.asarray(joint_pos, dtype=np.float64)
        return compose_pose(body_pos, body_mat, local_pos, rot)

    if int(joint_type) == MJ_JNT_SLIDE:
        local_pos = np.asarray(joint_axis, dtype=np.float64) * float(joint_value)
        return compose_pose(body_pos, body_mat, local_pos, np.eye(3, dtype=np.float64))

    return body_pos, body_mat


def body_poses_from_state(cache: dict, states: Optional[Mapping[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    body_parent_ids = np.asarray(cache["body_parent_ids"], dtype=np.int32)
    body_pos0 = np.asarray(cache["body_pos"], dtype=np.float64)
    body_quat0 = np.asarray(cache["body_quat"], dtype=np.float64)
    body_jntadr = np.asarray(cache["body_jntadr"], dtype=np.int32)
    body_jntnum = np.asarray(cache["body_jntnum"], dtype=np.int32)
    jnt_type = np.asarray(cache["jnt_type"], dtype=np.int32)
    jnt_pos = np.asarray(cache["jnt_pos"], dtype=np.float64)
    jnt_axis = np.asarray(cache["jnt_axis"], dtype=np.float64)
    joint_values = joint_values_for_state(cache, states)

    body_world_pos = np.zeros((len(body_parent_ids), 3), dtype=np.float64)
    body_world_mat = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], len(body_parent_ids), axis=0)

    for body_id in range(len(body_parent_ids)):
        parent_id = int(body_parent_ids[body_id])
        local_pos = body_pos0[body_id]
        local_mat = quat_wxyz_to_matrix(body_quat0[body_id])
        if parent_id >= 0:
            pos, mat = compose_pose(body_world_pos[parent_id], body_world_mat[parent_id], local_pos, local_mat)
        else:
            pos, mat = local_pos, local_mat

        joint_start = int(body_jntadr[body_id])
        joint_count = int(body_jntnum[body_id])
        for joint_id in range(joint_start, joint_start + joint_count):
            pos, mat = apply_joint_to_body_pose(
                pos,
                mat,
                jnt_type[joint_id],
                jnt_pos[joint_id],
                jnt_axis[joint_id],
                joint_values[joint_id],
            )

        body_world_pos[body_id] = pos
        body_world_mat[body_id] = mat

    return body_world_pos, body_world_mat


def states_to_base_points(
    cache: dict,
    states: Optional[Mapping[str, float]] = None,
    base_pos: Sequence[float] = DEFAULT_BASE_POS,
    base_ori: Optional[Sequence[float]] = None,
) -> dict:
    # 根据当前关节 state 计算每个 body 在世界坐标系下的位姿。
    body_world_pos, body_world_mat = body_poses_from_state(cache, states)
    base_pos_arr = np.asarray(base_pos, dtype=np.float64)
    base_rot = None if base_ori is None else quat_xyzw_to_matrix(base_ori)

    # local_points 是预先存好的 collision voxel 点，坐标系是各自 geom 的局部坐标系。
    local_points = np.asarray(cache["local_points"], dtype=np.float32)
    geom_slices = np.asarray(cache["geom_slices"], dtype=np.int32)
    geom_body_ids = np.asarray(cache["geom_body_ids"], dtype=np.int32)
    geom_pos = np.asarray(cache["geom_pos"], dtype=np.float64)
    geom_quat = np.asarray(cache["geom_quat"], dtype=np.float64)

    point_chunks = []
    link_id_chunks = []
    for link_id, (start, end) in enumerate(geom_slices):
        # 每个 geom 在 local_points 中占一个连续区间，geom_slices 记录这个区间。
        points = local_points[start:end]
        if len(points) == 0:
            continue
        body_id = int(geom_body_ids[link_id])

        # geom 自身还有一层相对 body 的局部位姿，需要和 body 的世界位姿合成。
        geom_world_pos, geom_world_mat = compose_pose(
            body_world_pos[body_id],
            body_world_mat[body_id],
            geom_pos[link_id],
            quat_wxyz_to_matrix(geom_quat[link_id]),
        )

        # 将 geom-local voxel 点直接变换到 base 坐标系。
        out_pos = geom_world_pos - base_pos_arr
        out_mat_t = geom_world_mat.T
        if base_rot is not None:
            out_pos = out_pos @ base_rot
            out_mat_t = out_mat_t @ base_rot

        transformed = points @ out_mat_t + out_pos # 将点云从 geom 坐标系下变换到 base 坐标系下
        point_chunks.append(transformed.astype(np.float32))  # 当前 geom 的 collision voxel 中心点在 base 坐标系下的位置
        link_id_chunks.append(np.full(len(transformed), link_id, dtype=np.int16))  # 每个点对应的 collision geom 编号

    raw_points = np.vstack(point_chunks) # collision voxel 中心点在base坐标系下的位置
    raw_link_ids = np.concatenate(link_id_chunks) # 每个点对应的 collision geom 编号

    return {
        "points": raw_points.astype(np.float32),
        "link_ids": raw_link_ids.astype(np.int16),
        "geom_names": cache["geom_names"],
        "joint_names": cache["joint_names"],
    }


def states_to_bounded_voxels(
    states: Optional[Mapping[str, float]] = None,
    cache_path: str | Path = PRECOMPUTED_1CM_PATH,
    workspace_bounds: np.ndarray = DEFAULT_WORKSPACE_BOUNDS,
    base_pos: Sequence[float] = DEFAULT_BASE_POS,
    base_ori: Optional[Sequence[float]] = None,
) -> dict:
    cache = load_standalone_cache(cache_path)
    lower_edge, upper_edge = bounds_to_grid_edges(workspace_bounds)
    workspace_bounds = np.stack([lower_edge, upper_edge], axis=1).astype(np.float32) * VOXEL_SIZE

    full = states_to_base_points(cache, states, base_pos=base_pos, base_ori=base_ori)
    cropped = crop_base_points_to_workspace(full["points"], full["link_ids"], workspace_bounds)

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
        "source_base_voxel_count": np.asarray(len(full["points"]), dtype=np.int32),
        "source_joint_names": full["joint_names"],
        "source_geom_names": full["geom_names"],
    }


def parse_states(args) -> Optional[Mapping[str, float]]:
    if not args.states_json:
        return None
    states = json.loads(args.states_json)
    if not isinstance(states, dict):
        raise ValueError("--states-json must be a JSON object, for example '{\"robot0_joint1\": 0.1}'")
    return {str(k): float(v) for k, v in states.items()}


def demo(args) -> None:
    bounds = workspace_bounds_from_args(args.bounds)
    voxels = states_to_bounded_voxels(
        states=parse_states(args),
        cache_path=args.npz,
        workspace_bounds=bounds,
        base_pos=args.base_pos,
        base_ori=args.base_ori,
    )

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
    print("source_base_voxel_count", int(voxels["source_base_voxel_count"]))
    print("occupied_after_crop", occupied)
    print("grid_capacity", capacity)
    print("occupancy_density", round(density, 6))

    if args.save:
        np.savez_compressed(args.save, **voxels)
        print(f"saved {args.save}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=str(PRECOMPUTED_1CM_PATH))
    parser.add_argument("--bounds", nargs=6, type=float, default=None, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"))
    parser.add_argument("--base-pos", nargs=3, type=float, default=DEFAULT_BASE_POS.tolist(), metavar=("X", "Y", "Z"))
    parser.add_argument("--base-ori", nargs=4, type=float, default=None, metavar=("QX", "QY", "QZ", "QW"))
    parser.add_argument("--states-json", default=None)
    parser.add_argument("--save", default=None)
    demo(parser.parse_args())


if __name__ == "__main__":
    main()
