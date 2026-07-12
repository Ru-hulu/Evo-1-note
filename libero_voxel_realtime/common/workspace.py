from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOXEL_SIZE = 0.01
PRECOMPUTED_1CM_PATH = PROJECT_ROOT / "mounted_panda_collision_voxels_010.npz"
DEFAULT_BASE_POS = np.asarray([-0.66, 0.0, 0.912], dtype=np.float32)
RAW_EE_WORKSPACE_BOUNDS = np.asarray(
    [
        [0.335313, 0.837761],
        [-0.286172, 0.394537],
        [-0.002086, 0.420008],
    ],
    dtype=np.float32,
)
DEFAULT_WORKSPACE_BOUNDS = np.asarray(
    [
        [0.33, 0.84],
        [-0.29, 0.40],
        [-0.01, 0.43],
    ],
    dtype=np.float32,
)


def quat_xyzw_to_matrix(quat: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world_to_base(points_world: np.ndarray, base_pos: Sequence[float], base_ori: Optional[Sequence[float]] = None) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    shifted = points - np.asarray(base_pos, dtype=np.float64)
    if base_ori is None:
        return shifted.astype(np.float32)
    rot = quat_xyzw_to_matrix(base_ori)
    return (shifted @ rot).astype(np.float32)


def bounds_to_grid_edges(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(bounds, dtype=np.float64)
    lower_scaled = bounds[:, 0] / VOXEL_SIZE
    upper_scaled = bounds[:, 1] / VOXEL_SIZE

    lower_round = np.round(lower_scaled)
    upper_round = np.round(upper_scaled)
    lower = np.where(
        np.isclose(lower_scaled, lower_round, rtol=0.0, atol=1e-5),
        lower_round,
        np.floor(lower_scaled),
    ).astype(np.int32)
    upper = np.where(
        np.isclose(upper_scaled, upper_round, rtol=0.0, atol=1e-5),
        upper_round,
        np.ceil(upper_scaled),
    ).astype(np.int32)
    return lower, upper


def workspace_bounds_from_args(bounds: Optional[Sequence[float]]) -> np.ndarray:
    if bounds is None:
        return DEFAULT_WORKSPACE_BOUNDS.copy()
    if len(bounds) != 6:
        raise ValueError("bounds must be x_min x_max y_min y_max z_min z_max")
    arr = np.asarray(
        [[bounds[0], bounds[1]], [bounds[2], bounds[3]], [bounds[4], bounds[5]]],
        dtype=np.float32,
    )
    lower_edge, upper_edge = bounds_to_grid_edges(arr)
    return np.stack([lower_edge, upper_edge], axis=1).astype(np.float32) * VOXEL_SIZE


def crop_base_points_to_workspace(points_base: np.ndarray, link_ids: np.ndarray, bounds: np.ndarray) -> dict:
    # bounds 先对齐到 1cm voxel grid，后续 occupancy 的每个 index 都对应一个固定 1cm 空间。
    lower_edge, upper_edge = bounds_to_grid_edges(bounds)
    bounds = np.stack([lower_edge, upper_edge], axis=1).astype(np.float32) * VOXEL_SIZE
    points_base = np.asarray(points_base, dtype=np.float32)
    link_ids = np.asarray(link_ids, dtype=np.int16)
    shape = (upper_edge - lower_edge).astype(np.int32)
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    occupancy = np.zeros(tuple(int(v) for v in shape), dtype=np.bool_) # 创建一个规定大小的三维数据，标记哪个voxel有被占据
    link_grid = np.full(tuple(int(v) for v in shape), -1, dtype=np.int16) # 创建一个规定大小的三维数据，标记当前voxel隶属于哪个关节

    occupied_count = 0
    for point, link_id in zip(points_base, link_ids):
        # 只保留落在 workspace box 内部的 base-frame 点。
        if np.any(point < lower) or np.any(point > upper):
            continue

        # 将 base-frame 连续坐标离散化成 workspace 内部的 voxel index。
        indices = np.floor((point - lower) / VOXEL_SIZE).astype(np.int32)
        indices = np.minimum(np.maximum(indices, 0), shape - 1)
        x, y, z = (int(v) for v in indices)

        # 多个点落入同一个 voxel 时，只保留第一次写入，相当于在这里完成去重。
        if occupancy[x, y, z]:
            continue
        occupancy[x, y, z] = True
        link_grid[x, y, z] = int(link_id)
        occupied_count += 1

    if occupied_count == 0:
        raise RuntimeError(
            "No robot voxels fell inside the workspace bounds. "
            f"bounds={bounds.tolist()}, input_points={len(points_base)}"
        )

    occupied_indices = np.argwhere(occupancy).astype(np.int32)
    # 把离散 index 转回 voxel 中心点，方便后续可视化或保存 point representation。
    centers = (lower + (occupied_indices.astype(np.float32) + 0.5) * VOXEL_SIZE).astype(np.float32)

    return {
        "points_base": centers,      # 被占据 voxel 的中心点坐标，坐标系为 base frame
        "occupancy": occupancy,      # 固定大小的三维 bool grid，True 表示该 voxel 被机器人占据
        "link_grid": link_grid,      # 固定大小的三维 int grid，-1 表示空，其他值表示对应 link/geom id
        "grid_shape_xyz": shape,     # occupancy/link_grid 的 xyz 维度大小
    }
