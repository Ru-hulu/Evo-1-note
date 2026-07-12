import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def get_default_dataset_root():
    try:
        from libero.libero.utils import get_libero_path

        return Path(get_libero_path("datasets"))
    except Exception:
        return Path("/root/libero_datasets")


def candidate_ee_keys(obs_group):
    keys = list(obs_group.keys())
    preferred = [
        "ee_states",
        "robot0_eef_pos",
        "robot0_eef_position",
        "eef_pos",
        "eef_position",
    ]
    out = [key for key in preferred if key in obs_group]
    out.extend(
        key for key in keys
        if key not in out and (
            key == "ee_states"
            or ("eef" in key.lower() and ("pos" in key.lower() or "position" in key.lower()))
        )
    )
    return out


def demo_groups(h5_file):
    if "data" not in h5_file:
        return []
    data = h5_file["data"]
    return [data[key] for key in sorted(data.keys()) if key.startswith("demo")]


def quat_xyzw_to_matrix(quat):
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world_to_base(xyz_world, base_pos, base_ori=None):
    xyz = np.asarray(xyz_world, dtype=np.float64)
    shifted = xyz - np.asarray(base_pos, dtype=np.float64)
    if base_ori is None:
        return shifted
    rot = quat_xyzw_to_matrix(base_ori)
    return shifted @ rot


def file_ee_range_from_obs(path, base_pose=None):
    total = 0
    mins = []
    maxs = []
    base_mins = []
    base_maxs = []
    used_keys = set()
    with h5py.File(path, "r") as f:
        for demo in demo_groups(f):
            if "obs" not in demo:
                continue
            obs = demo["obs"]
            keys = candidate_ee_keys(obs)
            if not keys:
                continue
            key = keys[0]
            arr = np.asarray(obs[key])
            if arr.ndim != 2 or arr.shape[1] < 3:
                continue
            xyz = arr[:, :3].astype(np.float64)
            mins.append(xyz.min(axis=0))
            maxs.append(xyz.max(axis=0))
            if base_pose is not None:
                base_xyz = world_to_base(
                    xyz,
                    base_pose["base_pos"],
                    base_pose.get("base_ori"),
                )
                base_mins.append(base_xyz.min(axis=0))
                base_maxs.append(base_xyz.max(axis=0))
            total += len(xyz)
            used_keys.add(key)

    if not mins:
        return None
    summary = {
        "path": str(path),
        "count": int(total),
        "min": np.min(np.stack(mins), axis=0),
        "max": np.max(np.stack(maxs), axis=0),
        "keys": sorted(used_keys),
    }
    if base_mins:
        summary["base_min"] = np.min(np.stack(base_mins), axis=0)
        summary["base_max"] = np.max(np.stack(base_maxs), axis=0)
        summary["base_pos"] = np.asarray(base_pose["base_pos"], dtype=np.float64)
        if base_pose.get("base_ori") is not None:
            summary["base_ori"] = np.asarray(base_pose["base_ori"], dtype=np.float64)
    return summary


def discover_hdf5_files(dataset_root):
    root = Path(dataset_root)
    return sorted(root.rglob("*.hdf5")) + sorted(root.rglob("*.h5"))


def print_range(label, mins, maxs):
    span = maxs - mins
    print(f"{label}_min {np.round(mins, 6).tolist()}")
    print(f"{label}_max {np.round(maxs, 6).tolist()}")
    print(f"{label}_span {np.round(span, 6).tolist()}")


def build_task_base_map(suites):
    from pathlib import Path

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    base_map = {}
    for suite in suites:
        bm = benchmark.get_benchmark_dict()[suite]()
        for task_id in range(bm.n_tasks):
            task = bm.get_task(task_id)
            demo_relpath = bm.get_task_demonstration(task_id)
            bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=32, camera_widths=32)
            env.reset()
            robot = env.robots[0]
            base_pose = {
                "suite": suite,
                "task_id": task_id,
                "task_name": task.name,
                "base_pos": np.asarray(robot.base_pos, dtype=np.float64),
                "base_ori": np.asarray(robot.base_ori, dtype=np.float64),
            }
            env.close()
            base_map[demo_relpath] = base_pose
            base_map[Path(demo_relpath).name] = base_pose
    return base_map


def resolve_base_pose(path, dataset_root, fixed_base_pos, fixed_base_ori, base_map):
    if fixed_base_pos is not None:
        return {
            "base_pos": np.asarray(fixed_base_pos, dtype=np.float64),
            "base_ori": None if fixed_base_ori is None else np.asarray(fixed_base_ori, dtype=np.float64),
        }
    if base_map is None:
        return None

    path = Path(path)
    try:
        rel = str(path.relative_to(dataset_root))
    except ValueError:
        rel = str(path)
    rel = rel.replace("\\", "/")
    return base_map.get(rel) or base_map.get(path.name)


def summarize_dataset_ee(dataset_root, fixed_base_pos=None, fixed_base_ori=None, base_map=None):
    files = discover_hdf5_files(dataset_root)
    print("dataset_root", str(dataset_root))
    print("hdf5_files", len(files))
    if not files:
        return None

    summaries = []
    missing = []
    missing_base = []
    for path in files:
        base_pose = resolve_base_pose(path, dataset_root, fixed_base_pos, fixed_base_ori, base_map)
        if base_pose is None:
            missing_base.append(str(path))
        summary = file_ee_range_from_obs(path, base_pose=base_pose)
        if summary is None:
            missing.append(str(path))
        else:
            summaries.append(summary)

    print("files_with_direct_ee_obs", len(summaries))
    print("files_missing_direct_ee_obs", len(missing))
    print("files_missing_base_pose", len(missing_base))
    if missing:
        print("missing_direct_ee_obs_examples", missing[:10])
    if missing_base:
        print("missing_base_pose_examples", missing_base[:10])
    if not summaries:
        return None

    mins = np.min(np.stack([s["min"] for s in summaries]), axis=0)
    maxs = np.max(np.stack([s["max"] for s in summaries]), axis=0)
    total = sum(s["count"] for s in summaries)
    keys = sorted({key for s in summaries for key in s["keys"]})
    print("ee_obs_keys", keys)
    print("ee_samples", total)
    print_range("ee_world", mins, maxs)

    base_summaries = [s for s in summaries if "base_min" in s]
    if base_summaries:
        base_mins = np.min(np.stack([s["base_min"] for s in base_summaries]), axis=0)
        base_maxs = np.max(np.stack([s["base_max"] for s in base_summaries]), axis=0)
        print("ee_base_files", len(base_summaries))
        print_range("ee_base", base_mins, base_maxs)

    print("per_file_ranges")
    for s in summaries:
        item = {
            "path": s["path"],
            "count": s["count"],
            "keys": s["keys"],
            "world_min": np.round(s["min"], 6).tolist(),
            "world_max": np.round(s["max"], 6).tolist(),
        }
        if "base_min" in s:
            item.update(
                {
                    "base_min": np.round(s["base_min"], 6).tolist(),
                    "base_max": np.round(s["base_max"], 6).tolist(),
                    "base_pos": np.round(s["base_pos"], 6).tolist(),
                }
            )
            if "base_ori" in s:
                item["base_ori"] = np.round(s["base_ori"], 6).tolist()
        print(json.dumps(item, ensure_ascii=False))
    return mins, maxs


def sample_task_base_positions(suites):
    from pathlib import Path

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    rows = []
    for suite in suites:
        bm = benchmark.get_benchmark_dict()[suite]()
        task_ids = sorted(set([0, bm.n_tasks // 2, bm.n_tasks - 1]))
        for task_id in task_ids:
            task = bm.get_task(task_id)
            bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=32, camera_widths=32)
            env.reset()
            robot = env.robots[0]
            eef = env.sim.data.site_xpos[int(robot.eef_site_id)].copy()
            rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": task.name,
                    "base_pos": np.asarray(robot.base_pos, dtype=np.float64),
                    "base_ori": np.asarray(robot.base_ori, dtype=np.float64),
                    "eef_init": np.asarray(eef, dtype=np.float64),
                }
            )
            env.close()
    return rows


def summarize_sampled_bases(suites):
    rows = sample_task_base_positions(suites)
    print("sampled_base_rows", len(rows))
    for row in rows:
        print(
            json.dumps(
                {
                    "suite": row["suite"],
                    "task_id": row["task_id"],
                    "task_name": row["task_name"],
                    "base_pos": np.round(row["base_pos"], 6).tolist(),
                    "base_ori": np.round(row["base_ori"], 6).tolist(),
                    "eef_init": np.round(row["eef_init"], 6).tolist(),
                },
                ensure_ascii=False,
            )
        )
    base_positions = np.stack([row["base_pos"] for row in rows])
    base_oris = np.stack([row["base_ori"] for row in rows])
    print("sampled_base_unique", np.unique(np.round(base_positions, 6), axis=0).tolist())
    print("sampled_base_ori_unique", np.unique(np.round(base_oris, 6), axis=0).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=get_default_dataset_root())
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--sample-bases", action="store_true")
    parser.add_argument("--infer-base-from-suites", action="store_true")
    parser.add_argument("--base-pos", nargs=3, type=float, default=None)
    parser.add_argument("--base-ori", nargs=4, type=float, default=None)
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    args = parser.parse_args()

    if not args.skip_dataset:
        base_map = build_task_base_map(args.suites) if args.infer_base_from_suites else None
        summarize_dataset_ee(
            args.dataset_root,
            fixed_base_pos=args.base_pos,
            fixed_base_ori=args.base_ori,
            base_map=base_map,
        )
    if args.sample_bases:
        summarize_sampled_bases(args.suites)


if __name__ == "__main__":
    main()
