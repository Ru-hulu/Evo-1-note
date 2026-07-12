from pathlib import Path
import argparse
import json
import time
import numpy as np

# 它负责把 LIBERO 里的 MountedPanda 机器人，从 MuJoCo/robosuite 模型转换成 voxel 表示。
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PRECOMPUTED_PATH = PROJECT_ROOT / "mounted_panda_collision_voxels_005.npz"
VOXEL_SIZE = 0.005
DEFAULT_SUITE = "libero_spatial"
DEFAULT_TASK_ID = 0


def _import_libero():
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    return benchmark, OffScreenRenderEnv, get_libero_path


def make_libero_env(task_suite=DEFAULT_SUITE, task_id=DEFAULT_TASK_ID, resolution=64):
    benchmark, OffScreenRenderEnv, get_libero_path = _import_libero()
    bm = benchmark.get_benchmark_dict()[task_suite]()
    task = bm.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.reset()
    return env, task


def find_robot(env):
    if not getattr(env, "robots", None):
        raise RuntimeError("LIBERO env did not expose env.robots.")
    robot = env.robots[0]
    robot_name = getattr(robot, "name", "")
    if robot_name != "MountedPanda":
        print(f"warning: expected MountedPanda, got {robot_name!r}")
    return robot


def mesh_points_from_mujoco_model(model, mesh_id, voxel_size):
    import trimesh

    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_end = vert_start + int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_end = face_start + int(model.mesh_facenum[mesh_id])

    vertices = np.asarray(model.mesh_vert[vert_start:vert_end], dtype=np.float64)
    faces = np.asarray(model.mesh_face[face_start:face_end], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    vox = mesh.voxelized(voxel_size)
    try:
        vox = vox.fill()
    except Exception:
        pass
    return np.asarray(vox.points, dtype=np.float32)


def box_points(size, voxel_size):
    size = np.asarray(size, dtype=np.float64)
    axes = []
    for half_extent in size:
        count = max(1, int(np.ceil((2.0 * half_extent) / voxel_size)))
        lo = -half_extent + 0.5 * voxel_size
        hi = half_extent - 0.5 * voxel_size
        if count == 1 or hi < lo:
            axes.append(np.array([0.0], dtype=np.float64))
        else:
            axes.append(np.linspace(lo, hi, count, dtype=np.float64))
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid.astype(np.float32)


def geom_local_points(model, geom_id, voxel_size):
    geom_type = int(model.geom_type[geom_id])
    data_id = int(model.geom_dataid[geom_id])

    # MuJoCo geom type 7 is mesh, type 6 is box.
    if geom_type == 7 and data_id >= 0:
        return mesh_points_from_mujoco_model(model, data_id, voxel_size)
    if geom_type == 6:
        return box_points(model.geom_size[geom_id], voxel_size)

    raise ValueError(f"Unsupported robot geom type {geom_type} for geom id {geom_id}.")


def collect_robot_geom_names(env):
    robot = find_robot(env)
    names = []

    robot_model = getattr(robot, "robot_model", None)
    if robot_model is not None:
        names.extend(getattr(robot_model, "contact_geoms", []) or [])

    gripper = getattr(robot, "gripper", None)
    if gripper is not None:
        names.extend(getattr(gripper, "contact_geoms", []) or [])

    model = env.sim.model
    present = []
    for name in names:
        try:
            model.geom_name2id(name)
            present.append(name)
        except Exception:
            print(f"warning: contact geom missing in compiled model: {name}")

    if not present:
        raise RuntimeError("No robot collision geoms found for LIBERO robot.")
    return present


def get_joint_metadata(env):
    robot = find_robot(env)
    model = env.sim.model
    robot_model = robot.robot_model
    gripper = getattr(robot, "gripper", None)

    arm_joints = list(robot_model.joints)
    gripper_joints = list(getattr(gripper, "joints", []) or [])
    joint_names = arm_joints + gripper_joints

    qpos_adrs = []
    joint_ranges = []
    for name in joint_names:
        joint_id = model.joint_name2id(name)
        qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        joint_ranges.append(np.asarray(model.jnt_range[joint_id], dtype=np.float32))

    arm_init = np.asarray(robot_model.init_qpos, dtype=np.float32)
    gripper_init = np.asarray(getattr(gripper, "init_qpos", []), dtype=np.float32)
    init_qpos = np.concatenate([arm_init, gripper_init], axis=0)

    return {
        "robot_name": str(getattr(robot, "name", "")),
        "arm_joint_names": np.asarray(arm_joints),
        "gripper_joint_names": np.asarray(gripper_joints),
        "joint_names": np.asarray(joint_names),
        "qpos_adrs": np.asarray(qpos_adrs, dtype=np.int32),
        "joint_ranges": np.asarray(joint_ranges, dtype=np.float32),
        "init_qpos": init_qpos.astype(np.float32),
    }


def collect_kinematic_metadata(env, geom_names):
    model = env.sim.model
    geom_ids = np.asarray([model.geom_name2id(str(name)) for name in geom_names], dtype=np.int32)

    body_names = [model.body_id2name(i) or "" for i in range(model.nbody)]
    joint_names = [model.joint_id2name(i) or "" for i in range(model.njnt)]

    return {
        "body_names": np.asarray(body_names),
        "body_parent_ids": np.asarray(model.body_parentid, dtype=np.int32),
        "body_pos": np.asarray(model.body_pos, dtype=np.float32),
        "body_quat": np.asarray(model.body_quat, dtype=np.float32),
        "body_jntadr": np.asarray(model.body_jntadr, dtype=np.int32),
        "body_jntnum": np.asarray(model.body_jntnum, dtype=np.int32),
        "all_joint_names": np.asarray(joint_names),
        "jnt_body_ids": np.asarray(model.jnt_bodyid, dtype=np.int32),
        "jnt_type": np.asarray(model.jnt_type, dtype=np.int32),
        "jnt_axis": np.asarray(model.jnt_axis, dtype=np.float32),
        "jnt_pos": np.asarray(model.jnt_pos, dtype=np.float32),
        "jnt_qposadr": np.asarray(model.jnt_qposadr, dtype=np.int32),
        "geom_ids": geom_ids,
        "geom_body_ids": np.asarray(model.geom_bodyid[geom_ids], dtype=np.int32),
        "geom_pos": np.asarray(model.geom_pos[geom_ids], dtype=np.float32),
        "geom_quat": np.asarray(model.geom_quat[geom_ids], dtype=np.float32),
    }


# 生成npz文件
def precompute(
    out_path=PRECOMPUTED_PATH,
    voxel_size=VOXEL_SIZE,
    task_suite=DEFAULT_SUITE,
    task_id=DEFAULT_TASK_ID,
):
    env, task = make_libero_env(task_suite=task_suite, task_id=task_id)
    try:
        model = env.sim.model
        geom_names = collect_robot_geom_names(env)
        joint_meta = get_joint_metadata(env)
        kinematic_meta = collect_kinematic_metadata(env, geom_names)

        local_chunks = []
        geom_slices = []
        geom_body_names = []
        cursor = 0

        for name in geom_names:
            geom_id = model.geom_name2id(name)
            body_id = int(model.geom_bodyid[geom_id])
            body_name = model.body_id2name(body_id)
            points = geom_local_points(model, geom_id, voxel_size)
            grid = np.unique(np.round(points / voxel_size).astype(np.int32), axis=0)
            points = (grid.astype(np.float32) * voxel_size).astype(np.float32)

            local_chunks.append(points)
            geom_slices.append([cursor, cursor + len(points)])
            geom_body_names.append(body_name)
            cursor += len(points)
            print(f"{name:32s} {body_name:28s} {len(points):7d} local voxels")

        local_points = np.vstack(local_chunks).astype(np.float32)
        geom_slices = np.asarray(geom_slices, dtype=np.int32)

        zero = build_current_voxels_from_env(env, local_points, geom_slices, geom_names, {}, voxel_size, joint_meta)

        np.savez_compressed(
            out_path,
            voxel_size=np.asarray(voxel_size, dtype=np.float32),
            task_suite=np.asarray(task_suite),
            task_id=np.asarray(task_id, dtype=np.int32),
            task_language=np.asarray(task.language),
            robot_name=np.asarray(joint_meta["robot_name"]),
            geom_names=np.asarray(geom_names),
            geom_body_names=np.asarray(geom_body_names),
            geom_slices=geom_slices,
            local_points=local_points,
            arm_joint_names=joint_meta["arm_joint_names"],
            gripper_joint_names=joint_meta["gripper_joint_names"],
            joint_names=joint_meta["joint_names"],
            qpos_adrs=joint_meta["qpos_adrs"],
            joint_ranges=joint_meta["joint_ranges"],
            init_qpos=joint_meta["init_qpos"],
            body_names=kinematic_meta["body_names"],
            body_parent_ids=kinematic_meta["body_parent_ids"],
            body_pos=kinematic_meta["body_pos"],
            body_quat=kinematic_meta["body_quat"],
            body_jntadr=kinematic_meta["body_jntadr"],
            body_jntnum=kinematic_meta["body_jntnum"],
            all_joint_names=kinematic_meta["all_joint_names"],
            jnt_body_ids=kinematic_meta["jnt_body_ids"],
            jnt_type=kinematic_meta["jnt_type"],
            jnt_axis=kinematic_meta["jnt_axis"],
            jnt_pos=kinematic_meta["jnt_pos"],
            jnt_qposadr=kinematic_meta["jnt_qposadr"],
            geom_ids=kinematic_meta["geom_ids"],
            geom_body_ids=kinematic_meta["geom_body_ids"],
            geom_pos=kinematic_meta["geom_pos"],
            geom_quat=kinematic_meta["geom_quat"],
            zero_indices=zero["indices"],
            zero_points=zero["points"],
            zero_link_ids=zero["link_ids"],
        )
        print(f"saved {out_path}")
        print(f"robot: {joint_meta['robot_name']}")
        print(f"task: {task.language}")
        print(f"zero pose occupied voxels: {len(zero['indices'])}")
    finally:
        env.close()


def load_precomputed(path=PRECOMPUTED_PATH):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _state_dict(pre, states):
    joint_names = [str(x) for x in pre["joint_names"]]
    arm_joint_names = [str(x) for x in pre["arm_joint_names"]]
    if states is None:
        return {}
    if isinstance(states, dict):
        return {str(k): float(v) for k, v in states.items()}
    if len(states) == len(arm_joint_names) + 1:
        names = arm_joint_names + ["gripper_open"]
        return {name: float(value) for name, value in zip(names, states)}
    return {name: float(value) for name, value in zip(joint_names, states)}


def set_robot_qpos(env, pre, states):
    model = env.sim.model
    data = env.sim.data
    state_by_name = _state_dict(pre, states)

    qpos_adrs = np.asarray(pre["qpos_adrs"], dtype=np.int32)
    init_qpos = np.asarray(pre["init_qpos"], dtype=np.float64)
    joint_names = [str(x) for x in pre["joint_names"]]

    for adr, value in zip(qpos_adrs, init_qpos):
        data.qpos[int(adr)] = float(value)

    for name, adr in zip(joint_names, qpos_adrs):
        if name in state_by_name:
            data.qpos[int(adr)] = state_by_name[name]

    if "gripper_open" in state_by_name:
        open_value = float(state_by_name["gripper_open"])
        gripper_names = [str(x) for x in pre["gripper_joint_names"]]
        if len(gripper_names) >= 1:
            data.qpos[model.jnt_qposadr[model.joint_name2id(gripper_names[0])]] = open_value
        if len(gripper_names) >= 2:
            data.qpos[model.jnt_qposadr[model.joint_name2id(gripper_names[1])]] = -open_value

    env.sim.forward()


def build_current_voxels_from_env(env, local_points, geom_slices, geom_names, states, voxel_size, pre):
    set_robot_qpos(env, pre, states)

    model = env.sim.model
    data = env.sim.data
    world_chunks = []
    link_id_chunks = []

    for link_id, geom_name in enumerate(geom_names):
        start, end = geom_slices[link_id]
        pts = local_points[start:end]
        if len(pts) == 0:
            continue
        geom_id = model.geom_name2id(str(geom_name))
        xpos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        world = pts @ xmat.T + xpos
        world_chunks.append(world.astype(np.float32))
        link_id_chunks.append(np.full(len(world), link_id, dtype=np.int16))

    raw_points = np.vstack(world_chunks)
    raw_link_ids = np.concatenate(link_id_chunks)
    indices = np.floor(raw_points / voxel_size).astype(np.int32)
    indices, keep = np.unique(indices, axis=0, return_index=True)
    points = (indices.astype(np.float32) + 0.5) * voxel_size

    return {
        "voxel_size": float(voxel_size),
        "origin": np.zeros(3, dtype=np.float32),
        "indices": indices.astype(np.int32),
        "points": points.astype(np.float32),
        "link_ids": raw_link_ids[keep].astype(np.int16),
        "geom_names": np.asarray(geom_names),
        "joint_names": np.asarray(pre["joint_names"]),
        "arm_joint_names": np.asarray(pre["arm_joint_names"]),
        "gripper_joint_names": np.asarray(pre["gripper_joint_names"]),
    }


def prepare_runtime(precomputed_path=PRECOMPUTED_PATH, task_suite=None, task_id=None):
    pre = load_precomputed(precomputed_path)
    suite = str(pre["task_suite"]) if task_suite is None else task_suite
    tid = int(pre["task_id"]) if task_id is None else int(task_id)
    env, _task = make_libero_env(task_suite=suite, task_id=tid)
    return env, pre


def close_runtime(runtime):
    env, _pre = runtime
    env.close()


def runtime_states_to_voxels(runtime, states):
    env, pre = runtime
    return build_current_voxels_from_env(
        env,
        pre["local_points"],
        pre["geom_slices"],
        pre["geom_names"],
        states,
        float(pre["voxel_size"]),
        pre,
    )


def states_to_voxels(states, precomputed_path=PRECOMPUTED_PATH):
    runtime = prepare_runtime(precomputed_path)
    try:
        return runtime_states_to_voxels(runtime, states)
    finally:
        close_runtime(runtime)


def demo(args):
    states = args.states
    if args.states_json:
        states = json.loads(args.states_json)

    runtime = prepare_runtime(args.npz)
    try:
        t0 = time.perf_counter()
        voxels = runtime_states_to_voxels(runtime, states)
        dt = (time.perf_counter() - t0) * 1000.0

        print("robot:", str(runtime[1]["robot_name"]))
        print("joint_order:", [str(x) for x in voxels["joint_names"]])
        print("geom_count:", len(voxels["geom_names"]))
        print("voxel_size:", voxels["voxel_size"])
        print("indices shape:", voxels["indices"].shape, "int32")
        print("points shape:", voxels["points"].shape, "float32")
        print("link_ids shape:", voxels["link_ids"].shape, "int16")
        print("time_ms:", round(dt, 3))
        print("first 5 indices:\n", voxels["indices"][:5])

        if args.save:
            np.savez_compressed(
                args.save,
                voxel_size=np.asarray(voxels["voxel_size"], dtype=np.float32),
                origin=voxels["origin"],
                indices=voxels["indices"],
                points=voxels["points"],
                link_ids=voxels["link_ids"],
                geom_names=voxels["geom_names"],
                joint_names=voxels["joint_names"],
            )
            print(f"saved {args.save}")
    finally:
        close_runtime(runtime)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("precompute")
    p0.add_argument("--out", default=str(PRECOMPUTED_PATH))
    p0.add_argument("--voxel-size", type=float, default=VOXEL_SIZE)
    p0.add_argument("--task-suite", default=DEFAULT_SUITE)
    p0.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)

    p1 = sub.add_parser("demo")
    p1.add_argument("--npz", default=str(PRECOMPUTED_PATH))
    p1.add_argument("--states", nargs="*", type=float, default=None)
    p1.add_argument("--states-json", default=None)
    p1.add_argument("--save", default=None)

    args = parser.parse_args()
    if args.cmd == "precompute":
        precompute(args.out, args.voxel_size, args.task_suite, args.task_id)
    elif args.cmd == "demo":
        demo(args)


if __name__ == "__main__":
    main()
