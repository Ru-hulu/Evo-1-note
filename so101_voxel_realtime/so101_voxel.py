from pathlib import Path
import argparse
import json
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
URDF_PATH = HERE / "SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
PRECOMPUTED_PATH = HERE / "so101_collision_voxels_005.npz"
VOXEL_SIZE = 0.005


def parse_vec(text, default):
    if text is None:
        return np.array(default, dtype=np.float64)
    return np.array([float(x) for x in text.split()], dtype=np.float64)


def transform_from_xyz_rpy(xyz, rpy):
    x, y, z = xyz
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rz @ ry @ rx
    mat[:3, 3] = [x, y, z]
    return mat


def origin_matrix(node):
    if node is None:
        return np.eye(4, dtype=np.float64)
    xyz = parse_vec(node.get("xyz"), [0, 0, 0])
    rpy = parse_vec(node.get("rpy"), [0, 0, 0])
    return transform_from_xyz_rpy(xyz, rpy)


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    r = np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = r
    return mat


def axis_translation(axis, value):
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = np.asarray(axis, dtype=np.float64) * value
    return mat


def read_robot(urdf_path):
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()

    links = [link.get("name") for link in root.findall("link")]
    collisions = {name: [] for name in links}
    for link in root.findall("link"):
        link_name = link.get("name")
        for col in link.findall("collision"):
            mesh_node = col.find("geometry/mesh")
            if mesh_node is None:
                continue
            collisions[link_name].append({
                "filename": mesh_node.get("filename"),
                "origin": origin_matrix(col.find("origin")),
                "scale": parse_vec(mesh_node.get("scale"), [1, 1, 1]),
            })

    children = {}
    parent_of = {}
    for j in root.findall("joint"):
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        joint = {
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": parent,
            "child": child,
            "origin": origin_matrix(j.find("origin")),
            "axis": parse_vec(j.find("axis").get("xyz"), [0, 0, 1]) if j.find("axis") is not None else np.array([0, 0, 1], dtype=np.float64),
        }
        children.setdefault(parent, []).append(joint)
        parent_of[child] = parent

    root_link = next(link for link in links if link not in parent_of)
    joint_order = []

    def walk(link_name):
        for joint in children.get(link_name, []):
            if joint["type"] != "fixed":
                joint_order.append(joint["name"])
            walk(joint["child"])

    walk(root_link)
    return {
        "urdf_path": urdf_path,
        "mesh_dir": urdf_path.parent,
        "links": links,
        "collisions": collisions,
        "children": children,
        "root_link": root_link,
        "joint_order": joint_order,
    }


def states_to_dict(robot, states):
    if states is None:
        return {}
    if isinstance(states, dict):
        return {k: float(v) for k, v in states.items()}
    return {name: float(value) for name, value in zip(robot["joint_order"], states)}


def forward_kinematics(robot, states):
    q = states_to_dict(robot, states)
    poses = {robot["root_link"]: np.eye(4, dtype=np.float64)}

    def walk(link_name):
        for joint in robot["children"].get(link_name, []):
            value = q.get(joint["name"], 0.0)
            motion = np.eye(4, dtype=np.float64)
            if joint["type"] in ("revolute", "continuous"):
                motion = axis_rotation(joint["axis"], value)
            elif joint["type"] == "prismatic":
                motion = axis_translation(joint["axis"], value)
            poses[joint["child"]] = poses[link_name] @ joint["origin"] @ motion
            walk(joint["child"])

    walk(robot["root_link"])
    return poses


def load_mesh_as_mesh(path):
    import trimesh
    loaded = trimesh.load_mesh(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def voxelize_mesh(mesh, voxel_size):
    vox = mesh.voxelized(voxel_size)
    try:
        vox = vox.fill()
    except Exception:
        pass
    return vox.points.astype(np.float32)


def precompute(urdf_path=URDF_PATH, out_path=PRECOMPUTED_PATH, voxel_size=VOXEL_SIZE):
    robot = read_robot(urdf_path)
    local_points_by_link = []
    link_slices = []
    link_names = []
    cursor = 0

    for link_name in robot["links"]:
        pieces = []
        for col in robot["collisions"].get(link_name, []):
            mesh_path = robot["mesh_dir"] / col["filename"]
            mesh = load_mesh_as_mesh(mesh_path)
            if not np.allclose(col["scale"], [1, 1, 1]):
                mesh.apply_scale(col["scale"])
            mesh.apply_transform(col["origin"])
            pieces.append(voxelize_mesh(mesh, voxel_size))

        if pieces:
            pts = np.vstack(pieces)
            grid = np.unique(np.round(pts / voxel_size).astype(np.int32), axis=0)
            pts = (grid.astype(np.float32) * voxel_size).astype(np.float32)
        else:
            pts = np.zeros((0, 3), dtype=np.float32)

        local_points_by_link.append(pts)
        link_names.append(link_name)
        link_slices.append([cursor, cursor + len(pts)])
        cursor += len(pts)
        print(f"{link_name:32s} {len(pts):7d} local voxels")

    local_points = np.vstack(local_points_by_link).astype(np.float32)
    link_slices = np.array(link_slices, dtype=np.int32)
    link_names = np.array(link_names)
    joint_names = np.array(robot["joint_order"])

    zero = build_current_voxels(robot, local_points, link_slices, link_names, {}, voxel_size)

    np.savez_compressed(
        out_path,
        voxel_size=np.array(voxel_size, dtype=np.float32),
        urdf_path=np.array(str(Path(urdf_path).resolve())),
        link_names=link_names,
        link_slices=link_slices,
        joint_names=joint_names,
        local_points=local_points,
        zero_indices=zero["indices"],
        zero_points=zero["points"],
        zero_link_ids=zero["link_ids"],
    )
    print(f"saved {out_path}")
    print(f"zero pose occupied voxels: {len(zero['indices'])}")


def build_current_voxels(robot, local_points, link_slices, link_names, states, voxel_size):
    poses = forward_kinematics(robot, states)
    world_chunks = []
    link_id_chunks = []

    for link_id, link_name in enumerate(link_names):
        start, end = link_slices[link_id]
        pts = local_points[start:end]
        if len(pts) == 0:
            continue
        T = poses[str(link_name)]
        world = pts @ T[:3, :3].T + T[:3, 3]
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
        "link_names": link_names,
        "joint_names": np.array(robot["joint_order"]),
    }


def load_precomputed(path=PRECOMPUTED_PATH):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def prepare_runtime(precomputed_path=PRECOMPUTED_PATH):
    pre = load_precomputed(precomputed_path)
    robot = read_robot(URDF_PATH)
    return robot, pre


def runtime_states_to_voxels(runtime, states):
    robot, pre = runtime
    return build_current_voxels(
        robot,
        pre["local_points"],
        pre["link_slices"],
        pre["link_names"],
        states,
        float(pre["voxel_size"]),
    )


def states_to_voxels(states, precomputed_path=PRECOMPUTED_PATH):
    return runtime_states_to_voxels(prepare_runtime(precomputed_path), states)


def demo(args):
    states = args.states
    if args.states_json:
        states = json.loads(args.states_json)

    runtime = prepare_runtime(args.npz)
    t0 = time.perf_counter()
    voxels = runtime_states_to_voxels(runtime, states)
    dt = (time.perf_counter() - t0) * 1000.0

    print("joint_order:", [str(x) for x in voxels["joint_names"]])
    print("voxel_size:", voxels["voxel_size"])
    print("indices shape:", voxels["indices"].shape, "int32")
    print("points shape:", voxels["points"].shape, "float32")
    print("link_ids shape:", voxels["link_ids"].shape, "int16")
    print("time_ms:", round(dt, 3))
    print("first 5 indices:\n", voxels["indices"][:5])

    if args.save:
        np.savez_compressed(
            args.save,
            voxel_size=np.array(voxels["voxel_size"], dtype=np.float32),
            origin=voxels["origin"],
            indices=voxels["indices"],
            points=voxels["points"],
            link_ids=voxels["link_ids"],
            link_names=voxels["link_names"],
            joint_names=voxels["joint_names"],
        )
        print(f"saved {args.save}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("precompute")
    p0.add_argument("--urdf", default=str(URDF_PATH))
    p0.add_argument("--out", default=str(PRECOMPUTED_PATH))
    p0.add_argument("--voxel-size", type=float, default=VOXEL_SIZE)

    p1 = sub.add_parser("demo")
    p1.add_argument("--npz", default=str(PRECOMPUTED_PATH))
    p1.add_argument("--states", nargs="*", type=float, default=[0, 0, 0, 0, 0, 0])
    p1.add_argument("--states-json", default=None)
    p1.add_argument("--save", default=None)

    args = parser.parse_args()
    if args.cmd == "precompute":
        precompute(args.urdf, args.out, args.voxel_size)
    elif args.cmd == "demo":
        demo(args)


if __name__ == "__main__":
    main()
