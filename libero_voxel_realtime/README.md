# LIBERO MountedPanda Realtime Voxels

This folder mirrors the SO101 voxel viewer for the robot used by LIBERO.

## Layout

```text
common/
  workspace.py              shared base-frame workspace crop utilities
sim/
  libero_voxel.py           LIBERO/MuJoCo cache generation and runtime voxel path
  bounded_robot_voxel.py    MuJoCo reference path for bounded 1cm occupancy
  interactive_*.py          simulator-backed realtime viewers
  analyze_libero_workspace.py
standalone/
  bounded_robot_voxel.py    npz-only state mapping -> bounded 1cm occupancy
```

The `.npz` cache files stay in this folder root.

Confirmed from a LIBERO `OffScreenRenderEnv`:

```text
robot: MountedPanda
arm joints:
  robot0_joint1 ... robot0_joint7
gripper joints:
  gripper0_finger_joint1
  gripper0_finger_joint2
```

The implementation uses the compiled robosuite / MuJoCo model instead of
hand-written FK:

```text
slider qpos -> env.sim.forward()
robot collision geom local voxels -> geom_xpos / geom_xmat -> world voxels
```

## Precompute

Run inside a Python environment with LIBERO, robosuite, mujoco, numpy, and
trimesh installed:

```bash
cd /Users/hongru/paper_project/Evo-1-note/libero_voxel_realtime
python sim/libero_voxel.py precompute
```

This writes:

```text
mounted_panda_collision_voxels_005.npz
```

## Check

```bash
python sim/libero_voxel.py demo
python sim/interactive_voxel_3d.py --check
```

## Interactive Viewer

The GUI additionally needs `PyQt5` and `pyqtgraph`:

```bash
python sim/interactive_voxel_3d.py
```

or:

```bash
./run_interactive.sh
```

The gripper is exposed as one slider named `gripper_open`; internally it maps to
the two Panda gripper finger joints with opposite signs.

At `voxel_size=0.005`, MountedPanda produces roughly 150k occupied voxels in the
default pose. The viewer starts in point-rendering mode for responsiveness; turn
on `Render as cubes` when inspecting a cropped region or when your machine can
comfortably draw the full mesh.

On the AutoDL VNC environment, prefer the CPU Matplotlib viewer because the
OpenGL widget can fail under TigerVNC. This viewer renders voxel centers as a
lightweight point cloud for responsiveness:

```bash
/root/start-libero-voxel.sh
```

The script starts VNC if needed, sets the Qt plugin path safely, checks the
precomputed voxel cache, and opens `sim/interactive_voxel_matplotlib.py`.

To change the point budget:

```bash
/root/start-libero-voxel.sh --max-points 50000
```

The Matplotlib viewer fixes the world bounds after the first draw so the grid
does not move when joint sliders update. Use `Fit Bounds` to recenter once, or
enable `Follow voxel bounds` only when you intentionally want the view to track
the current robot pose.

## Fixed 1cm Bounded Robot Voxel

`sim/bounded_robot_voxel.py` is the MuJoCo reference path for model input. It
keeps the robot collision voxel size fixed at `1cm`, transforms the current
robot voxels into the base frame, and clips them to a fixed workspace box.

Default LIBERO spatial EE workspace is rounded outward to the 1cm grid:

```text
raw x:  0.335313 to 0.837761 m  ->  x:  0.33 to 0.84 m
raw y: -0.286172 to 0.394537 m  ->  y: -0.29 to 0.40 m
raw z: -0.002086 to 0.420008 m  ->  z: -0.01 to 0.43 m
```

Generate the fixed 1cm collision cache:

```bash
cd /root/libero_voxel_realtime
/root/miniconda3/envs/spatialvla/bin/python sim/bounded_robot_voxel.py precompute
```

Generate one bounded occupancy grid from the default joint state:

```bash
/root/miniconda3/envs/spatialvla/bin/python sim/bounded_robot_voxel.py demo \
  --save /root/libero_voxel_realtime/example_bounded_voxel_010.npz
```

Override the crop range explicitly:

```bash
/root/miniconda3/envs/spatialvla/bin/python sim/bounded_robot_voxel.py demo \
  --bounds 0.33 0.84 -0.29 0.40 -0.01 0.43
```

The saved `.npz` uses `axis_order = xyz`; `occupancy.shape` is therefore
`(nx, ny, nz)`. For the default rounded bounds, the shape is `(51, 69, 44)`.

## Standalone Bounded Robot Voxel

`standalone/bounded_robot_voxel.py` is the no-simulator runtime path. The idea
is to store enough kinematic-tree information in the precomputed `.npz`, then
use only NumPy to compute:

```text
joint states -> body/geom poses -> world voxels -> base-frame bounded occupancy
```

This path still needs the simulator once when generating the cache, because the
cache is extracted from the compiled LIBERO / robosuite / MuJoCo model. After
that, the standalone script only needs the `.npz` file.

Regenerate the 1cm cache first so it includes the kinematic metadata:

```bash
/root/miniconda3/envs/spatialvla/bin/python sim/bounded_robot_voxel.py precompute
```

Then generate a bounded occupancy grid without creating a LIBERO environment:

```bash
python standalone/bounded_robot_voxel.py \
  --npz mounted_panda_collision_voxels_010.npz \
  --save example_bounded_voxel_standalone_010.npz
```

Pass robot states as a mapping from joint name to value:

```bash
python standalone/bounded_robot_voxel.py \
  --npz mounted_panda_collision_voxels_010.npz \
  --states-json '{"robot0_joint1": 0.1, "robot0_joint2": -0.2, "gripper_open": 0.02}'
```

The return fields match the data-oriented runtime path:

```text
points_base      occupied voxel centers in the robot base frame
occupancy        bool grid, True means the robot occupies that voxel
link_grid        int grid, -1 means empty, otherwise the source geom id
grid_shape_xyz   xyz shape of occupancy/link_grid
```

Before replacing the MuJoCo runtime path in training code, compare the standalone
output against `sim/bounded_robot_voxel.py demo` on the remote environment once.
The kinematic metadata is intentionally kept in the cache so this comparison can
be done directly.
