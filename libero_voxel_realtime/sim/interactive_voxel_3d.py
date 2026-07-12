import argparse
import importlib.util
import os
import sys
import time

import numpy as np

try:
    from .libero_voxel import PRECOMPUTED_PATH, close_runtime, prepare_runtime, runtime_states_to_voxels
except ImportError:
    from libero_voxel import PRECOMPUTED_PATH, close_runtime, prepare_runtime, runtime_states_to_voxels


NPZ_PATH = str(PRECOMPUTED_PATH)

LINK_COLORS = np.array([
    [0.10, 0.45, 0.95, 1.00],
    [1.00, 0.55, 0.05, 1.00],
    [0.15, 0.75, 0.25, 1.00],
    [0.95, 0.20, 0.18, 1.00],
    [0.65, 0.42, 0.92, 1.00],
    [0.65, 0.38, 0.22, 1.00],
    [0.95, 0.45, 0.78, 1.00],
    [0.12, 0.85, 0.90, 1.00],
    [0.95, 0.95, 0.20, 1.00],
    [0.40, 0.80, 0.95, 1.00],
    [0.90, 0.55, 0.35, 1.00],
    [0.75, 0.75, 0.75, 1.00],
    [0.35, 0.95, 0.55, 1.00],
], dtype=np.float32)

CUBE_VERTS = np.array([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
], dtype=np.float32)

CUBE_FACES = np.array([
    [0, 1, 2], [0, 2, 3],
    [4, 6, 5], [4, 7, 6],
    [0, 4, 5], [0, 5, 1],
    [1, 5, 6], [1, 6, 2],
    [2, 6, 7], [2, 7, 3],
    [3, 7, 4], [3, 4, 0],
], dtype=np.int32)


def build_voxel_mesh(points, link_ids, cube_size):
    n = len(points)
    vertices = (points[:, None, :] + CUBE_VERTS[None, :, :] * cube_size).reshape(n * 8, 3).astype(np.float32)
    offsets = (np.arange(n, dtype=np.int32) * 8)[:, None, None]
    faces = (CUBE_FACES[None, :, :] + offsets).reshape(n * 12, 3).astype(np.int32)
    cube_colors = LINK_COLORS[link_ids % len(LINK_COLORS)]
    face_colors = np.repeat(cube_colors, 12, axis=0).astype(np.float32)
    return vertices, faces, face_colors


def make_grid(gl, size=1.4, spacing=0.05):
    grid = gl.GLGridItem()
    grid.setSize(x=size, y=size, z=0)
    grid.setSpacing(x=spacing, y=spacing, z=spacing)
    grid.setDepthValue(10)
    return grid


def make_gripper_slider_name():
    return "gripper_open"


def joint_slider_specs(pre):
    names = [str(x) for x in pre["arm_joint_names"]]
    ranges = np.asarray(pre["joint_ranges"], dtype=np.float32)
    init_qpos = np.asarray(pre["init_qpos"], dtype=np.float32)

    specs = []
    for i, name in enumerate(names):
        low, high = ranges[i]
        specs.append((name, float(low), float(high), float(init_qpos[i])))

    gripper_names = [str(x) for x in pre["gripper_joint_names"]]
    if gripper_names:
        gripper_start = len(names)
        low, high = ranges[gripper_start]
        init = init_qpos[gripper_start]
        specs.append((make_gripper_slider_name(), float(low), float(high), float(init)))

    return specs


def default_states(pre):
    states = {}
    for name, _low, _high, init in joint_slider_specs(pre):
        states[name] = init
    return states


def run_check(npz_path=NPZ_PATH):
    runtime = prepare_runtime(npz_path)
    try:
        states = default_states(runtime[1])
        t0 = time.perf_counter()
        voxels = runtime_states_to_voxels(runtime, states)
        state_ms = (time.perf_counter() - t0) * 1000.0
        vertices, faces, face_colors = build_voxel_mesh(
            voxels["points"].astype(np.float32),
            voxels["link_ids"].astype(np.int32),
            float(voxels["voxel_size"]) * 1.25,
        )
        print("robot", str(runtime[1]["robot_name"]))
        print("check_voxels", len(voxels["indices"]))
        print("check_ms", round(state_ms, 3))
        print("mesh_vertices", vertices.shape)
        print("mesh_faces", faces.shape)
        print("mesh_face_colors", face_colors.shape)
        print("slider_joints", list(states.keys()))
    finally:
        close_runtime(runtime)


def run_gui(npz_path=NPZ_PATH):
    runtime = prepare_runtime(npz_path)
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["QT_OPENGL"] = "software"
    os.environ["QT_X11_NO_MITSHM"] = "1"
    os.environ.pop("QT_PLUGIN_PATH", None)
    pyqt_spec = importlib.util.find_spec("PyQt5")
    if pyqt_spec and pyqt_spec.submodule_search_locations:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
            pyqt_spec.submodule_search_locations[0],
            "Qt5",
            "plugins",
            "platforms",
        )

    from PyQt5 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    class JointSlider(QtWidgets.QWidget):
        valueChanged = QtCore.pyqtSignal()

        def __init__(self, name, lower, upper, init_value):
            super().__init__()
            self.name = name
            self.lower = lower
            self.upper = upper
            self.scale = 1000

            layout = QtWidgets.QHBoxLayout(self)
            layout.setContentsMargins(0, 2, 0, 2)
            self.label = QtWidgets.QLabel(name)
            self.label.setMinimumWidth(145)
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setRange(0, self.scale)
            self.value_label = QtWidgets.QLabel("0.000")
            self.value_label.setMinimumWidth(62)
            layout.addWidget(self.label)
            layout.addWidget(self.slider, 1)
            layout.addWidget(self.value_label)

            self.set_value(init_value)
            self.slider.valueChanged.connect(self._emit)

        def value(self):
            t = self.slider.value() / self.scale
            return self.lower + t * (self.upper - self.lower)

        def set_value(self, value):
            if self.upper <= self.lower:
                t = 0.0
            else:
                t = (value - self.lower) / (self.upper - self.lower)
            self.slider.setValue(int(round(max(0.0, min(1.0, t)) * self.scale)))
            self.value_label.setText(f"{self.value():.3f}")

        def _emit(self):
            self.value_label.setText(f"{self.value():.3f}")
            self.valueChanged.emit()

    class VoxelWindow(QtWidgets.QMainWindow):
        def __init__(self, runtime):
            super().__init__()
            self.setWindowTitle("LIBERO MountedPanda realtime 3D voxels")
            self.runtime = runtime
            self.pre = self.runtime[1]
            self.slider_specs = joint_slider_specs(self.pre)

            self.timer = QtCore.QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.update_voxels)

            root = QtWidgets.QWidget()
            self.setCentralWidget(root)
            layout = QtWidgets.QHBoxLayout(root)

            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor("k")
            layout.addWidget(self.view, 1)
            self.reset_camera()

            self.view.addItem(make_grid(gl))
            axis = gl.GLAxisItem()
            axis.setSize(0.30, 0.30, 0.30)
            self.view.addItem(axis)

            panel = QtWidgets.QWidget()
            panel.setMaximumWidth(440)
            panel_layout = QtWidgets.QVBoxLayout(panel)
            layout.addWidget(panel)

            self.info = QtWidgets.QLabel("initializing")
            self.info.setWordWrap(True)
            panel_layout.addWidget(self.info)

            self.render_cubes = QtWidgets.QCheckBox("Render as cubes")
            self.render_cubes.setChecked(False)
            self.render_cubes.stateChanged.connect(self.schedule_update)
            panel_layout.addWidget(self.render_cubes)

            self.clip_enabled = QtWidgets.QCheckBox("Enable render crop")
            self.clip_enabled.setChecked(False)
            self.clip_enabled.stateChanged.connect(self.schedule_update)
            panel_layout.addWidget(self.clip_enabled)

            clip_group = QtWidgets.QGroupBox("Render crop range (m)")
            clip_layout = QtWidgets.QGridLayout(clip_group)
            self.clip_boxes = {}
            defaults = {
                "x_min": -0.55, "x_max": 0.55,
                "y_min": -0.55, "y_max": 0.55,
                "z_min": -0.05, "z_max": 1.20,
            }
            rows = [("x", "x_min", "x_max"), ("y", "y_min", "y_max"), ("z", "z_min", "z_max")]
            for row, (axis_name, low_key, high_key) in enumerate(rows):
                clip_layout.addWidget(QtWidgets.QLabel(axis_name), row, 0)
                low_box = self.make_clip_spin(defaults[low_key])
                high_box = self.make_clip_spin(defaults[high_key])
                self.clip_boxes[low_key] = low_box
                self.clip_boxes[high_key] = high_box
                clip_layout.addWidget(low_box, row, 1)
                clip_layout.addWidget(high_box, row, 2)
            panel_layout.addWidget(clip_group)

            self.sliders = {}
            for name, low, high, init in self.slider_specs:
                widget = JointSlider(name, low, high, init)
                widget.valueChanged.connect(self.schedule_update)
                self.sliders[name] = widget
                panel_layout.addWidget(widget)

            reset_joints = QtWidgets.QPushButton("Reset Joints")
            reset_joints.clicked.connect(self.reset_joints)
            panel_layout.addWidget(reset_joints)

            reset_camera = QtWidgets.QPushButton("Reset Camera")
            reset_camera.clicked.connect(self.reset_camera)
            panel_layout.addWidget(reset_camera)

            reset_clip = QtWidgets.QPushButton("Reset Crop")
            reset_clip.clicked.connect(self.reset_clip)
            panel_layout.addWidget(reset_clip)
            panel_layout.addStretch(1)

            tiny_v = np.zeros((3, 3), dtype=np.float32)
            tiny_f = np.array([[0, 1, 2]], dtype=np.int32)
            self.mesh = gl.GLMeshItem(vertexes=tiny_v, faces=tiny_f, smooth=False, drawEdges=False, drawFaces=True, shader="shaded", glOptions="opaque")
            self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3), dtype=np.float32), size=10, color=(1, 1, 1, 1), pxMode=True)
            self.view.addItem(self.mesh)
            self.view.addItem(self.scatter)
            self.update_voxels()

        def closeEvent(self, event):
            close_runtime(self.runtime)
            event.accept()

        def reset_camera(self):
            self.view.setCameraPosition(
                pos=QtGui.QVector3D(0.00, 0.00, 0.55),
                distance=1.70,
                elevation=22,
                azimuth=-55,
            )

        def make_clip_spin(self, value):
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(-3.0, 3.0)
            box.setDecimals(3)
            box.setSingleStep(0.01)
            box.setValue(value)
            box.valueChanged.connect(self.schedule_update)
            return box

        def states(self):
            return {name: widget.value() for name, widget in self.sliders.items()}

        def clip_ranges(self):
            return {name: box.value() for name, box in self.clip_boxes.items()}

        def apply_clip(self, points, link_ids):
            if not self.clip_enabled.isChecked():
                return points, link_ids
            c = self.clip_ranges()
            mask = (
                (points[:, 0] >= c["x_min"]) & (points[:, 0] <= c["x_max"]) &
                (points[:, 1] >= c["y_min"]) & (points[:, 1] <= c["y_max"]) &
                (points[:, 2] >= c["z_min"]) & (points[:, 2] <= c["z_max"])
            )
            return points[mask], link_ids[mask]

        def schedule_update(self, *_args):
            self.timer.start(20)

        def reset_joints(self):
            for name, _low, _high, init in self.slider_specs:
                self.sliders[name].set_value(init)
            self.update_voxels()

        def reset_clip(self):
            defaults = {
                "x_min": -0.55, "x_max": 0.55,
                "y_min": -0.55, "y_max": 0.55,
                "z_min": -0.05, "z_max": 1.20,
            }
            for name, value in defaults.items():
                self.clip_boxes[name].setValue(value)
            self.update_voxels()

        def update_voxels(self):
            t0 = time.perf_counter()
            voxels = runtime_states_to_voxels(self.runtime, self.states())
            state_ms = (time.perf_counter() - t0) * 1000.0

            all_points = voxels["points"].astype(np.float32)
            all_link_ids = voxels["link_ids"].astype(np.int32)
            points, link_ids = self.apply_clip(all_points, all_link_ids)
            colors = LINK_COLORS[link_ids % len(LINK_COLORS)] if len(points) else LINK_COLORS[:0]

            t1 = time.perf_counter()
            if len(points) == 0:
                self.mesh.hide()
                self.scatter.setData(pos=np.zeros((0, 3), dtype=np.float32), color=(1, 1, 1, 1), size=10, pxMode=True)
                self.scatter.show()
                mode = "empty"
            elif self.render_cubes.isChecked():
                cube_size = float(voxels["voxel_size"]) * 1.25
                vertices, faces, face_colors = build_voxel_mesh(points, link_ids, cube_size)
                self.mesh.setMeshData(vertexes=vertices, faces=faces, faceColors=face_colors, smooth=False)
                self.mesh.show()
                self.scatter.hide()
                mode = "cubes"
            else:
                self.scatter.setData(pos=points, color=colors, size=10, pxMode=True)
                self.scatter.show()
                self.mesh.hide()
                mode = "points"
            render_ms = (time.perf_counter() - t1) * 1000.0
            crop_text = "on" if self.clip_enabled.isChecked() else "off"

            self.info.setText(
                f"robot: {str(self.pre['robot_name'])}\n"
                f"visible voxels: {len(points)} / {len(all_points)}\n"
                f"state->voxel: {state_ms:.2f} ms\n"
                f"render update: {render_ms:.2f} ms\n"
                f"mode: {mode} | crop: {crop_text}\n"
                f"mouse: rotate / zoom / pan"
            )

    pg.setConfigOptions(antialias=False)
    app = QtWidgets.QApplication(sys.argv)
    try:
        win = VoxelWindow(runtime)
        runtime = None
        win.resize(1280, 840)
        win.show()
        sys.exit(app.exec_())
    finally:
        if runtime is not None:
            close_runtime(runtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=NPZ_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        run_check(args.npz)
    else:
        run_gui(args.npz)


if __name__ == "__main__":
    main()
