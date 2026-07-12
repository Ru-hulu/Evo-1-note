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

def configure_qt_plugins():
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


def run_gui(npz_path=NPZ_PATH, max_points=50000):
    runtime = prepare_runtime(npz_path)
    configure_qt_plugins()

    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from PyQt5 import QtCore, QtWidgets

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
        def __init__(self, runtime, max_points):
            super().__init__()
            self.setWindowTitle("LIBERO MountedPanda voxel viewer")
            self.runtime = runtime
            self.pre = self.runtime[1]
            self.slider_specs = joint_slider_specs(self.pre)

            self.timer = QtCore.QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.update_voxels)

            root = QtWidgets.QWidget()
            self.setCentralWidget(root)
            layout = QtWidgets.QHBoxLayout(root)

            self.figure = Figure(figsize=(8, 7), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111, projection="3d")
            layout.addWidget(self.canvas, 1)

            panel = QtWidgets.QWidget()
            panel.setMaximumWidth(440)
            panel_layout = QtWidgets.QVBoxLayout(panel)
            layout.addWidget(panel)

            self.info = QtWidgets.QLabel("initializing")
            self.info.setWordWrap(True)
            panel_layout.addWidget(self.info)

            max_row = QtWidgets.QHBoxLayout()
            max_row.addWidget(QtWidgets.QLabel("Max plotted points"))
            self.max_points = QtWidgets.QSpinBox()
            self.max_points.setRange(1000, 200000)
            self.max_points.setSingleStep(5000)
            self.max_points.setValue(int(max_points))
            self.max_points.valueChanged.connect(self.schedule_update)
            max_row.addWidget(self.max_points)
            panel_layout.addLayout(max_row)

            self.follow_bounds = QtWidgets.QCheckBox("Follow voxel bounds")
            self.follow_bounds.setChecked(False)
            self.follow_bounds.stateChanged.connect(self.schedule_update)
            panel_layout.addWidget(self.follow_bounds)

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

            fit_bounds = QtWidgets.QPushButton("Fit Bounds")
            fit_bounds.clicked.connect(self.fit_bounds)
            panel_layout.addWidget(fit_bounds)

            reset_view = QtWidgets.QPushButton("Reset View")
            reset_view.clicked.connect(self.reset_view)
            panel_layout.addWidget(reset_view)

            reset_clip = QtWidgets.QPushButton("Reset Crop")
            reset_clip.clicked.connect(self.reset_clip)
            panel_layout.addWidget(reset_clip)
            panel_layout.addStretch(1)

            self.view_elev = 22
            self.view_azim = -55
            self.fixed_bounds = None
            self.last_visible_points = None
            self.update_voxels()

        def closeEvent(self, event):
            close_runtime(self.runtime)
            event.accept()

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

        def sample_points(self, points, link_ids):
            max_points = int(self.max_points.value())
            if len(points) <= max_points:
                return points, link_ids
            keep = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
            return points[keep], link_ids[keep]

        def schedule_update(self, *_args):
            self.timer.start(120)

        def reset_joints(self):
            for name, _low, _high, init in self.slider_specs:
                self.sliders[name].set_value(init)
            self.update_voxels()

        def reset_view(self):
            self.view_elev = 22
            self.view_azim = -55
            self.update_voxels()

        def fit_bounds(self):
            if self.last_visible_points is not None and len(self.last_visible_points):
                self.fixed_bounds = self.compute_bounds(self.last_visible_points)
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

        def compute_bounds(self, points):
            if points is not None and len(points):
                mins = points.min(axis=0)
                maxs = points.max(axis=0)
                center = (mins + maxs) * 0.5
                span = float(np.max(maxs - mins))
                span = max(span * 1.20, 0.35)
                half = span * 0.5
                return (
                    (float(center[0] - half), float(center[0] + half)),
                    (float(center[1] - half), float(center[1] + half)),
                    (float(center[2] - half), float(center[2] + half)),
                )
            return ((-0.70, 0.70), (-0.70, 0.70), (-0.05, 1.25))

        def set_axes(self, points=None):
            if points is not None and len(points) and self.follow_bounds.isChecked():
                bounds = self.compute_bounds(points)
            else:
                if self.fixed_bounds is None:
                    self.fixed_bounds = self.compute_bounds(points)
                bounds = self.fixed_bounds

            if bounds is not None:
                self.ax.set_xlim(*bounds[0])
                self.ax.set_ylim(*bounds[1])
                self.ax.set_zlim(*bounds[2])
                self.ax.set_box_aspect((1.0, 1.0, 1.0))
            else:
                self.ax.set_xlim(-0.70, 0.70)
                self.ax.set_ylim(-0.70, 0.70)
                self.ax.set_zlim(-0.05, 1.25)
                self.ax.set_box_aspect((1.4, 1.4, 1.3))
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")
            self.ax.view_init(elev=self.view_elev, azim=self.view_azim)

        def update_voxels(self):
            if hasattr(self, "ax"):
                self.view_elev = self.ax.elev
                self.view_azim = self.ax.azim

            t0 = time.perf_counter()
            voxels = runtime_states_to_voxels(self.runtime, self.states())
            state_ms = (time.perf_counter() - t0) * 1000.0

            all_points = voxels["points"].astype(np.float32)
            all_link_ids = voxels["link_ids"].astype(np.int32)
            points, link_ids = self.apply_clip(all_points, all_link_ids)
            self.last_visible_points = points
            plot_points, plot_link_ids = self.sample_points(points, link_ids)
            colors = LINK_COLORS[plot_link_ids % len(LINK_COLORS)] if len(plot_points) else LINK_COLORS[:0]

            t1 = time.perf_counter()
            self.ax.clear()
            self.set_axes(points)
            if len(plot_points):
                self.ax.scatter(
                    plot_points[:, 0],
                    plot_points[:, 1],
                    plot_points[:, 2],
                    c=colors,
                    s=1.0,
                    marker="s",
                    depthshade=False,
                    linewidths=0,
                )
            self.canvas.draw_idle()
            render_ms = (time.perf_counter() - t1) * 1000.0

            crop_text = "on" if self.clip_enabled.isChecked() else "off"
            bounds_text = "follow" if self.follow_bounds.isChecked() else "fixed"
            self.info.setText(
                f"robot: {str(self.pre['robot_name'])}\n"
                f"visible voxels: {len(points)} / {len(all_points)}\n"
                f"plotted points: {len(plot_points)}\n"
                f"state->voxel: {state_ms:.2f} ms\n"
                f"plot update: {render_ms:.2f} ms\n"
                f"crop: {crop_text} | bounds: {bounds_text}"
            )

    app = QtWidgets.QApplication(sys.argv)
    try:
        win = VoxelWindow(runtime, max_points=max_points)
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
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--max-voxels", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    max_points = args.max_points if args.max_voxels is None else args.max_voxels
    run_gui(args.npz, max_points=max_points)


if __name__ == "__main__":
    main()
