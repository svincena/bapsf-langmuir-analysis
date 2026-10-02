"""Interactive single-shot sweep selection, imported only when requested."""

# Initialize the GUI's dateutil workaround before Qt installs its import hook.
from langmuir_analysis_gui import DirectNumericInput  # isort: skip
from PySide6 import QtCore, QtWidgets

from langmuir_analysis_config import (
    SWEEP_TIMING_FIELDS,
    preview_shot_number,
    time_to_sample,
    validate_parameters,
)
from langmuir_analysis_metadata import read_sweep_preview_trace


class SweepSelectionDialog(QtWidgets.QDialog):
    """Approve the visible shared time interval as inclusive sweep limits."""

    def __init__(self, geometry, values, parent=None):
        super().__init__(parent)
        # FigureCanvas avoids pyplot and leaves the worker's backend untouched.
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        self.geometry = geometry
        self.values = dict(values)
        self.trace = None
        self.selected_samples = None
        self.setWindowTitle("Choose sweep limits")
        self.resize(1100, 800)
        layout = QtWidgets.QVBoxLayout(self)
        instructions = QtWidgets.QLabel(
            "Zoom or pan either graph; both share the same time limits. "
            "Home restores the full trace. Dashed lines mark the current sweep. "
            "Click Use visible interval to approve the sweep."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        locations = QtWidgets.QHBoxLayout()
        self.location_editors = {}
        for key, label, count in (
            ("x", "X index", values["nx"]),
            ("y", "Y index", values.get("ny", 1) if geometry == "xy_plane" else 1),
            ("shot", "Shot index", values["nshots"]),
        ):
            if key == "y" and geometry != "xy_plane":
                continue
            editor = DirectNumericInput(
                count // 2, numeric_kind="int", minimum=0, maximum=count - 1
            )
            editor.setMaximumWidth(95)
            editor.setToolTip(
                "Zero-based index. Y coordinates increase from negative to positive Y."
            )
            locations.addWidget(QtWidgets.QLabel(label))
            locations.addWidget(editor)
            self.location_editors[key] = editor
        self.load_button = QtWidgets.QPushButton("Load location")
        self.load_button.clicked.connect(self.load_trace)
        locations.addWidget(self.load_button)
        self.center_button = QtWidgets.QPushButton("Center of scan")
        self.center_button.clicked.connect(self.load_center)
        locations.addWidget(self.center_button)
        locations.addStretch(1)
        layout.addLayout(locations)
        self.location_label = QtWidgets.QLabel()
        self.location_label.setWordWrap(True)
        layout.addWidget(self.location_label)

        self.figure = Figure(figsize=(10, 6), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.voltage_axis, self.current_axis = self.figure.subplots(2, 1, sharex=True)
        self.axes = (self.voltage_axis, self.current_axis)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        navigation = QtWidgets.QHBoxLayout()
        self.home_button = QtWidgets.QPushButton("Home / full trace")
        self.home_button.clicked.connect(self.reset_view)
        navigation.addWidget(self.home_button)
        self.sweep_button = QtWidgets.QPushButton("Show current sweep")
        self.sweep_button.clicked.connect(self.show_current_sweep)
        navigation.addWidget(self.sweep_button)
        navigation.addStretch(1)
        layout.addLayout(navigation)
        layout.addWidget(self.canvas, 1)
        self.selection_label = QtWidgets.QLabel()
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.approve_button = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.approve_button.setText("Use visible interval")
        self.approve_button.setEnabled(False)
        buttons.accepted.connect(self.approve_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for editor in self.location_editors.values():
            editor.textChanged.connect(self._location_edited)
        self.load_trace()

    def _location_edited(self, *_args):
        self.approve_button.setEnabled(False)
        self.error_label.setText("Click Load location to preview the selected shot.")

    def load_center(self):
        for key, count in (
            ("x", self.values["nx"]),
            ("y", self.values.get("ny", 1)),
            ("shot", self.values["nshots"]),
        ):
            if key in self.location_editors:
                self.location_editors[key].set_value(count // 2)
        self.load_trace()

    def load_trace(self):
        self.approve_button.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            indices = {
                key: editor.value() for key, editor in self.location_editors.items()
            }
            shot_number = preview_shot_number(
                self.geometry,
                self.values,
                indices["x"],
                indices["shot"],
                indices.get("y", 0),
            )
            trace = read_sweep_preview_trace(self.values, shot_number)
            self.values.update(dt_s=trace["dt_s"], nt_full=trace["nt_full"])
            self.trace = trace
            x_cm = self._coordinate("x", indices["x"])
            location = f"X = {x_cm:g} cm"
            if self.geometry == "xy_plane":
                location += f", Y = {self._coordinate('y', indices['y']):g} cm"
            self.location_label.setText(
                f"{location}; shot index {indices['shot']}; acquisition shot {shot_number}. "
                "Full, unsmoothed traces with the configured scaling, polarity, and baseline."
            )
            self._draw_trace()
            self.error_label.clear()
            self.approve_button.setEnabled(True)
        except (ValueError, OSError) as error:
            self.error_label.setText(str(error))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _coordinate(self, axis, index):
        minimum, maximum = self.values[f"{axis}_min_cm"], self.values[f"{axis}_max_cm"]
        count = self.values[f"n{axis}"]
        return (
            minimum
            if count == 1
            else minimum + index * (maximum - minimum) / (count - 1)
        )

    def _draw_trace(self):
        for axis, signal, label, color in (
            (self.voltage_axis, self.trace["voltage_V"], "Voltage [V]", "tab:blue"),
            (self.current_axis, self.trace["current_A"], "Current [A]", "tab:orange"),
        ):
            axis.clear()
            axis.plot(self.trace["time_s"] * 1e6, signal, color=color, linewidth=0.8)
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
            for key in ("sweep_start_index", "sweep_end_index"):
                time_key = SWEEP_TIMING_FIELDS[key][1]
                unit_key = SWEEP_TIMING_FIELDS[key][0]
                time_s = (
                    self.values[key] * self.trace["dt_s"]
                    if self.values[unit_key] == "samples"
                    else self.values[time_key]
                )
                axis.axvline(time_s * 1e6, linestyle="--", color="0.45", linewidth=1)
            axis.callbacks.connect("xlim_changed", self._view_changed)
        self.current_axis.set_xlabel("Time from trace start [µs]")
        # Each loaded location gets its own navigation history and full-trace home.
        self.toolbar.update()
        self.reset_view()

    def reset_view(self):
        if self.trace is None:
            return
        self.voltage_axis.set_xlim(0, self.trace["time_s"][-1] * 1e6)
        self._view_changed()
        self.toolbar.push_current()
        self.canvas.draw_idle()

    def show_current_sweep(self):
        if self.trace is None:
            return
        limits = []
        for key in ("sweep_start_index", "sweep_end_index"):
            unit_key, time_key = SWEEP_TIMING_FIELDS[key]
            limits.append(
                self.values[key] * self.trace["dt_s"]
                if self.values[unit_key] == "samples"
                else self.values[time_key]
            )
        self.voltage_axis.set_xlim(limits[0] * 1e6, limits[1] * 1e6)
        self._view_changed()
        self.toolbar.push_current()
        self.canvas.draw_idle()

    def visible_samples(self):
        if self.trace is None:
            raise ValueError("Load a preview trace first.")
        low, high = sorted(self.voltage_axis.get_xlim())
        last_time = self.trace["time_s"][-1]
        low_s, high_s = max(0.0, low * 1e-6), min(last_time, high * 1e-6)
        if low_s >= high_s:
            raise ValueError("The visible interval must lie inside the acquired trace.")
        start, end = (
            time_to_sample(low_s, self.trace["dt_s"]),
            time_to_sample(high_s, self.trace["dt_s"]),
        )
        if start >= end:
            raise ValueError("Zoom out to include at least two samples.")
        return start, end

    def _view_changed(self, *_args):
        if self.trace is None:
            return
        import numpy as np

        low, high = sorted(self.voltage_axis.get_xlim())
        visible = (self.trace["time_s"] * 1e6 >= low) & (
            self.trace["time_s"] * 1e6 <= high
        )
        for axis, signal in zip(
            self.axes, (self.trace["voltage_V"], self.trace["current_A"])
        ):
            finite = signal[visible & np.isfinite(signal)]
            if finite.size:
                bottom, top = np.min(finite), np.max(finite)
                margin = max((top - bottom) * 0.05, abs(top) * 0.01, 1e-9)
                axis.set_ylim(bottom - margin, top + margin)
        try:
            start, end = self.visible_samples()
            self.selection_label.setText(
                f"Sweep start: {start} ({start * self.trace['dt_s'] * 1e6:.12g} µs); "
                f"Sweep end: {end} ({end * self.trace['dt_s'] * 1e6:.12g} µs), inclusive. "
                f"{end - start + 1} samples."
            )
        except ValueError as error:
            self.selection_label.setText(str(error))

    def approve_selection(self):
        if not self.approve_button.isEnabled():
            return
        try:
            start, end = self.visible_samples()
            candidate = dict(self.values)
            for key, index in (("sweep_start_index", start), ("sweep_end_index", end)):
                candidate[key] = index
                candidate[SWEEP_TIMING_FIELDS[key][1]] = index * self.trace["dt_s"]
            validate_parameters(self.geometry, candidate)
        except ValueError as error:
            self.error_label.setText(str(error))
            return
        self.selected_samples = (start, end)
        self.accept()
