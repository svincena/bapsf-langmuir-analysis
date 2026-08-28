"""Modern Qt parameter editor and process monitor for Langmuir analysis."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6 import QtCore, QtWidgets

from langmuir_analysis_config import (
    LAST_PARAMETERS_PATH,
    PARAMETER_SECTIONS,
    SUPPORTED_GEOMETRIES,
    default_parameters,
    load_last_parameters,
    save_last_parameters,
    validate_parameters,
)


GEOMETRY_TITLES = {
    "x_line": "X-line scan",
    "xy_plane": "XY-plane scan",
}


STYLE_SHEET = """
QWidget {
    color: #dce7f4;
    font-family: "Inter", "Avenir Next", "SF Pro Text", sans-serif;
    font-size: 13px;
}
QMainWindow, QWidget#root { background: #0b111b; }
QFrame#hero {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #132239, stop:0.56 #14273a, stop:1 #12342f);
    border: 1px solid #29425b;
    border-radius: 18px;
}
QLabel#eyebrow {
    color: #72dec6;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}
QLabel#title { color: #f7fbff; font-size: 28px; font-weight: 700; }
QLabel#subtitle { color: #a7bacd; font-size: 13px; }
QLabel#statusBadge {
    color: #81e6cf;
    background: #153b38;
    border: 1px solid #297164;
    border-radius: 12px;
    padding: 5px 10px;
    font-weight: 600;
}
QTabWidget::pane { border: 0; top: -1px; }
QTabBar::tab {
    color: #8fa5ba;
    background: #111a27;
    border: 1px solid #26364a;
    border-bottom: 0;
    padding: 12px 25px;
    margin-right: 5px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    font-weight: 650;
}
QTabBar::tab:selected { color: #f1f8ff; background: #182536; }
QTabBar::tab:hover:!selected { color: #d1deeb; background: #162130; }
QScrollArea { border: 0; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QFrame#sectionCard {
    background: #121c2a;
    border: 1px solid #25374c;
    border-radius: 14px;
}
QLabel#sectionTitle { color: #f3f8fd; font-size: 16px; font-weight: 700; }
QLabel#sectionDescription { color: #8399ae; font-size: 11px; }
QLabel#fieldLabel { color: #b6c6d6; font-weight: 550; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    color: #edf5fc;
    background: #0c1521;
    border: 1px solid #30445b;
    border-radius: 8px;
    min-height: 32px;
    padding: 0 9px;
    selection-background-color: #2b7a78;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #55cdb7;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled { color: #617488; background: #111923; }
QComboBox::drop-down { border: 0; width: 28px; }
QComboBox QAbstractItemView {
    color: #e9f2fa; background: #111c29; border: 1px solid #34495f;
    selection-background-color: #23645c;
}
QCheckBox { color: #b9c9d8; spacing: 9px; }
QCheckBox::indicator {
    width: 34px; height: 19px; border-radius: 10px;
    background: #28384a; border: 1px solid #3a4e63;
}
QCheckBox::indicator:checked {
    background: #34a995; border: 1px solid #64d3bf;
}
QPushButton {
    color: #dce8f3; background: #1b2a3b; border: 1px solid #344a61;
    border-radius: 9px; min-height: 34px; padding: 0 14px; font-weight: 600;
}
QPushButton:hover { background: #24384c; border-color: #4a657e; }
QPushButton:pressed { background: #162333; }
QPushButton:disabled { color: #607386; background: #17212d; border-color: #273646; }
QPushButton#browseButton { min-width: 36px; max-width: 36px; padding: 0; }
QPushButton#startButton {
    color: #071411;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #60dfc4, stop:1 #80e6b4);
    border: 1px solid #a0f2d2;
    border-radius: 12px;
    min-height: 52px;
    padding: 0 28px;
    font-size: 15px;
    font-weight: 750;
}
QPushButton#startButton:hover { background: #8af0d4; }
QPushButton#startButton:disabled { color: #728b84; background: #29413d; border-color: #35544e; }
QFrame#consoleCard { background: #0e1722; border: 1px solid #26384b; border-radius: 13px; }
QPlainTextEdit {
    color: #bad0df; background: #09111a; border: 0; border-radius: 8px;
    font-family: "SFMono-Regular", "Menlo", monospace; font-size: 11px;
    padding: 8px;
}
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #34485c; min-height: 30px; border-radius: 5px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { color: #eaf3fa; background: #152333; border: 1px solid #3a526a; padding: 5px; }
"""


class PathEditor(QtWidgets.QWidget):
    """Line editor with a native file or directory picker."""

    def __init__(self, value, *, directory=False, parent=None):
        super().__init__(parent)
        self.directory = directory
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.line_edit = QtWidgets.QLineEdit(str(value))
        self.line_edit.setToolTip(str(value))
        self.line_edit.textChanged.connect(self.line_edit.setToolTip)
        self.browse_button = QtWidgets.QPushButton("…")
        self.browse_button.setObjectName("browseButton")
        self.browse_button.setToolTip(
            "Choose a folder" if directory else "Choose an HDF5 experiment file"
        )
        self.browse_button.clicked.connect(self._browse)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.browse_button)

    def _browse(self):
        current = Path(self.line_edit.text()).expanduser()
        start = str(current if current.is_dir() else current.parent)
        if self.directory:
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Choose output folder", start
            )
        else:
            selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Choose LAPD experiment file",
                start,
                "HDF5 files (*.hdf5 *.h5);;All files (*)",
            )
        if selected:
            self.line_edit.setText(selected)

    def value(self):
        return self.line_edit.text().strip()

    def set_value(self, value):
        self.line_edit.setText(str(value))


class IntegerPairEditor(QtWidgets.QWidget):
    """Compact y/x integer-pair editor for two-dimensional neighborhoods."""

    def __init__(self, value, minimum, maximum, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.y_value = QtWidgets.QSpinBox()
        self.x_value = QtWidgets.QSpinBox()
        for label, editor in (("Y", self.y_value), ("X", self.x_value)):
            editor.setRange(int(minimum), int(maximum))
            editor.setPrefix(f"{label}  ")
            layout.addWidget(editor, 1)
        self.set_value(value)

    def value(self):
        return self.y_value.value(), self.x_value.value()

    def set_value(self, value):
        self.y_value.setValue(int(value[0]))
        self.x_value.setValue(int(value[1]))


def _make_editor(spec):
    if spec.kind == "bool":
        editor = QtWidgets.QCheckBox("Enabled")
        editor.setChecked(bool(spec.default))
    elif spec.kind == "int":
        editor = QtWidgets.QSpinBox()
        editor.setRange(int(spec.minimum), int(spec.maximum))
        editor.setSingleStep(max(1, int(spec.step)))
        editor.setValue(int(spec.default))
    elif spec.kind == "optional_int":
        editor = QtWidgets.QSpinBox()
        editor.setRange(int(spec.minimum), int(spec.maximum))
        editor.setSpecialValueText("Automatic")
        editor.setValue(0 if spec.default is None else int(spec.default))
    elif spec.kind == "float":
        editor = QtWidgets.QDoubleSpinBox()
        editor.setRange(float(spec.minimum), float(spec.maximum))
        editor.setDecimals(spec.decimals)
        editor.setSingleStep(float(spec.step))
        editor.setValue(float(spec.default))
        editor.setGroupSeparatorShown(True)
    elif spec.kind == "choice":
        editor = QtWidgets.QComboBox()
        editor.addItems(spec.choices)
        editor.setCurrentText(str(spec.default))
    elif spec.kind in {"file", "directory"}:
        editor = PathEditor(spec.default, directory=spec.kind == "directory")
    elif spec.kind == "int_pair":
        editor = IntegerPairEditor(spec.default, spec.minimum, spec.maximum)
    else:
        editor = QtWidgets.QLineEdit(str(spec.default))

    editor.setToolTip(spec.description)
    return editor


def _editor_value(editor, spec):
    if isinstance(editor, (PathEditor, IntegerPairEditor)):
        return editor.value()
    if spec.kind == "bool":
        return editor.isChecked()
    if spec.kind == "optional_int":
        return None if editor.value() == 0 else editor.value()
    if spec.kind in {"int", "float"}:
        return editor.value()
    if spec.kind == "choice":
        return editor.currentText()
    return editor.text().strip()


def _set_editor_value(editor, spec, value):
    if isinstance(editor, (PathEditor, IntegerPairEditor)):
        editor.set_value(value)
    elif spec.kind == "bool":
        editor.setChecked(bool(value))
    elif spec.kind == "optional_int":
        editor.setValue(0 if value is None else int(value))
    elif spec.kind == "int":
        editor.setValue(int(value))
    elif spec.kind == "float":
        editor.setValue(float(value))
    elif spec.kind == "choice":
        editor.setCurrentText(str(value))
    else:
        editor.setText(str(value))


class ParameterTab(QtWidgets.QWidget):
    start_requested = QtCore.Signal(str)

    def __init__(self, geometry, parent=None):
        super().__init__(parent)
        self.geometry = geometry
        self.editors = {}
        self.specs = {}

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 12, 0, 0)
        root_layout.setSpacing(12)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_body = QtWidgets.QWidget()
        card_columns_layout = QtWidgets.QHBoxLayout(scroll_body)
        card_columns_layout.setContentsMargins(0, 0, 4, 6)
        card_columns_layout.setSpacing(12)
        card_columns = [QtWidgets.QVBoxLayout(), QtWidgets.QVBoxLayout()]
        for column in card_columns:
            column.setSpacing(12)
            card_columns_layout.addLayout(column, 1)

        for index, section in enumerate(PARAMETER_SECTIONS[geometry]):
            card = self._build_section_card(section)
            card_columns[index % 2].addWidget(card)
        for column in card_columns:
            column.addStretch(1)
        self._wire_dependencies()
        scroll.setWidget(scroll_body)
        root_layout.addWidget(scroll, 1)

        actions = QtWidgets.QHBoxLayout()
        actions.setContentsMargins(4, 0, 4, 0)
        reset_button = QtWidgets.QPushButton("Restore this tab’s defaults")
        reset_button.clicked.connect(self.restore_defaults)
        self.start_button = QtWidgets.QPushButton(
            f"Start Analysis  ·  {GEOMETRY_TITLES[geometry]}  →"
        )
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(
            lambda: self.start_requested.emit(self.geometry)
        )
        actions.addWidget(reset_button)
        actions.addStretch(1)
        actions.addWidget(self.start_button)
        root_layout.addLayout(actions)

    def _build_section_card(self, section):
        card = QtWidgets.QFrame()
        card.setObjectName("sectionCard")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(16, 15, 16, 17)
        card_layout.setSpacing(5)

        title = QtWidgets.QLabel(section.title)
        title.setObjectName("sectionTitle")
        description = QtWidgets.QLabel(section.description)
        description.setObjectName("sectionDescription")
        description.setWordWrap(True)
        card_layout.addWidget(title)
        card_layout.addWidget(description)
        card_layout.addSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        for spec in section.parameters:
            label_text = spec.label + (f"  [{spec.unit}]" if spec.unit else "")
            label = QtWidgets.QLabel(label_text)
            label.setObjectName("fieldLabel")
            label.setToolTip(spec.description)
            editor = _make_editor(spec)
            self.editors[spec.key] = editor
            self.specs[spec.key] = spec
            form.addRow(label, editor)
        card_layout.addLayout(form)
        return card

    def _wire_dependencies(self):
        """Dim controls that have no effect while their feature is disabled."""
        dependencies = {
            "subtract_dc": (
                "isweep_dc_offset_start_index",
                "isweep_dc_offset_end_index",
            ),
            "calculate_shape_factor": tuple(
                key
                for key in (
                    "f_microwave_GHz",
                    "N_passes",
                    "interferometer_phase_rad",
                    "interferometer_physical_constant",
                    "interferometer_profile_y_cm",
                )
                if key in self.editors
            ),
            "parallel_analysis": ("analysis_processes",),
            "enable_vp_spike_rejection": tuple(
                key
                for key in (
                    "vp_spike_half_window",
                    "vp_spike_threshold_V",
                    "vp_replace_flagged_with_local_interp",
                    "vp_replace_flagged_with_local_median",
                )
                if key in self.editors
            ),
            "enable_neighbor_smoothing": (
                "neighbor_smooth_half_window",
                "neighbor_smooth_sigma",
            ),
        }
        for toggle_key, dependent_keys in dependencies.items():
            toggle = self.editors[toggle_key]

            def update(enabled, keys=dependent_keys):
                for key in keys:
                    self.editors[key].setEnabled(enabled)

            toggle.toggled.connect(update)
            update(toggle.isChecked())

    def values(self):
        return {
            key: _editor_value(self.editors[key], spec)
            for key, spec in self.specs.items()
        }

    def set_values(self, values):
        for key, value in values.items():
            if key in self.editors:
                _set_editor_value(self.editors[key], self.specs[key], value)

    def restore_defaults(self):
        self.set_values(default_parameters(self.geometry))

    def set_running(self, running):
        self.start_button.setDisabled(running)


class LangmuirAnalysisWindow(QtWidgets.QMainWindow):
    """Tabbed analysis configuration window with a live subprocess console."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Langmuir Analysis Studio")
        self.resize(1280, 900)
        self.setMinimumSize(940, 680)
        self.process = None

        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(12)
        outer.addWidget(self._build_hero())

        self.tabs = QtWidgets.QTabWidget()
        self.parameter_tabs = {}
        for geometry in SUPPORTED_GEOMETRIES:
            tab = ParameterTab(geometry)
            tab.start_requested.connect(self.start_analysis)
            self.parameter_tabs[geometry] = tab
            self.tabs.addTab(tab, GEOMETRY_TITLES[geometry])
        self.tabs.currentChanged.connect(self._active_tab_changed)

        self.console_card = self._build_console()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.console_card)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([680, 150])
        outer.addWidget(splitter, 1)

        saved, active_geometry, warning = load_last_parameters(LAST_PARAMETERS_PATH)
        for geometry, values in saved.items():
            self.parameter_tabs[geometry].set_values(values)
        self.tabs.setCurrentIndex(SUPPORTED_GEOMETRIES.index(active_geometry))
        if warning:
            self._append_console(warning)
        elif LAST_PARAMETERS_PATH.exists():
            self._append_console(f"Loaded settings from {LAST_PARAMETERS_PATH.name}.")
        else:
            self._append_console("No saved settings found; built-in defaults loaded.")
        self._active_tab_changed(self.tabs.currentIndex())

    def _build_hero(self):
        hero = QtWidgets.QFrame()
        hero.setObjectName("hero")
        layout = QtWidgets.QHBoxLayout(hero)
        layout.setContentsMargins(22, 18, 18, 18)
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(3)
        eyebrow = QtWidgets.QLabel("LAPD  •  LANGMUIR PROBE WORKFLOW")
        eyebrow.setObjectName("eyebrow")
        title = QtWidgets.QLabel("Langmuir Analysis Studio")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Configure acquisition, I–V physics, spatial cleanup, and interferometer calibration in one place."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        text_layout.addWidget(eyebrow)
        text_layout.addWidget(title)
        text_layout.addWidget(subtitle)
        layout.addLayout(text_layout, 1)

        controls = QtWidgets.QVBoxLayout()
        controls.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.status_badge = QtWidgets.QLabel("Ready")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setAlignment(QtCore.Qt.AlignCenter)
        save_button = QtWidgets.QPushButton("Save parameters")
        save_button.clicked.connect(self.save_parameters)
        controls.addWidget(self.status_badge)
        controls.addWidget(save_button)
        layout.addLayout(controls)
        return hero

    def _build_console(self):
        card = QtWidgets.QFrame()
        card.setObjectName("consoleCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(12, 9, 12, 12)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Analysis output")
        title.setObjectName("sectionTitle")
        self.stop_button = QtWidgets.QPushButton("Stop analysis")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_analysis)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.stop_button)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(10_000)
        layout.addLayout(header)
        layout.addWidget(self.console, 1)
        return card

    def _active_geometry(self):
        return SUPPORTED_GEOMETRIES[self.tabs.currentIndex()]

    def _active_tab_changed(self, _index):
        if self.process is None:
            geometry = self._active_geometry()
            self.status_badge.setText(f"Ready · {GEOMETRY_TITLES[geometry]}")

    def _all_values(self):
        return {geometry: tab.values() for geometry, tab in self.parameter_tabs.items()}

    def _append_console(self, text):
        self.console.appendPlainText(str(text).rstrip())
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @QtCore.Slot()
    def save_parameters(self):
        try:
            save_last_parameters(
                self._all_values(), self._active_geometry(), LAST_PARAMETERS_PATH
            )
        except (TypeError, ValueError, OSError) as error:
            QtWidgets.QMessageBox.critical(
                self, "Could not save parameters", str(error)
            )
            return False
        self._append_console(f"Saved both tabs to {LAST_PARAMETERS_PATH.name}.")
        return True

    @QtCore.Slot(str)
    def start_analysis(self, _button_geometry):
        if self.process is not None:
            return
        # Geometry always comes from the open tab, including after keyboard or
        # programmatic tab changes.
        active_geometry = self._active_geometry()
        try:
            validate_parameters(
                active_geometry,
                self.parameter_tabs[active_geometry].values(),
                require_input_file=True,
            )
            save_last_parameters(
                self._all_values(), active_geometry, LAST_PARAMETERS_PATH
            )
        except (TypeError, ValueError, OSError) as error:
            QtWidgets.QMessageBox.warning(self, "Check analysis parameters", str(error))
            self._append_console(f"Parameter check failed: {error}")
            return

        self.console.clear()
        self._append_console(
            f"Starting {GEOMETRY_TITLES[active_geometry]} analysis…\n"
            f"Parameters: {LAST_PARAMETERS_PATH}"
        )
        process = QtCore.QProcess(self)
        process.setWorkingDirectory(str(Path(__file__).resolve().parent))
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_process_output)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                "-u",
                str(Path(__file__).with_name("langmuir_analysis.py")),
                "--run-parameters",
                str(LAST_PARAMETERS_PATH),
                "--geometry",
                active_geometry,
            ]
        )
        self.process = process
        self._set_running(True, active_geometry)
        process.start()

    def _set_running(self, running, geometry=None):
        for tab in self.parameter_tabs.values():
            tab.set_running(running)
        self.tabs.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if running:
            self.status_badge.setText(f"Running · {GEOMETRY_TITLES[geometry]}")
        else:
            self._active_tab_changed(self.tabs.currentIndex())

    @QtCore.Slot()
    def _read_process_output(self):
        if self.process is None:
            return
        raw = bytes(self.process.readAllStandardOutput())
        if raw:
            self._append_console(raw.decode("utf-8", errors="replace"))

    @QtCore.Slot(int, QtCore.QProcess.ExitStatus)
    def _process_finished(self, exit_code, exit_status):
        self._read_process_output()
        crashed = exit_status == QtCore.QProcess.CrashExit
        outcome = "crashed" if crashed else f"finished with exit code {exit_code}"
        self._append_console(f"\nAnalysis {outcome}.")
        self.process = None
        self._set_running(False)

    @QtCore.Slot(QtCore.QProcess.ProcessError)
    def _process_error(self, error):
        if self.process is not None:
            self._append_console(
                f"Process error: {self.process.errorString()} ({error})"
            )

    @QtCore.Slot()
    def stop_analysis(self):
        if self.process is None:
            return
        self._append_console("Stopping analysis…")
        process = self.process
        process.terminate()
        QtCore.QTimer.singleShot(
            3_000,
            lambda: (
                process.kill()
                if process.state() != QtCore.QProcess.NotRunning
                else None
            ),
        )

    def closeEvent(self, event):
        if self.process is not None:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Analysis is still running",
                "Stop the analysis and close the window?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.process.kill()
            self.process.waitForFinished(2_000)
        try:
            save_last_parameters(
                self._all_values(), self._active_geometry(), LAST_PARAMETERS_PATH
            )
        except (TypeError, ValueError, OSError):
            pass
        event.accept()


def launch_gui():
    """Create and run the desktop application."""
    application = QtWidgets.QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QtWidgets.QApplication(sys.argv[:1])
    application.setApplicationName("Langmuir Analysis Studio")
    application.setStyle("Fusion")
    application.setStyleSheet(STYLE_SHEET)
    window = LangmuirAnalysisWindow()
    window.show()
    if owns_application:
        return application.exec()
    return window


if __name__ == "__main__":
    raise SystemExit(launch_gui())
