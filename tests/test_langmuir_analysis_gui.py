import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from langmuir_analysis_config import SUPPORTED_GEOMETRIES
import langmuir_analysis_gui as gui
from langmuir_analysis_gui import LangmuirAnalysisWindow


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_gui_has_one_fully_populated_tab_per_geometry(
    application, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "langmuir_analysis_gui.LAST_PARAMETERS_PATH",
        tmp_path / "last_parameters.json",
    )
    window = LangmuirAnalysisWindow()

    assert window.tabs.count() == 2
    assert tuple(window.parameter_tabs) == SUPPORTED_GEOMETRIES
    assert "Start Analysis" in window.parameter_tabs["x_line"].start_button.text()
    assert "X-line scan" in window.parameter_tabs["x_line"].start_button.text()
    assert "Start Analysis" in window.parameter_tabs["xy_plane"].start_button.text()
    assert "XY-plane scan" in window.parameter_tabs["xy_plane"].start_button.text()
    assert "interferometer_profile_y_cm" not in window.parameter_tabs["x_line"].values()
    assert "interferometer_profile_y_cm" in window.parameter_tabs["xy_plane"].values()
    assert all(
        "interferometer_physical_constant" not in tab.values()
        for tab in window.parameter_tabs.values()
    )

    xline_tab = window.parameter_tabs["x_line"]
    for tab in window.parameter_tabs.values():
        assert "Scan geometry" not in tab.section_cards
        assert tab.section_cards["Spatial geometry"].property(
            "layoutColumn"
        ) == tab.section_cards["Acquisition geometry"].property("layoutColumn")
    assert xline_tab.values()["ies_method"] == "high_bias_median"
    xline_tab.set_values({"ies_method": "at_vp"})
    assert xline_tab.values()["ies_method"] == "at_vp"
    assert "plasma potential" in xline_tab.editors["ies_method"].currentText()
    assert xline_tab.values()["shot_analysis_mode"] == "individual"
    xline_tab.set_values({"shot_analysis_mode": "average"})
    assert xline_tab.values()["shot_analysis_mode"] == "average"
    assert "Average shots" in xline_tab.editors["shot_analysis_mode"].currentText()
    assert xline_tab.values()["nramps"] == 1
    xline_tab.set_values(
        {"nramps": 3, "ramp_display_mode": "ramp_time_map"}
    )
    assert xline_tab.values()["nramps"] == 3
    assert "X vs ramp/time" in xline_tab.editors[
        "ramp_display_mode"
    ].currentText()
    xy_tab = window.parameter_tabs["xy_plane"]
    assert xy_tab.values()["nramps"] == 1
    xy_tab.set_values({"nramps": 2, "ramp_start_spacing_samples": 2_000})
    assert xy_tab.values()["nramps"] == 2
    assert "ramp_display_mode" not in xy_tab.values()
    assert not xline_tab.editors["isweep_dc_offset_start_index"].isEnabled()
    xline_tab.editors["subtract_dc"].setChecked(True)
    assert xline_tab.editors["isweep_dc_offset_start_index"].isEnabled()

    window.tabs.setCurrentIndex(1)
    assert window._active_geometry() == "xy_plane"
    window.close()


def test_numeric_fields_use_direct_inputs_except_worker_processes(
    application, monkeypatch, tmp_path
):
    monkeypatch.setattr(gui, "LAST_PARAMETERS_PATH", tmp_path / "parameters.json")
    window = LangmuirAnalysisWindow()

    for tab in window.parameter_tabs.values():
        for key, spec in tab.specs.items():
            editor = tab.editors[key]
            if spec.kind in {"int", "float"}:
                assert isinstance(editor, gui.DirectNumericInput)
                assert not isinstance(editor, QtWidgets.QAbstractSpinBox)
            elif spec.kind == "int_pair":
                assert isinstance(editor, gui.IntegerPairEditor)
                assert isinstance(editor.y_value, gui.DirectNumericInput)
                assert isinstance(editor.x_value, gui.DirectNumericInput)
            elif spec.kind == "choice":
                assert isinstance(editor, QtWidgets.QComboBox)
            elif spec.kind == "optional_int":
                assert key == "analysis_processes"
                assert isinstance(editor, QtWidgets.QSpinBox)
                assert editor.specialValueText() == "Automatic"

    nx_editor = window.parameter_tabs["x_line"].editors["nx"]
    nx_editor.setText("17")
    assert window.parameter_tabs["x_line"].values()["nx"] == 17
    nx_editor.setText("")
    with pytest.raises(ValueError, match="must be an integer"):
        window.parameter_tabs["x_line"].values()

    window.close()


def test_sis_configuration_picker_reads_immediate_digitizer_groups(
    application, monkeypatch, tmp_path
):
    import h5py

    hdf5_path = tmp_path / "experiment.hdf5"
    with h5py.File(hdf5_path, "w") as h5_file:
        digitizer_group = h5_file.create_group(
            "Raw data + config"
        ).create_group("SIS crate")
        digitizer_group.create_group("Config B")
        digitizer_group.create_group("Config A")
        digitizer_group.create_dataset(
            "not_a_configuration", data=[1, 2, 3]
        )

    assert gui.discover_sis_configurations(hdf5_path, "SIS crate") == (
        "Config A",
        "Config B",
    )

    monkeypatch.setattr(gui, "LAST_PARAMETERS_PATH", tmp_path / "parameters.json")
    offered_choices = []

    def choose_second(_parent, _title, _label, items, _current, _editable):
        offered_choices.append(tuple(items))
        return "Config B", True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", choose_second)
    window = LangmuirAnalysisWindow()
    for tab in window.parameter_tabs.values():
        tab.set_values(
            {
                "filename": str(hdf5_path),
                "digitizer": "SIS crate",
                "adc": "This value is deliberately ignored by the picker",
                "sis_config_name": "previous value",
            }
        )
        editor = tab.editors["sis_config_name"]
        assert isinstance(editor, gui.SisConfigurationEditor)
        editor.browse_button.click()
        assert editor.value() == "Config B"

    assert offered_choices == [("Config A", "Config B")] * 2
    window.close()


def test_splash_composes_required_title_trace_and_branding(application):
    assert gui.SPLASH_DURATION_MS == 2_000
    assert gui.SPLASH_TITLE == "LAPD Langmuir Analysis Studio"
    assert gui.SPLASH_BACKGROUND_PATH.is_file()
    assert all(path.is_file() for path in gui.SPLASH_LOGO_PATHS.values())

    pixmap = gui.compose_splash_pixmap()
    assert not pixmap.isNull()
    assert pixmap.size() == gui.SPLASH_CANVAS_SIZE

    splash = gui.create_splash_screen(application)
    assert splash.accessibleName() == gui.SPLASH_TITLE
    assert tuple(splash.property("logoPaths")) == tuple(
        str(path) for path in gui.SPLASH_LOGO_PATHS.values()
    )
    title_bounds = splash.property("titleBounds")
    trace_bounds = splash.property("traceBounds")
    logo_bounds = splash.property("logoRailBounds")
    assert title_bounds.bottom() < trace_bounds.top()
    assert trace_bounds.bottom() < logo_bounds.top()
    splash.close()


def test_launch_gui_keeps_main_window_hidden_for_two_seconds(
    application, monkeypatch, tmp_path
):
    monkeypatch.setattr(gui, "LAST_PARAMETERS_PATH", tmp_path / "last_parameters.json")
    window = gui.launch_gui()

    assert not window.isVisible()
    assert window._startup_splash is not None
    assert window._startup_splash.isVisible()
    assert window._startup_timer.isSingleShot()
    assert window._startup_timer.timerType() == QtCore.Qt.PreciseTimer
    assert window._startup_timer.interval() == 2_000
    assert window._startup_timer.isActive()

    QtTest.QTest.mouseClick(window._startup_splash, QtCore.Qt.LeftButton)
    assert window._startup_splash.isVisible()
    assert not window.isVisible()

    window._startup_timer.stop()
    window._startup_timer.timeout.emit()
    assert window.isVisible()
    assert window._startup_splash is None
    assert window._startup_timer is None
    window.close()
