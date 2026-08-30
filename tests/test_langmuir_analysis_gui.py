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
    assert not xline_tab.editors["isweep_dc_offset_start_index"].isEnabled()
    xline_tab.editors["subtract_dc"].setChecked(True)
    assert xline_tab.editors["isweep_dc_offset_start_index"].isEnabled()

    window.tabs.setCurrentIndex(1)
    assert window._active_geometry() == "xy_plane"
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
