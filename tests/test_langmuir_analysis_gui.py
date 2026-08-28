import pytest
from PySide6 import QtWidgets

from langmuir_analysis_config import SUPPORTED_GEOMETRIES
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
    assert not xline_tab.editors["isweep_dc_offset_start_index"].isEnabled()
    xline_tab.editors["subtract_dc"].setChecked(True)
    assert xline_tab.editors["isweep_dc_offset_start_index"].isEnabled()

    window.tabs.setCurrentIndex(1)
    assert window._active_geometry() == "xy_plane"
    window.close()
