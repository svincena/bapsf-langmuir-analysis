# Load the GUI's dateutil workaround before importing the Qt binding.
import langmuir_analysis_gui as gui  # isort: skip

import numpy as np
import pytest
from PySide6 import QtWidgets

import langmuir_sweep_selector as selector
from langmuir_analysis_config import default_parameters


def test_selector_import_keeps_scientific_imports_lazy_and_supports_later_bapsflib():
    import subprocess
    import sys

    script = (
        "import sys; import langmuir_sweep_selector; "
        "assert 'matplotlib' not in sys.modules; assert 'bapsflib' not in sys.modules; "
        "import bapsflib"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="module")
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def preview(monkeypatch):
    values = default_parameters("xy_plane")
    values.update(
        nx=5,
        ny=3,
        nshots=3,
        data_offset=100,
        nt_full=1000,
        dt_s=1e-6,
        y_min_cm=-20,
        y_max_cm=20,
        temporal_metadata_source="manual",
        sweep_start_index=100,
        sweep_end_index=300,
        isat_end_index=4,
        sg_smooth_bins=5,
        sg_smooth_order=1,
        subtract_dc=False,
    )
    calls = []

    def read(values, shot_number):
        calls.append(shot_number)
        return {
            "time_s": np.arange(1000) * values["dt_s"],
            "voltage_V": np.arange(1000) * 0.1,
            "current_A": np.arange(1000) * 0.001,
            "dt_s": values["dt_s"],
            "nt_full": 1000,
            "shot_number": shot_number,
        }

    monkeypatch.setattr(selector, "read_sweep_preview_trace", read)
    return values, calls


def test_shared_zoom_pan_home_and_navigation_history(application, preview):
    values, calls = preview
    dialog = selector.SweepSelectionDialog("xy_plane", values)
    assert calls == [123]
    assert dialog.voltage_axis.get_shared_x_axes().joined(
        dialog.voltage_axis, dialog.current_axis
    )
    dialog.voltage_axis.set_xlim(110.5, 209.5)
    dialog.toolbar.push_current()
    assert dialog.current_axis.get_xlim() == pytest.approx((110.5, 209.5))
    assert dialog.visible_samples() == (111, 210)
    # Moving either plot translates the same interval for both signals.
    dialog.current_axis.set_xlim(130, 230)
    dialog.toolbar.push_current()
    assert dialog.visible_samples() == (130, 230)
    assert dialog.voltage_axis.get_ylim()[1] < 25
    dialog.home_button.click()
    assert dialog.visible_samples() == (0, 999)
    dialog.toolbar.back()
    assert dialog.visible_samples() == (130, 230)
    dialog.toolbar.forward()
    assert dialog.visible_samples() == (0, 999)
    dialog.toolbar.home()
    assert dialog.visible_samples() == (0, 999)
    dialog.sweep_button.click()
    assert dialog.visible_samples() == (100, 300)
    dialog.close()


def test_only_approval_accepts_valid_visible_interval(application, preview):
    values, _calls = preview
    dialog = selector.SweepSelectionDialog("xy_plane", values)
    dialog.voltage_axis.set_xlim(120, 220)
    assert values["sweep_start_index"] == 100
    dialog.approve_button.click()
    assert dialog.result() == QtWidgets.QDialog.Accepted
    assert dialog.selected_samples == (120, 220)
    assert values["sweep_start_index"] == 100
    dialog.close()
    canceled = selector.SweepSelectionDialog("xy_plane", values)
    canceled.voltage_axis.set_xlim(120, 220)
    canceled.reject()
    assert canceled.result() == QtWidgets.QDialog.Rejected
    assert canceled.selected_samples is None
    canceled.close()


def test_invalid_zoom_is_rejected_and_home_allows_retry(application, preview):
    values, _calls = preview
    dialog = selector.SweepSelectionDialog("xy_plane", values)
    dialog.voltage_axis.set_xlim(120, 121)
    dialog.approve_button.click()
    assert dialog.selected_samples is None
    assert "Isat window" in dialog.error_label.text()
    dialog.reset_view()
    dialog.voltage_axis.set_xlim(120, 220)
    dialog.approve_button.click()
    assert dialog.selected_samples == (120, 220)
    dialog.close()


def test_optional_location_requires_reload_and_maps_descending_y(application, preview):
    values, calls = preview
    dialog = selector.SweepSelectionDialog("xy_plane", values)
    dialog.location_editors["y"].set_value(0)
    dialog.location_editors["x"].set_value(1)
    dialog.location_editors["shot"].set_value(2)
    assert not dialog.approve_button.isEnabled()
    dialog.load_button.click()
    assert calls[-1] == 136
    assert dialog.approve_button.isEnabled()
    assert "Y = -20 cm" in dialog.location_label.text()
    dialog.center_button.click()
    assert calls[-1] == 123
    assert dialog.visible_samples() == (0, 999)
    dialog.close()


def test_read_failure_disables_approval_and_shows_actionable_error(
    application, preview, monkeypatch
):
    values, _calls = preview

    def unavailable(*_args):
        raise ValueError("Current channel does not contain this shot.")

    monkeypatch.setattr(selector, "read_sweep_preview_trace", unavailable)
    dialog = selector.SweepSelectionDialog("xy_plane", values)
    assert not dialog.approve_button.isEnabled()
    assert "Current channel" in dialog.error_label.text()
    assert QtWidgets.QApplication.overrideCursor() is None
    dialog.close()


@pytest.mark.parametrize("accepted", [False, True])
def test_tab_applies_only_approved_limits_and_preserves_time_units(
    application, monkeypatch, accepted
):
    monkeypatch.setattr(gui.ParameterTab, "reconcile_bmotion_geometry", lambda self: ())
    monkeypatch.setattr(
        gui.ParameterTab, "reconcile_sis_acquisition_metadata", lambda self: ()
    )
    tab = gui.ParameterTab("x_line")
    tab.set_values(
        {
            "temporal_metadata_source": "manual",
            "dt_s": 40e-9,
            "sweep_start_input_unit": "microseconds",
            "sweep_start_time_s": 100e-6,
            "sweep_end_input_unit": "milliseconds",
            "sweep_end_time_s": 300e-6,
        }
    )
    before = tab.values()

    class FakeDialog:
        def __init__(self, geometry, values, parent):
            self.values = values
            self.selected_samples = (3000, 8000)

        def exec(self):
            return (
                QtWidgets.QDialog.Accepted if accepted else QtWidgets.QDialog.Rejected
            )

        def deleteLater(self):
            pass

    monkeypatch.setattr(selector, "SweepSelectionDialog", FakeDialog)
    tab.choose_sweep_button.click()
    if accepted:
        assert tab.values()["sweep_start_index"] == 3000
        assert tab.values()["sweep_start_time_s"] == pytest.approx(120e-6)
        assert tab.values()["sweep_end_index"] == 8000
        assert tab.values()["sweep_end_time_s"] == pytest.approx(320e-6)
        assert tab.editors["sweep_start_index"].units.currentText() == "µs"
        assert tab.editors["sweep_end_index"].units.currentText() == "ms"
    else:
        assert tab.values() == before
    tab.set_running(True)
    assert not tab.choose_sweep_button.isEnabled()
    tab.close()
