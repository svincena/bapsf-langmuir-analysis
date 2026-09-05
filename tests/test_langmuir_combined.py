import astropy.units as u
import numpy as np
import pytest

import langmuir_analysis as analysis
from langmuir_analysis_config import default_parameters


def test_run_analysis_dispatches_selected_geometry(monkeypatch):
    called = []
    monkeypatch.setattr(
        analysis,
        "run_xline_analysis",
        lambda: called.append("x_line"),
    )
    monkeypatch.setattr(
        analysis,
        "run_xy_analysis",
        lambda: called.append("xy_plane"),
    )

    for geometry in ("x_line", "xy_plane"):
        analysis.run_analysis(geometry, default_parameters(geometry))

    assert called == ["x_line", "xy_plane"]


def test_run_analysis_rejects_unknown_geometry():
    with pytest.raises(ValueError, match="analysis_geometry"):
        analysis.run_analysis("radial", {})


def test_configure_xy_analysis_derives_spatial_and_shot_shapes():
    values = default_parameters("xy_plane")
    configured = analysis.configure_analysis("xy_plane", values)

    assert configured == values
    assert analysis.x.shape == (values["nx"],)
    assert analysis.y.shape == (values["ny"],)
    assert analysis.X.shape == (values["ny"], values["nx"])
    assert analysis.Y.shape == (values["ny"], values["nx"])
    assert analysis.nt == (values["sweep_end_index"] - values["sweep_start_index"] + 1)
    assert analysis.n_expected_shots == (values["ny"] * values["nx"] * values["nshots"])
    assert analysis.shot_analysis_mode == "individual"
    expected_scaling = (
        analysis.INTERFEROMETER_PHYSICAL_COEFFICIENT
        * values["f_microwave_GHz"]
        * u.GHz
        * values["interferometer_phase_rad"]
        * u.rad
        / values["N_passes"]
    )
    assert u.allclose(analysis.interferometer_scaling, expected_scaling)
    source_path = analysis.Path(values["filename"])
    assert analysis.diagnostic_plot_output_dir == source_path.with_name(
        f"{source_path.stem}_Langmuir_diagnostic_plots"
    )


def test_diagnostic_plot_directory_is_named_for_and_beside_source_file(tmp_path):
    source_path = tmp_path / "experiment run.hdf5"

    assert analysis.diagnostic_plot_directory(source_path) == (
        tmp_path / "experiment run_Langmuir_diagnostic_plots"
    )


def test_prepare_trace_for_analysis_selects_one_individual_shot():
    voltage = np.array([[[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]]])
    current = 2.0 * voltage - 1.0

    selected_voltage, selected_current = analysis.prepare_trace_for_analysis(
        voltage,
        current,
        (0, 1),
        "individual",
        0.25,
    )

    assert np.array_equal(selected_voltage, voltage[0, 1])
    assert np.array_equal(selected_current, current[0, 1])


def test_prepare_trace_for_analysis_averages_shots_in_voltage_bins():
    voltage = np.array([[[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]]])
    current = np.array([[[0.0, 2.0, 4.0], [2.0, 4.0, 6.0]]])

    averaged_voltage, averaged_current = analysis.prepare_trace_for_analysis(
        voltage,
        current,
        (0, 0),
        "average",
        0.5,
    )

    assert np.allclose(averaged_voltage, [0.05, 1.05, 2.05])
    assert np.allclose(averaged_current, [1.0, 3.0, 5.0])


def test_averaged_mode_marks_per_shot_fit_products_unavailable():
    assert analysis.analysis_trace_shape((2, 3), 4, "individual") == (2, 3, 4)
    assert analysis.analysis_trace_shape((2, 3), 4, "average") == (2, 3, 1)

    products = analysis.unavailable_per_shot_fit_products((2, 3), 4, 8)

    assert products["te_shot"].shape == (2, 3, 4)
    assert np.all(np.isnan(products["te_shot"]))
    assert products["iv_current_grid_shot"].shape == (2, 3, 4, 8)
    assert np.all(np.isnan(products["iv_current_grid_shot"]))
    assert not np.any(products["analysis_ok_shot"])

    available, stage = analysis.shot_statistics_metadata("average", 4)
    assert not available
    assert "unavailable" in stage

    available, stage = analysis.shot_statistics_metadata("individual", 4)
    assert available
    assert "individual fits" in stage


def test_select_xline_from_xy_map_uses_nearest_y_coordinate():
    y = np.array([-4.0, -1.0, 2.0, 5.0])
    density = np.arange(12, dtype=float).reshape(4, 3) * u.m**-3

    index, selected_y, profile = analysis.select_xline_from_xy_map(
        y,
        density,
        requested_y_cm=0.2,
    )

    assert index == 1
    assert selected_y == pytest.approx(-1.0)
    assert u.allclose(profile, density[1])


def test_xy_interferometer_scaling_uses_selected_xline_shape_factor():
    x = np.array([-10.0, 0.0, 10.0])
    y = np.array([-2.0, 1.0, 4.0])
    density = (
        np.array(
            [
                [0.5, 1.0, 0.5],
                [1.0, 2.0, 1.0],
                [1.5, 3.0, 1.5],
            ]
        )
        * 1e16
        * u.m**-3
    )
    line_integrated_density = 6e15 * u.m**-2

    result = analysis.scale_xy_density_map_to_interferometer(
        x,
        y,
        density,
        line_integrated_density,
        requested_y_cm=0.0,
    )

    assert result["selected_y_index"] == 1
    assert result["selected_y_cm"] == pytest.approx(1.0)
    assert result["shape_factor"].to_value(u.m) == pytest.approx(0.15)
    assert result["density_scale"] == pytest.approx(2.0)
    assert np.allclose(result["normalized_xline"], [0.5, 1.0, 0.5])
    assert u.allclose(result["scaled_xline"], density[1] * 2.0)
    assert u.allclose(result["scaled_density_map"], density * 2.0)


@pytest.mark.parametrize(
    ("y, density, message"),
    [
        ([], np.empty((0, 2)), "non-empty one-dimensional"),
        ([0.0, 1.0], np.empty((3, 2)), "shape"),
    ],
)
def test_select_xline_from_xy_map_rejects_inconsistent_shapes(y, density, message):
    with pytest.raises(ValueError, match=message):
        analysis.select_xline_from_xy_map(y, density)
