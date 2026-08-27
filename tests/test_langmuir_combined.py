import astropy.units as u
import numpy as np
import pytest

import langmuir_analysis as analysis


def test_main_dispatches_selected_geometry(monkeypatch):
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
        monkeypatch.setattr(analysis, "analysis_geometry", geometry)
        analysis.main()

    assert called == ["x_line", "xy_plane"]


def test_main_rejects_unknown_geometry(monkeypatch):
    monkeypatch.setattr(analysis, "analysis_geometry", "radial")
    with pytest.raises(ValueError, match="analysis_geometry"):
        analysis.main()


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
