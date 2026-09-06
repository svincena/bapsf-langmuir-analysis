import numpy as np
import pytest

import read_langmuir_results_npz as combined_reader
from read_langmuir_results_npz import (
    EXPECTED_LANGMUIR_XLINE_NPZ_KEYS,
    EXPECTED_LANGMUIR_XY_NPZ_KEYS,
    get_langmuir_result_geometry,
    load_langmuir_results_npz,
    validate_langmuir_results_npz,
    validate_langmuir_xy_npz,
)


def _minimal_schema(geometry="xy_plane"):
    keys = (
        EXPECTED_LANGMUIR_XLINE_NPZ_KEYS
        if geometry == "x_line"
        else EXPECTED_LANGMUIR_XY_NPZ_KEYS
    )
    data = {key: np.array(0) for key in keys}
    data["geometry"] = np.array(geometry)
    return data


def test_xy_npz_schema_allows_forward_compatible_diagnostics():
    data = _minimal_schema()
    data["future_quality_metric"] = np.array(1.0)

    assert validate_langmuir_xy_npz(data) is data


def test_xy_npz_schema_still_rejects_missing_required_key():
    data = _minimal_schema()
    del data["te_eV"]

    with pytest.raises(ValueError, match="missing keys.*te_eV"):
        validate_langmuir_xy_npz(data)


@pytest.mark.parametrize("geometry", ["x_line", "xy_plane"])
def test_combined_validator_detects_both_geometries(geometry):
    data = _minimal_schema(geometry)

    assert get_langmuir_result_geometry(data) == geometry
    assert validate_langmuir_results_npz(data) is data


@pytest.mark.parametrize("geometry", ["x_line", "xy_plane"])
def test_combined_loader_round_trips_both_geometries(tmp_path, geometry):
    path = tmp_path / f"langmuir_{geometry}.npz"
    np.savez(path, **_minimal_schema(geometry))

    loaded = load_langmuir_results_npz(path)

    assert get_langmuir_result_geometry(loaded) == geometry


def test_combined_validator_rejects_unknown_geometry():
    data = _minimal_schema()
    data["geometry"] = np.array("radial")

    with pytest.raises(ValueError, match="Unsupported.*geometry"):
        validate_langmuir_results_npz(data)


def test_geometry_specific_reader_rejects_wrong_result_type():
    with pytest.raises(ValueError, match="Expected.*xy_plane.*x_line"):
        validate_langmuir_xy_npz(_minimal_schema("x_line"))


@pytest.mark.parametrize(
    ("geometry", "expected"),
    [("x_line", "x plot"), ("xy_plane", "xy plot")],
)
def test_plot_summary_dispatches_by_stored_geometry(
    monkeypatch,
    geometry,
    expected,
):
    monkeypatch.setattr(
        combined_reader,
        "plot_xline_summary",
        lambda data: "x plot",
    )
    monkeypatch.setattr(
        combined_reader,
        "plot_xy_summary",
        lambda data: "xy plot",
    )

    assert combined_reader.plot_summary(_minimal_schema(geometry)) == expected


@pytest.mark.parametrize("display_mode", ["separate_profiles", "ramp_time_map"])
def test_xline_reader_plots_multi_ramp_results(display_mode):
    import matplotlib.pyplot as plt

    x = np.array([-1.0, 0.0, 1.0])
    values = np.arange(6, dtype=float).reshape(3, 2) + 1.0
    data = {
        "geometry": np.array("x_line"),
        "source_file": np.array("synthetic.hdf5"),
        "x_cm": x,
        "ramp_center_time_s": np.array([0.01, 0.02]),
        "ramp_display_mode": np.array(display_mode),
        "analysis_ok": np.ones((3, 2), dtype=np.uint8),
    }
    for key in ("te_eV", "vp_V", "vf_V", "ies_A", "iis_A", "n_e_m3"):
        data[key] = values

    figure = combined_reader.plot_xline_summary(data)

    assert len(figure.axes) >= 6
    plt.close(figure)
