import json

import pytest

from langmuir_analysis_config import (
    PARAMETER_SECTIONS,
    SUPPORTED_GEOMETRIES,
    default_parameters,
    load_last_parameters,
    save_last_parameters,
    validate_parameters,
)


@pytest.mark.parametrize("geometry", SUPPORTED_GEOMETRIES)
def test_defaults_are_complete_and_valid(geometry):
    defaults = default_parameters(geometry)
    keys = [
        parameter.key
        for section in PARAMETER_SECTIONS[geometry]
        for parameter in section.parameters
    ]

    assert len(keys) == len(set(keys))
    assert set(defaults) == set(keys)
    assert validate_parameters(geometry, defaults) == defaults


def test_parameter_file_round_trip_preserves_both_tabs(tmp_path):
    path = tmp_path / "last_parameters.json"
    parameters = {
        geometry: default_parameters(geometry) for geometry in SUPPORTED_GEOMETRIES
    }
    parameters["x_line"]["nx"] = 17
    parameters["xy_plane"]["interferometer_profile_y_cm"] = 2.5

    save_last_parameters(parameters, "xy_plane", path)
    loaded, active_geometry, warning = load_last_parameters(path)

    assert warning is None
    assert active_geometry == "xy_plane"
    assert loaded == parameters
    assert json.loads(path.read_text())["version"] == 1


def test_missing_parameter_file_uses_defaults(tmp_path):
    loaded, active_geometry, warning = load_last_parameters(
        tmp_path / "does-not-exist.json"
    )

    assert warning is None
    assert active_geometry == "x_line"
    assert loaded["x_line"] == default_parameters("x_line")
    assert loaded["xy_plane"] == default_parameters("xy_plane")


def test_invalid_parameter_file_uses_defaults_with_warning(tmp_path):
    path = tmp_path / "last_parameters.json"
    path.write_text("not JSON", encoding="utf-8")

    loaded, active_geometry, warning = load_last_parameters(path)

    assert active_geometry == "x_line"
    assert warning is not None
    assert loaded["x_line"] == default_parameters("x_line")


def test_cross_field_validation_rejects_invalid_sweep():
    values = default_parameters("x_line")
    values["sweep_end_index"] = values["nt_full"]

    with pytest.raises(ValueError, match="smaller than samples per trace"):
        validate_parameters("x_line", values)
