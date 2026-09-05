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


@pytest.mark.parametrize(
    ("geometry", "spatial_keys"),
    (
        ("x_line", ("nx", "x_min_cm", "x_max_cm")),
        (
            "xy_plane",
            ("nx", "x_min_cm", "x_max_cm", "ny", "y_min_cm", "y_max_cm"),
        ),
    ),
)
def test_spatial_and_acquisition_geometry_are_separate_adjacent_sections(
    geometry, spatial_keys
):
    sections = PARAMETER_SECTIONS[geometry]
    titles = tuple(section.title for section in sections)
    spatial_index = titles.index("Spatial geometry")
    acquisition_index = titles.index("Acquisition geometry")

    assert "Scan geometry" not in titles
    assert acquisition_index == spatial_index + 1
    assert tuple(
        parameter.key for parameter in sections[spatial_index].parameters
    ) == spatial_keys
    assert tuple(
        parameter.key for parameter in sections[acquisition_index].parameters
    ) == ("nshots", "shot_analysis_mode", "nt_full", "data_offset")
    assert sections[acquisition_index].stack_with_previous


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


def test_ies_method_is_selectable_and_validated():
    values = default_parameters("x_line")
    assert values["ies_method"] == "high_bias_median"

    values["ies_method"] = "at_vp"
    assert validate_parameters("x_line", values)["ies_method"] == "at_vp"

    values["ies_method"] = "unsupported"
    with pytest.raises(ValueError, match="Ies estimation method"):
        validate_parameters("x_line", values)


@pytest.mark.parametrize("geometry", SUPPORTED_GEOMETRIES)
def test_shot_analysis_mode_is_selectable_and_validated(geometry):
    values = default_parameters(geometry)
    assert values["shot_analysis_mode"] == "individual"

    values["shot_analysis_mode"] = "average"
    assert validate_parameters(geometry, values)["shot_analysis_mode"] == "average"

    values["shot_analysis_mode"] = "unsupported"
    with pytest.raises(ValueError, match="Shot fitting mode"):
        validate_parameters(geometry, values)
