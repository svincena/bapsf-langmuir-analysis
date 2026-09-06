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
    assert "interferometer_physical_constant" not in defaults
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


def test_repeated_ramp_controls_are_available_for_both_geometries():
    xline = default_parameters("x_line")
    xy = default_parameters("xy_plane")

    assert xline["nramps"] == 1
    assert xy["nramps"] == 1
    assert "ramp_start_spacing_samples" in xy
    assert xline["ramp_display_mode"] == "separate_profiles"
    assert "ramp_display_mode" not in xy

    xline.update(
        nt_full=100,
        sweep_start_index=10,
        sweep_end_index=19,
        nramps=3,
        ramp_start_spacing_samples=30,
        ramp_display_mode="ramp_time_map",
        isat_end_index=5,
        sg_smooth_bins=7,
    )
    validated = validate_parameters("x_line", xline)
    assert validated["nramps"] == 3
    assert validated["ramp_display_mode"] == "ramp_time_map"

    xline["ramp_start_spacing_samples"] = 9
    with pytest.raises(ValueError, match="at least the extracted ramp length"):
        validate_parameters("x_line", xline)

    xy.update(
        nt_full=100,
        sweep_start_index=10,
        sweep_end_index=19,
        nramps=3,
        ramp_start_spacing_samples=30,
        isat_end_index=5,
        sg_smooth_bins=7,
    )
    assert validate_parameters("xy_plane", xy)["nramps"] == 3


def test_repeated_ramps_must_fit_inside_trace_and_avoid_dc_window():
    values = default_parameters("x_line")
    values.update(
        nt_full=100,
        sweep_start_index=10,
        sweep_end_index=19,
        nramps=3,
        ramp_start_spacing_samples=41,
        isat_end_index=5,
        sg_smooth_bins=7,
    )
    with pytest.raises(ValueError, match="Every extracted ramp"):
        validate_parameters("x_line", values)

    values.update(
        ramp_start_spacing_samples=30,
        subtract_dc=True,
        isweep_dc_offset_start_index=42,
        isweep_dc_offset_end_index=45,
    )
    with pytest.raises(ValueError, match="must not overlap any I–V ramp"):
        validate_parameters("x_line", values)
