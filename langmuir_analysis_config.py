"""Typed user-parameter definitions for the Langmuir analysis GUI.

This module is the single source of truth for GUI labels, defaults, basic
types, ranges, persistence, and cross-field validation.  The analysis engine
receives normalized dictionaries and contains no experiment-specific defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


PARAMETER_FILE_VERSION = 1
LAST_PARAMETERS_PATH = Path(__file__).with_name("last_parameters.json")
SUPPORTED_GEOMETRIES = ("x_line", "xy_plane")
SUPPORTED_SPATIAL_GEOMETRY_SOURCES = ("manual", "bmotion")
SUPPORTED_XY_Y_ACQUISITION_ORDERS = ("descending", "ascending")
SUPPORTED_SHOT_ANALYSIS_MODES = ("individual", "average")
SUPPORTED_RAMP_DISPLAY_MODES = ("separate_profiles", "ramp_time_map")


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    kind: str
    default: Any
    description: str
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    decimals: int = 3
    step: float = 1.0
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class SectionSpec:
    title: str
    description: str
    parameters: tuple[ParameterSpec, ...]
    stack_with_previous: bool = False


def _p(
    key,
    label,
    kind,
    default,
    description,
    *,
    unit="",
    minimum=None,
    maximum=None,
    decimals=3,
    step=1.0,
    choices=(),
):
    return ParameterSpec(
        key=key,
        label=label,
        kind=kind,
        default=default,
        description=description,
        unit=unit,
        minimum=minimum,
        maximum=maximum,
        decimals=decimals,
        step=step,
        choices=tuple(choices),
    )


def _sections_for_geometry(geometry):
    is_xline = geometry == "x_line"
    if is_xline:
        filename = (
            "/Users/vincena/data/Mini_Magnetospheres/August2026/"
            "run_06_langmuir_xline_dipole_removed 2026-08-26 09.33.06.hdf5"
        )
        digitizer = "SIS crate"
        adc = "SIS 3302"
        sis_config_name = "Isat_Isweep_Vsweep_282624S_50MHz"
        current_zero_calibrated = True
        nshots, nt_full, nx = 5, 282_624, 81
        board, vsweep_channel, isweep_channel = 2, 3, 2
        isweep_attenuation, isweep_resistance = 1.0, 3.1
        data_offset = 0
        sweep_start, sweep_end = 5_010, 15_000
        dc_start, dc_end = 0, 127
        sg_bins, sg_order = 32, 2
        diagnostic_every = 2
        iv_npts, voltage_bin_width = 2_048, 0.05
        te_margin, te_floor = 0.2, 0.03
        te_min, te_max, te_r2 = 0.05, 20.0, 0.98
        vp_half_window: int | tuple[int, int] = 2
        smooth_half_window: int | tuple[int, int] = 1
        smooth_sigma = 1.0
    else:
        filename = (
            "/Users/vincena/data/Gekelman/Sparse_Alfven/"
            "Vp_p25_then_Lang_p35 2026-02-11 16.59.11.hdf5"
        )
        digitizer = "SIS crate"
        adc = "SIS 3302"
        sis_config_name = "64kS_100MHz_div_32__Lang_longtime"
        current_zero_calibrated = False
        nshots, nt_full, nx = 8, 65_536, 31
        board, vsweep_channel, isweep_channel = 4, 4, 5
        isweep_attenuation, isweep_resistance = 4.0, 1.0
        data_offset = 31 * 31 * 8
        sweep_start, sweep_end = 950, 2_400
        dc_start, dc_end = 64_000, 65_023
        sg_bins, sg_order = 17, 1
        diagnostic_every = 31 * 8
        iv_npts, voltage_bin_width = 512, 0.01
        te_margin, te_floor = 1.0, 0.075
        te_min, te_max, te_r2 = 0.1, 30.0, 0.975
        vp_half_window = (1, 1)
        smooth_half_window = (1, 1)
        smooth_sigma = 2.0

    sections = [
        SectionSpec(
            "Data source",
            "Experiment file, bapsflib digitizer identifiers, and channel routing.",
            (
                _p(
                    "filename",
                    "Experiment HDF5 file",
                    "file",
                    filename,
                    "Source LAPD experiment file. Analysis results may be written back to this file.",
                ),
                _p(
                    "digitizer",
                    "Digitizer group",
                    "str",
                    digitizer,
                    "Digitizer group name passed to bapsflib.",
                ),
                _p(
                    "adc",
                    "ADC",
                    "str",
                    adc,
                    "ADC identifier passed to bapsflib.",
                ),
                _p(
                    "sis_config_name",
                    "SIS configuration",
                    "str",
                    sis_config_name,
                    "Digitizer configuration name stored in the HDF5 metadata.",
                ),
                _p(
                    "board",
                    "Board",
                    "int",
                    board,
                    "Digitizer board number.",
                    minimum=0,
                    maximum=10_000,
                ),
                _p(
                    "vsweep_channel",
                    "Voltage channel",
                    "int",
                    vsweep_channel,
                    "Swept-bias channel number.",
                    minimum=0,
                    maximum=10_000,
                ),
                _p(
                    "isweep_channel",
                    "Current channel",
                    "int",
                    isweep_channel,
                    "Probe-current channel number.",
                    minimum=0,
                    maximum=10_000,
                ),
            ),
        ),
        SectionSpec(
            "Spatial geometry",
            "Spatial coordinates sampled by the probe scan, entered manually or read from bmotion.",
            tuple(
                [
                    _p(
                        "spatial_geometry_source",
                        "Geometry source",
                        "choice",
                        "manual",
                        "Use manually entered geometry or load authoritative target positions from the HDF5 bmotion control.",
                        choices=SUPPORTED_SPATIAL_GEOMETRY_SOURCES,
                    ),
                    _p(
                        "bmotion_config_name",
                        "bmotion configuration",
                        "optional_str",
                        "",
                        "Motion-group configuration used to obtain geometry, repeated shots, and the shot offset.",
                    ),
                    _p(
                        "nx",
                        "X positions",
                        "int",
                        nx,
                        "Number of x positions.",
                        minimum=1,
                        maximum=100_000,
                    ),
                    _p(
                        "x_min_cm",
                        "Minimum x",
                        "float",
                        -37.5 if is_xline else -15.0,
                        "First x coordinate.",
                        unit="cm",
                        minimum=-100_000,
                        maximum=100_000,
                        decimals=3,
                        step=0.5,
                    ),
                    _p(
                        "x_max_cm",
                        "Maximum x",
                        "float",
                        37.5 if is_xline else 15.0,
                        "Last x coordinate.",
                        unit="cm",
                        minimum=-100_000,
                        maximum=100_000,
                        decimals=3,
                        step=0.5,
                    ),
                ]
                + (
                    []
                    if is_xline
                    else [
                        _p(
                            "ny",
                            "Y positions",
                            "int",
                            31,
                            "Number of y positions.",
                            minimum=1,
                            maximum=100_000,
                        ),
                        _p(
                            "y_min_cm",
                            "Minimum y",
                            "float",
                            -15.0,
                            "First y coordinate.",
                            unit="cm",
                            minimum=-100_000,
                            maximum=100_000,
                            decimals=3,
                            step=0.5,
                        ),
                        _p(
                            "y_max_cm",
                            "Maximum y",
                            "float",
                            15.0,
                            "Last y coordinate.",
                            unit="cm",
                            minimum=-100_000,
                            maximum=100_000,
                            decimals=3,
                            step=0.5,
                        ),
                        _p(
                            "xy_y_acquisition_order",
                            "Y acquisition order",
                            "choice",
                            "descending",
                            "Order in which XY rows were acquired before they are normalized to ascending Y.",
                            choices=SUPPORTED_XY_Y_ACQUISITION_ORDERS,
                        ),
                    ]
                )
            ),
        ),
        SectionSpec(
            "Acquisition geometry",
            "Repeated shots, trace length, and acquisition offset.",
            (
                _p(
                    "nshots",
                    "Shots per position",
                    "int",
                    nshots,
                    "Repeated shots at each spatial location.",
                    minimum=1,
                    maximum=100_000,
                ),
                _p(
                    "shot_analysis_mode",
                    "Shot fitting mode",
                    "choice",
                    "individual",
                    "Fit every shot independently, or average shots in voltage bins before fitting.",
                    choices=SUPPORTED_SHOT_ANALYSIS_MODES,
                ),
                _p(
                    "nt_full",
                    "Samples per trace",
                    "int",
                    nt_full,
                    "Full digitizer trace length.",
                    minimum=2,
                    maximum=100_000_000,
                ),
                _p(
                    "data_offset",
                    "Shot offset",
                    "int",
                    data_offset,
                    "Number of leading acquisition shots to skip.",
                    minimum=0,
                    maximum=2_000_000_000,
                ),
            ),
            stack_with_previous=True,
        ),
        SectionSpec(
            "Digitizer and probe",
            "Conversion from digitizer signal to probe voltage and current.",
            (
                _p(
                    "vsweep_attenuation",
                    "Voltage attenuation",
                    "float",
                    100.0,
                    "Voltage-channel scale factor.",
                    minimum=1e-12,
                    maximum=1e12,
                    decimals=6,
                ),
                _p(
                    "isweep_attenuation",
                    "Current attenuation",
                    "float",
                    isweep_attenuation,
                    "Current-channel scale factor before shunt conversion.",
                    minimum=1e-12,
                    maximum=1e12,
                    decimals=6,
                ),
                _p(
                    "isweep_resistance_ohm",
                    "Current shunt resistance",
                    "float",
                    isweep_resistance,
                    "Resistance used to convert the measured signal to amperes.",
                    unit="Ω",
                    minimum=1e-12,
                    maximum=1e12,
                    decimals=6,
                ),
                _p(
                    "probe_area_mm2",
                    "Probe collection area",
                    "float",
                    4.0,
                    "Effective planar electron-collection area.",
                    unit="mm²",
                    minimum=1e-12,
                    maximum=1e12,
                    decimals=6,
                ),
                _p(
                    "negate_Isweep_current",
                    "Reverse current polarity",
                    "bool",
                    True,
                    "Enable when acquisition electronics report electron current with the opposite sign.",
                ),
            ),
        ),
        SectionSpec(
            "Sweep windows",
            "Indices are zero-based. Saturation indices are relative to the extracted sweep.",
            tuple(
                [
                    _p(
                        "sweep_start_index",
                        "Sweep start",
                        "int",
                        sweep_start,
                        "First full-trace sample included in the I–V sweep.",
                        minimum=0,
                        maximum=100_000_000,
                    ),
                    _p(
                        "sweep_end_index",
                        "Sweep end",
                        "int",
                        sweep_end,
                        "Last full-trace sample included in the I–V sweep.",
                        minimum=0,
                        maximum=100_000_000,
                    ),
                ]
                + [
                    _p(
                        "nramps",
                        "Number of ramps",
                        "int",
                        1,
                        "Number of equal-length voltage ramps analyzed in each discharge.",
                        minimum=1,
                        maximum=100_000,
                    ),
                    _p(
                        "ramp_start_spacing_samples",
                        "Ramp start spacing",
                        "int",
                        sweep_end - sweep_start + 1,
                        "Sample spacing between the starts of consecutive ramps.",
                        unit="samples",
                        minimum=1,
                        maximum=100_000_000,
                    ),
                ]
                + (
                    [
                        _p(
                            "ramp_display_mode",
                            "Multiple-ramp display",
                            "choice",
                            "separate_profiles",
                            "Show independent x profiles or an x-versus-ramp-time map.",
                            choices=SUPPORTED_RAMP_DISPLAY_MODES,
                        ),
                    ]
                    if is_xline
                    else []
                )
                + [
                    _p(
                        "isat_start_index",
                        "Isat window start",
                        "int",
                        0,
                        "First extracted-sweep sample used for the auxiliary Isat mean.",
                        minimum=0,
                        maximum=100_000_000,
                    ),
                    _p(
                        "isat_end_index",
                        "Isat window end",
                        "int",
                        127,
                        "Last extracted-sweep sample used for the auxiliary Isat mean.",
                        minimum=0,
                        maximum=100_000_000,
                    ),
                    _p(
                        "subtract_dc",
                        "Subtract electronics baseline",
                        "bool",
                        False,
                        "Use only with an independently measured plasma-off electronics interval.",
                    ),
                    _p(
                        "isweep_dc_offset_start_index",
                        "Baseline start",
                        "int",
                        dc_start,
                        "First full-trace sample in the independent electronics baseline.",
                        minimum=0,
                        maximum=100_000_000,
                    ),
                    _p(
                        "isweep_dc_offset_end_index",
                        "Baseline end",
                        "int",
                        dc_end,
                        "Last full-trace sample in the independent electronics baseline.",
                        minimum=0,
                        maximum=100_000_000,
                    ),
                ]
            ),
        ),
        SectionSpec(
            "I–V physics model",
            "Trace-level plasma-potential, temperature, saturation-current, and density controls.",
            (
                _p(
                    "current_zero_calibrated",
                    "Absolute current zero calibrated",
                    "bool",
                    current_zero_calibrated,
                    "Confirm only when current zero was established independently; required for physical fitting.",
                ),
                _p(
                    "enforce_ideal_model_checks",
                    "Enforce ideal Maxwellian checks",
                    "bool",
                    False,
                    "Reject sloped saturation branches and other deviations instead of recording model notes.",
                ),
                _p(
                    "ies_method",
                    "Ies estimation method",
                    "choice",
                    "high_bias_median",
                    "Use the legacy high-bias regional median or the ion-subtracted electron current at derivative Vp.",
                    choices=("high_bias_median", "at_vp"),
                ),
                _p(
                    "iv_npts",
                    "Diagnostic I–V points",
                    "int",
                    iv_npts,
                    "Fixed-size interpolated grid used for plots and export, not the physical fit.",
                    minimum=7,
                    maximum=1_000_000,
                ),
                _p(
                    "voltage_bin_width",
                    "Voltage-bin width",
                    "float",
                    voltage_bin_width,
                    "Merge nearby measured voltages before fitting.",
                    unit="V",
                    minimum=1e-9,
                    maximum=1e6,
                    decimals=6,
                    step=0.01,
                ),
                _p(
                    "vp_smoothing",
                    "Vp derivative smoothing",
                    "choice",
                    "savgol",
                    "Smoothing method used before locating the dI/dV peak.",
                    choices=("none", "moving", "savgol"),
                ),
                _p(
                    "vp_smoothing_width_V",
                    "Vp smoothing width",
                    "float",
                    2.5,
                    "Physical voltage width of the derivative smoother.",
                    unit="V",
                    minimum=1e-9,
                    maximum=1e6,
                    decimals=4,
                    step=0.1,
                ),
                _p(
                    "vp_savgol_order",
                    "Vp Savitzky–Golay order",
                    "int",
                    2,
                    "Polynomial order for derivative smoothing.",
                    minimum=1,
                    maximum=20,
                ),
                _p(
                    "te_min_points",
                    "Minimum Te fit points",
                    "int",
                    12,
                    "Minimum independent voltage bins in the semilog fit.",
                    minimum=3,
                    maximum=1_000_000,
                ),
                _p(
                    "te_margin_from_vp",
                    "Te margin below Vp",
                    "float",
                    te_margin,
                    "Exclude fit points this close to plasma potential.",
                    unit="V",
                    minimum=0,
                    maximum=1e6,
                    decimals=4,
                    step=0.1,
                ),
                _p(
                    "te_current_floor_frac",
                    "Te current floor",
                    "float",
                    te_floor,
                    "Minimum electron current as a fraction of electron saturation current.",
                    minimum=0,
                    maximum=0.799999,
                    decimals=5,
                    step=0.005,
                ),
                _p(
                    "te_min_eV",
                    "Minimum Te",
                    "float",
                    te_min,
                    "Lowest accepted electron temperature.",
                    unit="eV",
                    minimum=1e-9,
                    maximum=1e9,
                    decimals=4,
                    step=0.1,
                ),
                _p(
                    "te_max_eV",
                    "Maximum Te",
                    "float",
                    te_max,
                    "Highest accepted electron temperature.",
                    unit="eV",
                    minimum=1e-9,
                    maximum=1e9,
                    decimals=4,
                    step=0.5,
                ),
                _p(
                    "te_min_r2",
                    "Minimum Te fit R²",
                    "float",
                    te_r2,
                    "Minimum coefficient of determination for an accepted semilog fit.",
                    minimum=0,
                    maximum=1,
                    decimals=4,
                    step=0.005,
                ),
                _p(
                    "ion_min_snr",
                    "Minimum ion-current SNR",
                    "float",
                    3.0,
                    "Minimum resolved ion-region signal-to-uncertainty ratio.",
                    minimum=0,
                    maximum=1e9,
                    decimals=3,
                    step=0.25,
                ),
            ),
        ),
        SectionSpec(
            "Time and spatial processing",
            "Smoothing used for exported traces and optional cleanup of spatial products.",
            tuple(
                [
                    _p(
                        "sg_smooth_bins",
                        "Time Savitzky–Golay window",
                        "int",
                        sg_bins,
                        "Window length in original time samples.",
                        unit="samples",
                        minimum=2,
                        maximum=100_000_000,
                    ),
                    _p(
                        "sg_smooth_order",
                        "Time Savitzky–Golay order",
                        "int",
                        sg_order,
                        "Polynomial order for time-domain diagnostic smoothing.",
                        minimum=0,
                        maximum=20,
                    ),
                    _p(
                        "enable_vp_spike_rejection",
                        "Reject Vp spikes",
                        "bool",
                        True,
                        "Flag plasma-potential values far from their local median.",
                    ),
                    _p(
                        "vp_spike_half_window",
                        "Vp median half-window",
                        "int" if is_xline else "int_pair",
                        vp_half_window,
                        "Neighbor radius"
                        + (" along x." if is_xline else " in y and x."),
                        minimum=0,
                        maximum=10_000,
                    ),
                    _p(
                        "vp_spike_threshold_V",
                        "Vp spike threshold",
                        "float",
                        5.0,
                        "Absolute deviation from the local median required to flag a point.",
                        unit="V",
                        minimum=0,
                        maximum=1e9,
                        decimals=4,
                        step=0.25,
                    ),
                    _p(
                        "enable_neighbor_smoothing",
                        "Smooth spatial neighbors",
                        "bool",
                        True,
                        "Apply NaN-aware Gaussian-like spatial smoothing.",
                    ),
                    _p(
                        "neighbor_smooth_half_window",
                        "Smoothing half-window",
                        "int" if is_xline else "int_pair",
                        smooth_half_window,
                        "Neighbor radius"
                        + (" along x." if is_xline else " in y and x."),
                        minimum=0,
                        maximum=10_000,
                    ),
                    _p(
                        "neighbor_smooth_sigma",
                        "Neighbor smoothing sigma",
                        "float",
                        smooth_sigma,
                        "Gaussian-like weight width in spatial index units.",
                        minimum=1e-9,
                        maximum=1e9,
                        decimals=4,
                        step=0.1,
                    ),
                ]
                + [
                    _p(
                        "vp_replace_flagged_with_local_interp"
                        if is_xline
                        else "vp_replace_flagged_with_local_median",
                        "Replace flagged Vp values",
                        "bool",
                        True,
                        "Replace flagged points before optional neighbor smoothing.",
                    )
                ]
            ),
        ),
        SectionSpec(
            "Interferometer calibration",
            "Normalize the measured density profile to the microwave line-integrated density.",
            tuple(
                [
                    _p(
                        "calculate_shape_factor",
                        "Calibrate density to interferometer",
                        "bool",
                        True,
                        "Integrate the normalized x profile and use it to calibrate density.",
                    ),
                    _p(
                        "f_microwave_GHz",
                        "Microwave frequency",
                        "float",
                        288.0,
                        "Interferometer microwave frequency.",
                        unit="GHz",
                        minimum=1e-12,
                        maximum=1e9,
                        decimals=6,
                    ),
                    _p(
                        "N_passes",
                        "Beam passes",
                        "float",
                        2.0,
                        "Number of interferometer passes through the plasma.",
                        minimum=1e-12,
                        maximum=1e9,
                        decimals=4,
                    ),
                    _p(
                        "interferometer_phase_rad",
                        "Measured phase",
                        "float",
                        35.0,
                        "Measured microwave phase shift.",
                        unit="rad",
                        minimum=1e-12,
                        maximum=1e12,
                        decimals=6,
                    ),
                ]
                + (
                    []
                    if is_xline
                    else [
                        _p(
                            "interferometer_profile_y_cm",
                            "Calibration x-line y",
                            "float",
                            0.0,
                            "Use the measured xy row nearest this y coordinate.",
                            unit="cm",
                            minimum=-1e9,
                            maximum=1e9,
                            decimals=4,
                            step=0.5,
                        ),
                    ]
                )
            ),
        ),
        SectionSpec(
            "Execution and output",
            "Parallel execution, plots, diagnostics, and result persistence.",
            tuple(
                [
                    _p(
                        "save_results",
                        "Save HDF5 and NPZ results",
                        "bool",
                        True,
                        "Write the selected analysis products to both output formats.",
                    ),
                    _p(
                        "plot_results",
                        "Show summary plot",
                        "bool",
                        True,
                        "Create the six-panel geometry summary.",
                    ),
                ]
                + (
                    [
                        _p(
                            "plot_summary_stds",
                            "Plot shot standard deviations",
                            "bool",
                            True,
                            "Draw ±1σ error bars when repeated shots are available.",
                        )
                    ]
                    if is_xline
                    else []
                )
                + [
                    _p(
                        "parallel_analysis",
                        "Analyze traces in parallel",
                        "bool",
                        True,
                        "Use independent worker processes when the platform supports fork.",
                    ),
                    _p(
                        "analysis_processes",
                        "Worker processes",
                        "optional_int",
                        None,
                        "Zero/Automatic uses one fewer than the detected CPU count.",
                        minimum=0,
                        maximum=10_000,
                    ),
                    _p(
                        "diagnostic_plot_every",
                        "Diagnostic plot cadence",
                        "int",
                        diagnostic_every,
                        "Create a detailed I–V plot every N traces; zero disables it.",
                        minimum=0,
                        maximum=2_000_000_000,
                    ),
                    _p(
                        "example_pause_seconds",
                        "Diagnostic pause",
                        "float",
                        1.0,
                        "Time to show each interactive diagnostic plot.",
                        unit="s",
                        minimum=0,
                        maximum=1e9,
                        decimals=3,
                        step=0.1,
                    ),
                    _p(
                        "make_all_iv_diagnostic_plot",
                        "Plot all representative I–V curves",
                        "bool",
                        True,
                        "Create the geometry-wide interpolated I–V diagnostic plot.",
                    ),
                ]
            ),
        ),
    ]
    return tuple(sections)


PARAMETER_SECTIONS = {
    geometry: _sections_for_geometry(geometry) for geometry in SUPPORTED_GEOMETRIES
}
PARAMETER_SPECS = {
    geometry: {
        parameter.key: parameter
        for section in sections
        for parameter in section.parameters
    }
    for geometry, sections in PARAMETER_SECTIONS.items()
}


def default_parameters(geometry):
    """Return a fresh default parameter dictionary for one geometry."""
    if geometry not in SUPPORTED_GEOMETRIES:
        raise ValueError(f"Unsupported analysis geometry {geometry!r}.")
    return {key: spec.default for key, spec in PARAMETER_SPECS[geometry].items()}


def _coerce_parameter(spec, value):
    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{spec.label} must be true or false.")
        return value
    if spec.kind in {"str", "file", "directory", "optional_str"}:
        value = str(value).strip()
        if not value and spec.kind != "optional_str":
            raise ValueError(f"{spec.label} cannot be empty.")
        return value
    if spec.kind in {"int", "optional_int"}:
        if spec.kind == "optional_int" and value is None:
            return None
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{spec.label} must be an integer.")
        value = int(value)
    elif spec.kind == "float":
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{spec.label} must be finite.")
    elif spec.kind == "int_pair":
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"{spec.label} must contain y and x integer values.")
        pair = []
        for item in value:
            if isinstance(item, bool) or int(item) != item:
                raise ValueError(f"{spec.label} values must be integers.")
            pair.append(int(item))
        value = tuple(pair)
    elif spec.kind == "choice":
        value = str(value)
        if value not in spec.choices:
            raise ValueError(f"{spec.label} must be one of {', '.join(spec.choices)}.")
        return value
    else:
        raise ValueError(f"Unknown parameter kind {spec.kind!r}.")

    values = value if isinstance(value, tuple) else (value,)
    for item in values:
        if spec.minimum is not None and item < spec.minimum:
            raise ValueError(f"{spec.label} must be at least {spec.minimum}.")
        if spec.maximum is not None and item > spec.maximum:
            raise ValueError(f"{spec.label} must be at most {spec.maximum}.")
    return value


def validate_parameters(geometry, values, *, require_input_file=False):
    """Merge, coerce, and cross-validate parameters for one geometry."""
    if geometry not in SUPPORTED_GEOMETRIES:
        raise ValueError(f"Unsupported analysis geometry {geometry!r}.")
    if not isinstance(values, dict):
        raise TypeError("Parameter values must be a dictionary.")

    specs = PARAMETER_SPECS[geometry]
    merged = default_parameters(geometry)
    merged.update({key: value for key, value in values.items() if key in specs})
    normalized = {
        key: _coerce_parameter(spec, merged[key]) for key, spec in specs.items()
    }

    if (
        normalized["spatial_geometry_source"] == "bmotion"
        and not normalized["bmotion_config_name"]
    ):
        raise ValueError(
            "bmotion configuration cannot be empty when geometry comes from HDF5."
        )

    if require_input_file and not Path(normalized["filename"]).is_file():
        raise ValueError(
            f"Experiment HDF5 file does not exist: {normalized['filename']}"
        )
    if normalized["x_min_cm"] >= normalized["x_max_cm"]:
        raise ValueError("Minimum x must be less than maximum x.")
    if geometry == "xy_plane" and normalized["y_min_cm"] >= normalized["y_max_cm"]:
        raise ValueError("Minimum y must be less than maximum y.")
    if normalized["sweep_start_index"] >= normalized["sweep_end_index"]:
        raise ValueError("Sweep start must be less than sweep end.")
    if normalized["sweep_end_index"] >= normalized["nt_full"]:
        raise ValueError("Sweep end must be smaller than samples per trace.")
    sweep_points = normalized["sweep_end_index"] - normalized["sweep_start_index"] + 1
    if (
        normalized["nramps"] > 1
        and normalized["ramp_start_spacing_samples"] < sweep_points
    ):
        raise ValueError(
            "Ramp start spacing must be at least the extracted ramp length."
        )
    last_sweep_end = (
        normalized["sweep_end_index"]
        + (normalized["nramps"] - 1)
        * normalized["ramp_start_spacing_samples"]
    )
    if last_sweep_end >= normalized["nt_full"]:
        raise ValueError(
            "Every extracted ramp must end before samples per trace."
        )

    if normalized["isat_start_index"] > normalized["isat_end_index"]:
        raise ValueError("Isat window start must not exceed its end.")
    if normalized["isat_end_index"] >= sweep_points:
        raise ValueError("The Isat window must lie inside the extracted sweep.")
    if normalized["sg_smooth_bins"] > sweep_points:
        raise ValueError("The time smoothing window cannot exceed the sweep length.")
    if normalized["sg_smooth_order"] >= normalized["sg_smooth_bins"]:
        raise ValueError("Time smoothing order must be smaller than its window.")
    if normalized["te_min_eV"] >= normalized["te_max_eV"]:
        raise ValueError("Minimum Te must be less than maximum Te.")

    if normalized["subtract_dc"]:
        dc_start = normalized["isweep_dc_offset_start_index"]
        dc_end = normalized["isweep_dc_offset_end_index"]
        if dc_start > dc_end or dc_end >= normalized["nt_full"]:
            raise ValueError("The electronics baseline must lie inside the full trace.")
        nramps = normalized.get("nramps", 1)
        ramp_spacing = normalized.get("ramp_start_spacing_samples", sweep_points)
        for ramp_index in range(nramps):
            ramp_start = normalized["sweep_start_index"] + ramp_index * ramp_spacing
            ramp_end = ramp_start + sweep_points - 1
            if not (dc_end < ramp_start or dc_start > ramp_end):
                raise ValueError(
                    "The electronics baseline must not overlap any I–V ramp."
                )

    return normalized


def load_last_parameters(path=LAST_PARAMETERS_PATH):
    """Load saved parameters, falling back to defaults on absence or error.

    Returns ``(parameters_by_geometry, active_geometry, warning)``. ``warning``
    is ``None`` for a successful load or a human-readable fallback reason.
    """
    path = Path(path)
    defaults = {
        geometry: default_parameters(geometry) for geometry in SUPPORTED_GEOMETRIES
    }
    if not path.exists():
        return defaults, "x_line", None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("top-level JSON value is not an object")
        if payload.get("version") != PARAMETER_FILE_VERSION:
            raise ValueError(
                f"unsupported parameter-file version {payload.get('version')!r}"
            )
        stored = payload.get("parameters", {})
        active_geometry = payload.get("active_geometry", "x_line")
        if active_geometry not in SUPPORTED_GEOMETRIES:
            active_geometry = "x_line"
        loaded = {
            geometry: validate_parameters(
                geometry,
                stored.get(geometry, {}),
            )
            for geometry in SUPPORTED_GEOMETRIES
        }
        return loaded, active_geometry, None
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return (
            defaults,
            "x_line",
            f"Could not load {path.name}: {error}. Defaults were restored.",
        )


def save_last_parameters(
    parameters_by_geometry, active_geometry, path=LAST_PARAMETERS_PATH
):
    """Validate and atomically save both tab configurations as JSON."""
    if active_geometry not in SUPPORTED_GEOMETRIES:
        raise ValueError(f"Unsupported active geometry {active_geometry!r}.")
    normalized = {
        geometry: validate_parameters(
            geometry,
            parameters_by_geometry.get(geometry, {}),
        )
        for geometry in SUPPORTED_GEOMETRIES
    }
    payload = {
        "version": PARAMETER_FILE_VERSION,
        "active_geometry": active_geometry,
        "parameters": normalized,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return normalized
