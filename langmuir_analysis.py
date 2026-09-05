"""Analyze swept Langmuir-probe data along an x line or across an xy plane.

Running this module without arguments opens the graphical parameter editor.
The GUI starts the selected geometry in a separate worker process so its event
loop stays responsive while data are read, analyzed, plotted, and saved.  The
trace-level numerical analysis remains in :mod:`langmuir_analysis_core`.
"""

# The documented no-argument desktop launch takes this lightweight path before
# importing Astropy, SciPy, Matplotlib, bapsflib, and the numerical pipelines.
# Worker/CLI invocations continue below and load the full analysis environment.
if __name__ == "__main__":
    import sys as _startup_sys

    if len(_startup_sys.argv) == 1:
        from langmuir_analysis_gui import launch_gui as _launch_gui_immediately

        raise SystemExit(_launch_gui_immediately())

# %% Imports
import argparse
import io
import multiprocessing as mp
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import astropy.units as u
from bapsflib import lapd
import h5py
import matplotlib
import numpy as np
from scipy.signal import savgol_filter

# Keep Matplotlib and the parameter GUI on the same Qt binding.
os.environ["QT_API"] = "pyside6"
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from langmuir_analysis_config import (
    SUPPORTED_GEOMETRIES,
    load_last_parameters,
    validate_parameters,
)
from langmuir_analysis_core import analyze_iv_trace, bin_average_by_voltage
from langmuir_diagnostics import (
    render_iv_diagnostic_plot as render_analysis_iv_diagnostic_plot,
)


# %% Runtime configuration
# Experiment-specific values are supplied by the GUI or a saved parameter file.
# Keeping them out of this numerical engine makes a run explicit and repeatable.
analysis_geometry = None
_SUPPORTED_ANALYSIS_GEOMETRIES = set(SUPPORTED_GEOMETRIES)
_configured_parameter_values = None

# The geometry pipelines predate the GUI and intentionally read module-level
# runtime values. These placeholders document the names installed atomically by
# ``configure_analysis`` without reintroducing experiment-specific defaults.
_RUNTIME_UNCONFIGURED = object()
filename: Any = _RUNTIME_UNCONFIGURED
digitizer: Any = _RUNTIME_UNCONFIGURED
adc: Any = _RUNTIME_UNCONFIGURED
sis_config_name: Any = _RUNTIME_UNCONFIGURED
nx: Any = _RUNTIME_UNCONFIGURED
ny: Any = _RUNTIME_UNCONFIGURED
nshots: Any = _RUNTIME_UNCONFIGURED
shot_analysis_mode: Any = _RUNTIME_UNCONFIGURED
nt_full: Any = _RUNTIME_UNCONFIGURED
x_min: Any = _RUNTIME_UNCONFIGURED
x_max: Any = _RUNTIME_UNCONFIGURED
y_min: Any = _RUNTIME_UNCONFIGURED
y_max: Any = _RUNTIME_UNCONFIGURED
x: Any = _RUNTIME_UNCONFIGURED
y: Any = _RUNTIME_UNCONFIGURED
X: Any = _RUNTIME_UNCONFIGURED
Y: Any = _RUNTIME_UNCONFIGURED
board: Any = _RUNTIME_UNCONFIGURED
vsweep_channel: Any = _RUNTIME_UNCONFIGURED
isweep_channel: Any = _RUNTIME_UNCONFIGURED
vsweep_attenuation: Any = _RUNTIME_UNCONFIGURED
isweep_attenuation: Any = _RUNTIME_UNCONFIGURED
isweep_resistance: Any = _RUNTIME_UNCONFIGURED
probe_area: Any = _RUNTIME_UNCONFIGURED
data_offset: Any = _RUNTIME_UNCONFIGURED
shotnum_start: Any = _RUNTIME_UNCONFIGURED
shotnum_end: Any = _RUNTIME_UNCONFIGURED
n_expected_shots: Any = _RUNTIME_UNCONFIGURED
first_sweep_index: Any = _RUNTIME_UNCONFIGURED
sweep_start_index: Any = _RUNTIME_UNCONFIGURED
sweep_end_index: Any = _RUNTIME_UNCONFIGURED
nt: Any = _RUNTIME_UNCONFIGURED
isweep_dc_offset_start_index: Any = _RUNTIME_UNCONFIGURED
isweep_dc_offset_end_index: Any = _RUNTIME_UNCONFIGURED
isat_start_index: Any = _RUNTIME_UNCONFIGURED
isat_end_index: Any = _RUNTIME_UNCONFIGURED
subtract_dc: Any = _RUNTIME_UNCONFIGURED
sg_smooth_bins: Any = _RUNTIME_UNCONFIGURED
sg_smooth_order: Any = _RUNTIME_UNCONFIGURED
current_zero_calibrated: Any = _RUNTIME_UNCONFIGURED
negate_Isweep_current: Any = _RUNTIME_UNCONFIGURED
enforce_ideal_model_checks: Any = _RUNTIME_UNCONFIGURED
iv_npts: Any = _RUNTIME_UNCONFIGURED
voltage_bin_width: Any = _RUNTIME_UNCONFIGURED
vp_smoothing: Any = _RUNTIME_UNCONFIGURED
vp_smoothing_width_V: Any = _RUNTIME_UNCONFIGURED
vp_savgol_order: Any = _RUNTIME_UNCONFIGURED
ies_method: Any = _RUNTIME_UNCONFIGURED
te_min_points: Any = _RUNTIME_UNCONFIGURED
te_margin_from_vp: Any = _RUNTIME_UNCONFIGURED
te_current_floor_frac: Any = _RUNTIME_UNCONFIGURED
te_min_eV: Any = _RUNTIME_UNCONFIGURED
te_max_eV: Any = _RUNTIME_UNCONFIGURED
te_min_r2: Any = _RUNTIME_UNCONFIGURED
ion_min_snr: Any = _RUNTIME_UNCONFIGURED
te_subtract_i0: Any = _RUNTIME_UNCONFIGURED
langmuir_analysis_config: Any = _RUNTIME_UNCONFIGURED
enable_vp_spike_rejection: Any = _RUNTIME_UNCONFIGURED
vp_spike_half_window: Any = _RUNTIME_UNCONFIGURED
vp_spike_threshold_V: Any = _RUNTIME_UNCONFIGURED
vp_replace_flagged_with_local_interp: Any = _RUNTIME_UNCONFIGURED
vp_replace_flagged_with_local_median: Any = _RUNTIME_UNCONFIGURED
enable_neighbor_smoothing: Any = _RUNTIME_UNCONFIGURED
neighbor_smooth_half_window: Any = _RUNTIME_UNCONFIGURED
neighbor_smooth_sigma: Any = _RUNTIME_UNCONFIGURED
calculate_shape_factor: Any = _RUNTIME_UNCONFIGURED
f_microwave: Any = _RUNTIME_UNCONFIGURED
N_passes: Any = _RUNTIME_UNCONFIGURED
interferometer_phase: Any = _RUNTIME_UNCONFIGURED
interferometer_physical_constant: Any = _RUNTIME_UNCONFIGURED
interferometer_scaling: Any = _RUNTIME_UNCONFIGURED
interferometer_profile_y_cm: Any = _RUNTIME_UNCONFIGURED
save_results: Any = _RUNTIME_UNCONFIGURED
plot_results: Any = _RUNTIME_UNCONFIGURED
plot_summary_stds: Any = _RUNTIME_UNCONFIGURED
parallel_analysis: Any = _RUNTIME_UNCONFIGURED
analysis_processes: Any = _RUNTIME_UNCONFIGURED
diagnostic_plot_every: Any = _RUNTIME_UNCONFIGURED
example_pause_seconds: Any = _RUNTIME_UNCONFIGURED
diagnostic_plot_output_dir: Any = _RUNTIME_UNCONFIGURED
make_all_iv_diagnostic_plot: Any = _RUNTIME_UNCONFIGURED


def diagnostic_plot_directory(source_filename):
    """Return the diagnostic directory stored beside an HDF5 source file."""
    source_path = Path(source_filename).expanduser()
    return source_path.parent / f"{source_path.stem}_Langmuir_diagnostic_plots"


def configure_analysis(geometry, parameter_values):
    """Validate GUI values and populate the globals consumed by one pipeline."""
    if geometry not in _SUPPORTED_ANALYSIS_GEOMETRIES:
        choices = ", ".join(sorted(_SUPPORTED_ANALYSIS_GEOMETRIES))
        raise ValueError(
            f"analysis_geometry must be one of {{{choices}}}; got {geometry!r}."
        )

    values = validate_parameters(geometry, parameter_values)
    runtime = dict(values)
    runtime.update(
        analysis_geometry=geometry,
        filename=values["filename"],
        digitizer=values["digitizer"],
        adc=values["adc"],
        sis_config_name=values["sis_config_name"],
        nx=values["nx"],
        nshots=values["nshots"],
        nt_full=values["nt_full"],
        x_min=values["x_min_cm"],
        x_max=values["x_max_cm"],
        x=np.linspace(values["x_min_cm"], values["x_max_cm"], values["nx"]),
        board=values["board"],
        vsweep_channel=values["vsweep_channel"],
        isweep_channel=values["isweep_channel"],
        vsweep_attenuation=values["vsweep_attenuation"],
        isweep_attenuation=values["isweep_attenuation"],
        isweep_resistance=values["isweep_resistance_ohm"],
        probe_area=values["probe_area_mm2"] * u.mm**2,
        data_offset=values["data_offset"],
        sweep_start_index=values["sweep_start_index"],
        first_sweep_index=values["sweep_start_index"],
        sweep_end_index=values["sweep_end_index"],
        nt=values["sweep_end_index"] - values["sweep_start_index"] + 1,
        isweep_dc_offset_start_index=values["isweep_dc_offset_start_index"],
        isweep_dc_offset_end_index=values["isweep_dc_offset_end_index"],
        isat_start_index=values["isat_start_index"],
        isat_end_index=values["isat_end_index"],
        diagnostic_plot_output_dir=diagnostic_plot_directory(values["filename"]),
        te_subtract_i0=True,
        f_microwave=values["f_microwave_GHz"] * u.GHz,
        N_passes=values["N_passes"],
        interferometer_phase=values["interferometer_phase_rad"] * u.rad,
        interferometer_physical_constant=(
            values["interferometer_physical_constant"] * u.s / u.m**2 / u.rad
        ),
    )

    n_spatial_positions = values["nx"]
    if geometry == "xy_plane":
        runtime.update(
            ny=values["ny"],
            y_min=values["y_min_cm"],
            y_max=values["y_max_cm"],
            y=np.linspace(values["y_min_cm"], values["y_max_cm"], values["ny"]),
        )
        runtime["X"], runtime["Y"] = np.meshgrid(
            runtime["x"], runtime["y"], indexing="xy"
        )
        n_spatial_positions *= values["ny"]

    runtime["shotnum_start"] = values["data_offset"] + 1
    runtime["shotnum_end"] = (
        runtime["shotnum_start"] + n_spatial_positions * values["nshots"]
    )
    runtime["n_expected_shots"] = (
        runtime["shotnum_end"] - runtime["shotnum_start"]
    )
    runtime["interferometer_scaling"] = (
        runtime["interferometer_physical_constant"]
        * runtime["f_microwave"]
        * runtime["interferometer_phase"]
        / runtime["N_passes"]
    )
    runtime["langmuir_analysis_config"] = {
        "iv_npts": values["iv_npts"],
        "voltage_bin_width": values["voltage_bin_width"],
        "vp_smoothing": values["vp_smoothing"],
        "vp_smoothing_width_V": values["vp_smoothing_width_V"],
        "vp_savgol_order": values["vp_savgol_order"],
        "ies_method": values["ies_method"],
        "te_min_points": values["te_min_points"],
        "te_margin_from_vp": values["te_margin_from_vp"],
        "te_current_floor_frac": values["te_current_floor_frac"],
        "te_min_eV": values["te_min_eV"],
        "te_max_eV": values["te_max_eV"],
        "te_min_r2": values["te_min_r2"],
        "ion_min_snr": values["ion_min_snr"],
        "probe_area": runtime["probe_area"],
        "current_zero_calibrated": values["current_zero_calibrated"],
        "enforce_ideal_model_checks": values["enforce_ideal_model_checks"],
    }
    runtime["_configured_parameter_values"] = values
    globals().update(runtime)
    return values


def run_analysis(geometry, parameter_values):
    """Configure and run exactly one geometry-specific analysis pipeline."""
    configure_analysis(geometry, parameter_values)
    if geometry == "x_line":
        return run_xline_analysis()
    if geometry == "xy_plane":
        return run_xy_analysis()
    raise AssertionError("Geometry validation did not reject an invalid value.")


def describe_ies_method(method):
    """Return the persisted human-readable definition of an Ies method."""
    if method == "at_vp":
        return "ion-subtracted electron current interpolated at derivative Vp"
    if method == "high_bias_median":
        return (
            "median measured current in the upper half of the "
            "Vp-to-maximum-bias interval"
        )
    raise ValueError(f"Unknown Ies method {method!r}.")


def shot_mean_and_std(values, valid_mask=None):
    """Return nan-aware mean, sample standard deviation, and count over shots."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    count = np.sum(valid, axis=-1)
    total = np.sum(np.where(valid, values, 0.0), axis=-1)
    mean = np.full(count.shape, np.nan, dtype=float)
    np.divide(total, count, out=mean, where=count > 0)

    squared_deviation = np.where(valid, (values - mean[..., np.newaxis]) ** 2, 0.0)
    std = np.full(count.shape, np.nan, dtype=float)
    np.divide(
        np.sum(squared_deviation, axis=-1),
        count - 1,
        out=std,
        where=count > 1,
    )
    np.sqrt(std, out=std)
    return mean, std, count


def sanitize_filename_component(text):
    """Make a short filesystem-safe filename component."""
    text = str(text)
    safe_chars = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        elif char.isspace():
            safe_chars.append("_")
    return "".join(safe_chars).strip("._") or "plot"


def save_diagnostic_figure(fig, output_dir, filename_stem, dpi=600):
    """Save a diagnostic figure to the configured output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sanitize_filename_component(filename_stem)}.png"
    fig.savefig(path, format="png", dpi=dpi)
    print(f"Saved diagnostic plot: {path}")
    return path


def electron_density_profile_shape_factor(x_cm, electron_density_profile):
    """Integrate peak-normalized electron density over x and return meters."""
    x_values_cm = np.asarray(x_cm, dtype=float)
    profile_values = np.asarray(
        getattr(electron_density_profile, "value", electron_density_profile),
        dtype=float,
    )

    if x_values_cm.shape != profile_values.shape:
        raise ValueError(
            "x coordinates and electron-density profile must have the same shape; "
            f"got {x_values_cm.shape} and {profile_values.shape}."
        )

    finite = np.isfinite(x_values_cm) & np.isfinite(profile_values)
    if np.count_nonzero(finite) < 2:
        return np.nan * u.m

    x_values_m = (x_values_cm[finite] * u.cm).to_value(u.m)
    finite_profile = profile_values[finite]
    profile_maximum = np.max(finite_profile)
    if not np.isfinite(profile_maximum) or profile_maximum == 0.0:
        return np.nan * u.m

    normalized_profile = finite_profile / profile_maximum
    # Apply the trapezoidal rule after converting the position samples to
    # meters, so the dimensionless normalized profile integrates to a length.
    shape_factor_m = np.sum(
        np.diff(x_values_m)
        * 0.5
        * (normalized_profile[:-1] + normalized_profile[1:])
    )
    return shape_factor_m * u.m


def scale_density_profile_to_interferometer(
    electron_density_profile,
    line_integrated_density,
    shape_factor,
):
    """Normalize density to unit peak and calibrate it to an interferometer."""
    density = u.Quantity(electron_density_profile).to(u.m**-3)
    density_values = np.asarray(density.value, dtype=float)
    normalized_density = np.full(density_values.shape, np.nan, dtype=float)
    finite = np.isfinite(density_values)
    if not np.any(finite):
        return normalized_density, normalized_density * u.m**-3, np.nan

    density_maximum = np.max(density_values[finite])
    shape_factor_m = u.Quantity(shape_factor).to(u.m)
    line_integrated_density = u.Quantity(line_integrated_density).to(u.m**-2)
    if (
        not np.isfinite(density_maximum)
        or density_maximum <= 0.0
        or not np.isfinite(shape_factor_m.value)
        or shape_factor_m <= 0.0 * u.m
        or not np.isfinite(line_integrated_density.value)
        or line_integrated_density <= 0.0 * u.m**-2
    ):
        return normalized_density, normalized_density * u.m**-3, np.nan

    normalized_density[finite] = density_values[finite] / density_maximum
    peak_density = (line_integrated_density / shape_factor_m).to(u.m**-3)
    scaled_density = normalized_density * peak_density
    density_scale = (peak_density / (density_maximum * u.m**-3)).to_value(
        u.dimensionless_unscaled
    )
    return normalized_density, scaled_density, density_scale


def select_xline_from_xy_map(y_cm, density_map, requested_y_cm=0.0):
    """Return the xy-map row nearest a requested y position.

    Parameters
    ----------
    y_cm : array-like or `~astropy.units.Quantity`
        One-dimensional y coordinates for the first map axis.
    density_map : array-like or `~astropy.units.Quantity`
        Two-dimensional electron-density map ordered as ``(y, x)``.
    requested_y_cm : float or `~astropy.units.Quantity`
        Desired x-line location. Unitless values are interpreted as cm.
    """
    y_values = (
        u.Quantity(y_cm).to_value(u.cm)
        if isinstance(y_cm, u.Quantity)
        else np.asarray(y_cm, dtype=float)
    )
    if y_values.ndim != 1 or y_values.size == 0:
        raise ValueError("y_cm must be a non-empty one-dimensional array.")

    density_shape = np.shape(density_map)
    if len(density_shape) != 2 or density_shape[0] != y_values.size:
        raise ValueError(
            "density_map must have shape (len(y_cm), nx); "
            f"got {density_shape} for {y_values.size} y coordinates."
        )

    requested_value = (
        u.Quantity(requested_y_cm).to_value(u.cm)
        if isinstance(requested_y_cm, u.Quantity)
        else float(requested_y_cm)
    )
    if not np.isfinite(requested_value):
        raise ValueError("requested_y_cm must be finite.")

    finite_y = np.isfinite(y_values)
    if not np.any(finite_y):
        raise ValueError("y_cm must contain at least one finite coordinate.")
    finite_indices = np.flatnonzero(finite_y)
    local_index = np.argmin(np.abs(y_values[finite_y] - requested_value))
    y_index = int(finite_indices[local_index])
    return y_index, float(y_values[y_index]), density_map[y_index, :]


def scale_xy_density_map_to_interferometer(
    x_cm,
    y_cm,
    density_map,
    line_integrated_density,
    requested_y_cm=0.0,
):
    """Calibrate an xy density map from one measured x-line profile.

    The selected x-line is normalized and integrated in exactly the same way
    as the native x-line pipeline. Its scalar calibration factor is then
    applied uniformly to the complete xy map.
    """
    density = u.Quantity(density_map).to(u.m**-3)
    y_index, selected_y_cm, xline_density = select_xline_from_xy_map(
        y_cm,
        density,
        requested_y_cm=requested_y_cm,
    )
    shape_factor = electron_density_profile_shape_factor(x_cm, xline_density)
    normalized_xline, scaled_xline, density_scale = (
        scale_density_profile_to_interferometer(
            xline_density,
            line_integrated_density,
            shape_factor,
        )
    )

    if np.isfinite(density_scale):
        scaled_density_map = density * density_scale
    else:
        scaled_density_map = np.full(density.shape, np.nan) * u.m**-3

    return {
        "selected_y_index": y_index,
        "selected_y_cm": selected_y_cm,
        "shape_factor": shape_factor,
        "density_scale": density_scale,
        "normalized_xline": normalized_xline,
        "scaled_xline": scaled_xline,
        "scaled_density_map": scaled_density_map,
    }


def read_channel_xline(
    file_obj,
    board,
    channel,
    shotnum_start,
    shotnum_end,
    nx,
    nshots,
    nt_full,
    digitizer,
    adc,
    config_name,
    scale_factor=1.0,
    negate=False,
):
    """Read one digitizer channel and reshape it as (nx, nshots, nt_full)."""
    raw = file_obj.read_data(
        board,
        channel,
        digitizer=digitizer,
        adc=adc,
        config_name=config_name,
        shotnum=slice(shotnum_start, shotnum_end, 1),
    )

    signal = raw["signal"]
    dt = raw.dt.value

    if signal.ndim != 2:
        raise ValueError(
            f"Channel {channel}: expected raw signal with shape (shots, time), got {signal.shape}."
        )

    n_actual_shots, n_actual_time = signal.shape
    print(f"Channel {channel}: raw shape = {signal.shape}")
    print(f"Channel {channel}: expected shots = {shotnum_end - shotnum_start}, actual shots = {n_actual_shots}")
    print(f"Channel {channel}: expected nt_full = {nt_full}, actual time samples = {n_actual_time}")

    if n_actual_shots != (shotnum_end - shotnum_start):
        raise ValueError(
            f"Channel {channel}: shot count mismatch. "
            f"Expected {shotnum_end - shotnum_start}, found {n_actual_shots}. "
            f"Likely wrong data_offset, nx, or nshots."
        )

    if n_actual_time != nt_full:
        raise ValueError(
            f"Channel {channel}: time-length mismatch. Expected {nt_full}, found {n_actual_time}."
        )

    if negate:
        signal = -signal

    expected_shape = (nx, nshots, nt_full)
    if signal.size != np.prod(expected_shape):
        raise ValueError(
            f"Channel {channel}: cannot reshape size {signal.size} into {expected_shape}."
        )

    signal = signal.reshape(expected_shape)
    signal = signal * scale_factor
    return signal, dt


def open_h5_for_update(path):
    try:
        return h5py.File(path, "r+")
    except BlockingIOError as exc:
        if getattr(exc, "errno", None) == 35:
            print("HDF5 file lock unavailable (errno 35); retrying with locking disabled.")
            return h5py.File(path, "r+", locking=False)
        raise


def analyze_trace_worker(task):
    """Analyze one trace and preserve caller-owned geometry indexing metadata."""
    flat_index, idx, bias_values, current_values, include_diagnostic_data, config = task
    result = analyze_iv_trace(
        bias_values,
        current_values,
        config,
        include_diagnostic_data=include_diagnostic_data,
        trace_label=idx,
    )
    result["flat_index"] = flat_index
    result["idx"] = idx
    return result


def analysis_trace_shape(spatial_shape, shot_count, mode):
    """Return the fit-array shape for individual-shot or averaged analysis."""
    if mode == "individual":
        return tuple(spatial_shape) + (int(shot_count),)
    if mode == "average":
        return tuple(spatial_shape) + (1,)
    raise ValueError(f"Unknown shot analysis mode {mode!r}.")


def prepare_trace_for_analysis(
    voltage_by_shot,
    current_by_shot,
    fit_index,
    mode,
    voltage_bin_width,
):
    """Select one shot or form one voltage-binned mean across all shots."""
    voltage_by_shot = np.asarray(voltage_by_shot, dtype=float)
    current_by_shot = np.asarray(current_by_shot, dtype=float)
    if voltage_by_shot.shape != current_by_shot.shape:
        raise ValueError("Voltage and current shot arrays must have the same shape.")

    fit_index = tuple(fit_index)
    if mode == "individual":
        return (
            voltage_by_shot[fit_index + (slice(None),)],
            current_by_shot[fit_index + (slice(None),)],
        )
    if mode != "average":
        raise ValueError(f"Unknown shot analysis mode {mode!r}.")

    # The final fit-index entry is a singleton fit axis, not a physical shot.
    # Pooling first and then binning by measured voltage tolerates small timing
    # and endpoint differences between otherwise repeated voltage sweeps.
    spatial_index = fit_index[:-1]
    voltage = voltage_by_shot[spatial_index].reshape(-1)
    current = current_by_shot[spatial_index].reshape(-1)
    voltage_mean, current_mean = bin_average_by_voltage(
        voltage,
        current,
        bin_width=voltage_bin_width,
    )
    if voltage_mean is None:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    return voltage_mean, current_mean


def unavailable_per_shot_fit_products(spatial_shape, shot_count, diagnostic_points):
    """Return empty per-shot products when only an averaged fit was performed."""
    shot_shape = tuple(spatial_shape) + (int(shot_count),)
    grid_shape = shot_shape + (int(diagnostic_points),)
    return {
        "te_shot": np.full(shot_shape, np.nan),
        "vp_shot": np.full(shot_shape, np.nan),
        "vf_shot": np.full(shot_shape, np.nan),
        "ies_shot": np.full(shot_shape, np.nan),
        "iis_shot": np.full(shot_shape, np.nan),
        "n_e_shot": np.full(shot_shape, np.nan),
        "analysis_ok_shot": np.zeros(shot_shape, dtype=np.uint8),
        "statistics_fit_mask": np.zeros(shot_shape, dtype=bool),
        "te_fit_r2_shot": np.full(shot_shape, np.nan),
        "te_fit_rmse_shot": np.full(shot_shape, np.nan),
        "te_fit_npts_shot": np.full(shot_shape, np.nan),
        "te_fit_vstart_shot": np.full(shot_shape, np.nan),
        "te_fit_vstop_shot": np.full(shot_shape, np.nan),
        "te_fit_slope_shot": np.full(shot_shape, np.nan),
        "te_fit_intercept_shot": np.full(shot_shape, np.nan),
        "te_fit_i0_shot": np.full(shot_shape, np.nan),
        "te_fit_passed_r2_shot": np.zeros(shot_shape, dtype=np.uint8),
        "te_fit_candidate_count_shot": np.zeros(shot_shape, dtype=int),
        "iv_voltage_grid_shot": np.full(grid_shape, np.nan),
        "iv_current_grid_shot": np.full(grid_shape, np.nan),
        "iv_didv_grid_shot": np.full(grid_shape, np.nan),
        "iv_fit_mask_grid_shot": np.zeros(grid_shape, dtype=np.uint8),
    }


def shot_statistics_metadata(mode, shot_count):
    """Describe whether fit-derived shot statistics exist for this run."""
    if mode == "average":
        return False, "unavailable; shots averaged in voltage bins before fitting"
    if mode == "individual":
        if int(shot_count) < 2:
            return False, "unavailable; only one shot was configured"
        return True, "individual fits before spatial post-processing"
    raise ValueError(f"Unknown shot analysis mode {mode!r}.")


def choose_analysis_process_count(n_traces, requested_processes=None):
    """Choose a conservative worker count for independent trace fits."""
    n_traces = int(n_traces)
    if n_traces < 1:
        return 0
    if requested_processes is not None:
        return max(1, min(int(requested_processes), n_traces))
    return max(1, min((mp.cpu_count() or 1) - 1, n_traces))


def reject_spikes_by_local_median(y, half_window=2, threshold=5.0):
    """Return a mask of spike-like points using local median comparison."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    is_spike = np.zeros(n, dtype=bool)

    for i in range(n):
        if not np.isfinite(y[i]):
            continue

        i0 = max(0, i - half_window)
        i1 = min(n, i + half_window + 1)

        neighborhood = y[i0:i1].copy()
        local_idx = i - i0
        neighborhood = np.delete(neighborhood, local_idx)
        neighborhood = neighborhood[np.isfinite(neighborhood)]

        if neighborhood.size < 2:
            continue

        med = np.median(neighborhood)
        if np.abs(y[i] - med) > threshold:
            is_spike[i] = True

    return is_spike


def replace_flagged_by_local_interp(x_coord, y, flagged):
    """Replace flagged points in y by linear interpolation from unflagged finite neighbors."""
    x_coord = np.asarray(x_coord, dtype=float)
    y = np.asarray(y, dtype=float).copy()
    flagged = np.asarray(flagged, dtype=bool)

    good = np.isfinite(y) & (~flagged)
    if np.count_nonzero(good) < 2:
        return y

    y_out = y.copy()
    bad = flagged & np.isfinite(y)
    y_out[bad] = np.interp(x_coord[bad], x_coord[good], y[good])
    return y_out


def gaussian_neighbor_smooth_nanaware(x_coord, y, half_window=1, sigma=1.0):
    """Smooth 1D data using a local gaussian-like weighted average, ignoring NaNs."""
    x_coord = np.asarray(x_coord, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    y_smooth = np.full(n, np.nan)

    offsets = np.arange(-half_window, half_window + 1, dtype=float)
    weights_template = np.exp(-0.5 * (offsets / sigma) ** 2)

    for i in range(n):
        if not np.isfinite(y[i]):
            continue
        idx = np.arange(i - half_window, i + half_window + 1)
        keep = (idx >= 0) & (idx < n)
        idx = idx[keep]
        weights = weights_template[keep]

        vals = y[idx]
        finite = np.isfinite(vals)
        if np.count_nonzero(finite) == 0:
            continue

        vals = vals[finite]
        w = weights[finite]
        y_smooth[i] = np.sum(w * vals) / np.sum(w)

    return y_smooth


def apply_optional_x_postprocessing(
    x_coord,
    te,
    vp,
    vf,
    ies=None,
    iis=None,
    n_e=None,
    enable_vp_spike_rejection=True,
    vp_spike_half_window=2,
    vp_spike_threshold_V=5.0,
    vp_replace_flagged_with_local_interp=True,
    enable_neighbor_smoothing=True,
    neighbor_smooth_half_window=1,
    neighbor_smooth_sigma=1.0,
):
    """
    Optional x-line post-processing of Te(x), Vp(x), Vf(x), I_es(x), I_is(x), n_e(x):
      1) detect/reject Vp spikes
      2) neighbor-aware smoothing
    """
    x_coord = np.asarray(x_coord, dtype=float)

    te_in = np.asarray(te, dtype=float)
    vp_in = np.asarray(vp, dtype=float)
    vf_in = np.asarray(vf, dtype=float)
    ies_in = np.asarray(ies, dtype=float) if ies is not None else None
    iis_in = np.asarray(iis, dtype=float) if iis is not None else None
    n_e_in = np.asarray(n_e, dtype=float) if n_e is not None else None

    te_proc = te_in.copy()
    vp_proc = vp_in.copy()
    vf_proc = vf_in.copy()
    ies_proc = ies_in.copy() if ies_in is not None else None
    iis_proc = iis_in.copy() if iis_in is not None else None
    n_e_proc = n_e_in.copy() if n_e_in is not None else None

    vp_spike_mask = np.zeros_like(vp_proc, dtype=bool)

    if enable_vp_spike_rejection:
        vp_spike_mask = reject_spikes_by_local_median(
            vp_proc,
            half_window=vp_spike_half_window,
            threshold=vp_spike_threshold_V,
        )
        if vp_replace_flagged_with_local_interp:
            vp_proc = replace_flagged_by_local_interp(x_coord, vp_proc, vp_spike_mask)

    te_smooth = te_proc.copy()
    vp_smooth = vp_proc.copy()
    vf_smooth = vf_proc.copy()
    ies_smooth = ies_proc.copy() if ies_proc is not None else None
    iis_smooth = iis_proc.copy() if iis_proc is not None else None
    n_e_smooth = n_e_proc.copy() if n_e_proc is not None else None

    if enable_neighbor_smoothing:
        te_smooth = gaussian_neighbor_smooth_nanaware(
            x_coord, te_proc,
            half_window=neighbor_smooth_half_window,
            sigma=neighbor_smooth_sigma
        )
        vp_smooth = gaussian_neighbor_smooth_nanaware(
            x_coord, vp_proc,
            half_window=neighbor_smooth_half_window,
            sigma=neighbor_smooth_sigma
        )
        vf_smooth = gaussian_neighbor_smooth_nanaware(
            x_coord, vf_proc,
            half_window=neighbor_smooth_half_window,
            sigma=neighbor_smooth_sigma
        )
        if ies_proc is not None:
            ies_smooth = gaussian_neighbor_smooth_nanaware(
                x_coord, ies_proc,
                half_window=neighbor_smooth_half_window,
                sigma=neighbor_smooth_sigma
            )
        if iis_proc is not None:
            iis_smooth = gaussian_neighbor_smooth_nanaware(
                x_coord, iis_proc,
                half_window=neighbor_smooth_half_window,
                sigma=neighbor_smooth_sigma
            )
        if n_e_proc is not None:
            n_e_smooth = gaussian_neighbor_smooth_nanaware(
                x_coord, n_e_proc,
                half_window=neighbor_smooth_half_window,
                sigma=neighbor_smooth_sigma
            )

    result = {
        "te_raw": te_in,
        "vp_raw": vp_in,
        "vf_raw": vf_in,
        "te_processed": te_smooth,
        "vp_processed": vp_smooth,
        "vf_processed": vf_smooth,
        "vp_spike_mask": vp_spike_mask,
    }
    if ies_proc is not None:
        result["ies_raw"] = ies_proc
        result["ies_processed"] = ies_smooth
    if iis_proc is not None:
        result["iis_raw"] = iis_proc
        result["iis_processed"] = iis_smooth
    if n_e_proc is not None:
        result["n_e_raw"] = n_e_proc
        result["n_e_processed"] = n_e_smooth
    return result


def render_xline_summary_plot(
    x,
    te,
    vp,
    vf,
    ies,
    iis,
    n_e=None,
    te_std=None,
    vp_std=None,
    vf_std=None,
    ies_std=None,
    iis_std=None,
    n_e_std=None,
    plot_stds=False,
    vp_spike_mask=None,
    vp_raw=None,
    te_fit_r2=None,
    te_poor_fit_r2=None,
    analysis_ok=None,
    analysis_status=None,
    shape_factor=None,
    title="Langmuir X-Line Summary",
):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    finite_x = np.asarray(x)[np.isfinite(x)]
    fit_accepted = (
        None if analysis_ok is None else np.asarray(analysis_ok, dtype=bool)
    )

    def _finish_profile_axis(ax, plot_title, y_label):
        ax.set_title(plot_title)
        ax.set_xlabel("X (cm)")
        ax.set_ylabel(y_label)
        if finite_x.size:
            ax.set_xlim(np.min(finite_x), np.max(finite_x))
        ax.grid(True)

    def _mark_empty(ax):
        ax.text(
            0.5,
            0.5,
            "No accepted values",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="0.35",
        )

    def _plot_standard_deviations(ax, values, std, color):
        if not plot_stds or std is None:
            return

        plot_values = np.asarray(values.value, dtype=float)
        std_values = np.asarray(std.value, dtype=float)
        finite = np.isfinite(x) & np.isfinite(plot_values) & np.isfinite(std_values)
        if np.any(finite):
            ax.errorbar(
                np.asarray(x)[finite],
                plot_values[finite],
                yerr=std_values[finite],
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=3,
                alpha=0.7,
                label="shot-to-shot $\u00b11\u03c3$",
            )

    def _plot_profile(ax, values, std, plot_title, y_label):
        plot_values = np.asarray(values.value, dtype=float)
        finite = np.isfinite(x) & np.isfinite(plot_values)
        if np.any(finite):
            line = ax.plot(np.asarray(x)[finite], plot_values[finite])[0]
            _plot_standard_deviations(ax, values, std, line.get_color())
            if fit_accepted is not None:
                rejected = finite & ~fit_accepted
                if np.any(rejected):
                    ax.plot(
                        np.asarray(x)[rejected],
                        plot_values[rejected],
                        "x",
                        color="tab:orange",
                        label="no accepted fits",
                    )
        else:
            _mark_empty(ax)
        _finish_profile_axis(ax, plot_title, y_label)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    _plot_profile(axs[0, 0], te, te_std, "Electron Temperature", "T_e (eV)")
    if te_fit_r2 is not None and te_poor_fit_r2 is not None:
        poor_te_fit = np.isfinite(te_fit_r2) & np.isfinite(te.value) & (te_fit_r2 < te_poor_fit_r2)
        if np.any(poor_te_fit):
            axs[0, 0].plot(
                x[poor_te_fit],
                te.value[poor_te_fit],
                "rx",
                ms=6,
                mew=1.5,
                label=f"Te fit R^2 < {te_poor_fit_r2:.2f}",
            )
            axs[0, 0].legend()

    vp_values = np.asarray(vp.value, dtype=float)
    vp_finite = np.isfinite(x) & np.isfinite(vp_values)
    if np.any(vp_finite):
        vp_line = axs[0, 1].plot(
            np.asarray(x)[vp_finite],
            vp_values[vp_finite],
            label="Vp",
        )[0]
        _plot_standard_deviations(axs[0, 1], vp, vp_std, vp_line.get_color())
        if fit_accepted is not None:
            rejected = vp_finite & ~fit_accepted
            if np.any(rejected):
                axs[0, 1].plot(
                    np.asarray(x)[rejected],
                    vp_values[rejected],
                    "x",
                    color="tab:orange",
                    label="no accepted fits",
                )
    any_vp_values = bool(np.any(vp_finite))
    if vp_raw is not None:
        vp_raw_values = np.asarray(vp_raw.value, dtype=float)
        vp_raw_finite = np.isfinite(x) & np.isfinite(vp_raw_values)
        if np.any(vp_raw_finite):
            axs[0, 1].plot(
                np.asarray(x)[vp_raw_finite],
                vp_raw_values[vp_raw_finite],
                alpha=0.35,
                label="Vp raw",
            )
            any_vp_values = True
    if vp_spike_mask is not None and vp_raw is not None:
        m = vp_spike_mask.astype(bool)
        if np.any(m):
            axs[0, 1].plot(x[m], vp_raw.value[m], "o", label="flagged spikes")
    if not any_vp_values:
        _mark_empty(axs[0, 1])
    _finish_profile_axis(axs[0, 1], "Plasma Potential", "V_p (V)")
    handles, _ = axs[0, 1].get_legend_handles_labels()
    if handles:
        axs[0, 1].legend()

    _plot_profile(axs[1, 0], vf, vf_std, "Floating Potential", "V_f (V)")
    _plot_profile(
        axs[1, 1],
        ies,
        ies_std,
        "Electron Saturation Current",
        "I_es (A)",
    )
    _plot_profile(
        axs[2, 0],
        iis,
        iis_std,
        "Ion Saturation Current",
        "I_is (A)",
    )
    if n_e is not None:
        _plot_profile(
            axs[2, 1],
            n_e,
            n_e_std,
            "Electron Density",
            "n_e (m^-3)",
        )
        if shape_factor is not None:
            shape_factor_m = u.Quantity(shape_factor).to_value(u.m)
            shape_factor_text = (
                f"Shape factor = {shape_factor_m:.4g} m"
                if np.isfinite(shape_factor_m)
                else "Shape factor unavailable"
            )
            axs[2, 1].text(
                0.03,
                0.95,
                shape_factor_text,
                transform=axs[2, 1].transAxes,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
            )
    else:
        axs[2, 1].axis("off")

    summary_title = title if analysis_status is None else f"{title}\n{analysis_status}"
    fig.suptitle(summary_title, fontsize=16)
    return fig


def render_xline_all_iv_curves_plot(x, iv_voltage_grid, iv_current_grid):
    """Plot all interpolated I-V curves colored by x position."""
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    cmap = plt.get_cmap("viridis")
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)

    for i in range(len(x)):
        voltage_curve = iv_voltage_grid[i]
        current_curve = iv_current_grid[i]
        good = np.isfinite(voltage_curve) & np.isfinite(current_curve)
        if np.count_nonzero(good) < 2:
            continue

        frac = 0.5 if x_max == x_min else (x[i] - x_min) / (x_max - x_min)
        ax.plot(
            voltage_curve[good],
            current_curve[good],
            color=cmap(frac),
            alpha=0.9,
            lw=1.2,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=x_min, vmax=x_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="X (cm)")

    ax.set_title("Interpolated I–V Curves Along X-Line")
    ax.set_xlabel("Bias (V)")
    ax.set_ylabel("Current (A)")
    ax.grid(True)

    return fig


def read_channel_xy(
    file_obj,
    board,
    channel,
    shotnum_start,
    shotnum_end,
    ny,
    nx,
    nshots,
    nt_full,
    digitizer,
    adc,
    config_name,
    scale_factor=1.0,
    negate=False,
    flipup=False,
):
    """
    Read one digitizer channel and reshape it as (ny, nx, nshots, nt_full).

    This assumes shot order is y-major, then x, then repeated shots at each
    location. If the acquisition order differs, change expected_shape and the
    corresponding coordinate metadata before running the analysis.

    If flipup is True, return the reshaped signal with the y dimension flipped
    using numpy.flipud.
    """
    raw = file_obj.read_data(
        board,
        channel,
        digitizer=digitizer,
        adc=adc,
        config_name=config_name,
        shotnum=slice(shotnum_start, shotnum_end, 1),
    )

    signal = raw["signal"]
    dt = raw.dt.value

    if signal.ndim != 2:
        raise ValueError(
            f"Channel {channel}: expected raw signal with shape (shots, time), got {signal.shape}."
        )

    n_actual_shots, n_actual_time = signal.shape
    print(f"Channel {channel}: raw shape = {signal.shape}")
    print(f"Channel {channel}: expected shots = {shotnum_end - shotnum_start}, actual shots = {n_actual_shots}")
    print(f"Channel {channel}: expected nt_full = {nt_full}, actual time samples = {n_actual_time}")

    if n_actual_shots != (shotnum_end - shotnum_start):
        raise ValueError(
            f"Channel {channel}: shot count mismatch. "
            f"Expected {shotnum_end - shotnum_start}, found {n_actual_shots}. "
            f"Likely wrong data_offset, ny, nx, or nshots."
        )

    if n_actual_time != nt_full:
        raise ValueError(
            f"Channel {channel}: time-length mismatch. Expected {nt_full}, found {n_actual_time}."
        )

    if negate:
        signal = -signal

    expected_shape = (ny, nx, nshots, nt_full)
    if signal.size != np.prod(expected_shape):
        raise ValueError(
            f"Channel {channel}: cannot reshape size {signal.size} into {expected_shape}."
        )

    signal = signal.reshape(expected_shape)
    signal = signal * scale_factor
    if flipup:
        signal = np.flipud(signal)
    return signal, dt


def _as_yx_half_window(half_window):
    if np.isscalar(half_window):
        hw = int(half_window)
        return hw, hw
    if len(half_window) != 2:
        raise ValueError("half_window must be a scalar or a two-item (y, x) tuple.")
    return int(half_window[0]), int(half_window[1])


def reject_spikes_by_local_median_2d(z, half_window=(1, 1), threshold=5.0):
    """Return a mask of spike-like map points using 2D local median comparison."""
    z = np.asarray(z, dtype=float)
    hy, hx = _as_yx_half_window(half_window)
    ny_map, nx_map = z.shape
    is_spike = np.zeros(z.shape, dtype=bool)

    for iy in range(ny_map):
        for ix in range(nx_map):
            if not np.isfinite(z[iy, ix]):
                continue

            y0 = max(0, iy - hy)
            y1 = min(ny_map, iy + hy + 1)
            x0 = max(0, ix - hx)
            x1 = min(nx_map, ix + hx + 1)

            neighborhood = z[y0:y1, x0:x1].copy()
            neighborhood[iy - y0, ix - x0] = np.nan
            neighborhood = neighborhood[np.isfinite(neighborhood)]

            if neighborhood.size < 2:
                continue

            med = np.median(neighborhood)
            if np.abs(z[iy, ix] - med) > threshold:
                is_spike[iy, ix] = True

    return is_spike


def replace_flagged_by_local_median_2d(z, flagged, half_window=(1, 1)):
    """Replace flagged map points with the finite median of nearby unflagged points."""
    z = np.asarray(z, dtype=float).copy()
    flagged = np.asarray(flagged, dtype=bool)
    hy, hx = _as_yx_half_window(half_window)
    ny_map, nx_map = z.shape
    z_out = z.copy()

    for iy, ix in zip(*np.where(flagged & np.isfinite(z))):
        y0 = max(0, iy - hy)
        y1 = min(ny_map, iy + hy + 1)
        x0 = max(0, ix - hx)
        x1 = min(nx_map, ix + hx + 1)

        neighborhood = z[y0:y1, x0:x1]
        neighborhood_flags = flagged[y0:y1, x0:x1]
        vals = neighborhood[np.isfinite(neighborhood) & (~neighborhood_flags)]
        if vals.size > 0:
            z_out[iy, ix] = np.median(vals)

    return z_out


def gaussian_neighbor_smooth_nanaware_2d(z, half_window=(1, 1), sigma=1.0):
    """Smooth a 2D map using local gaussian-like weights, ignoring NaNs."""
    z = np.asarray(z, dtype=float)
    hy, hx = _as_yx_half_window(half_window)
    ny_map, nx_map = z.shape
    z_smooth = np.full(z.shape, np.nan)

    y_offsets = np.arange(-hy, hy + 1, dtype=float)
    x_offsets = np.arange(-hx, hx + 1, dtype=float)
    dy, dx = np.meshgrid(y_offsets, x_offsets, indexing="ij")
    weights_template = np.exp(-0.5 * (dy**2 + dx**2) / float(sigma) ** 2)

    for iy in range(ny_map):
        for ix in range(nx_map):
            if not np.isfinite(z[iy, ix]):
                continue
            y0 = max(0, iy - hy)
            y1 = min(ny_map, iy + hy + 1)
            x0 = max(0, ix - hx)
            x1 = min(nx_map, ix + hx + 1)

            wy0 = y0 - (iy - hy)
            wy1 = wy0 + (y1 - y0)
            wx0 = x0 - (ix - hx)
            wx1 = wx0 + (x1 - x0)

            vals = z[y0:y1, x0:x1]
            weights = weights_template[wy0:wy1, wx0:wx1]
            finite = np.isfinite(vals)
            if np.count_nonzero(finite) == 0:
                continue

            z_smooth[iy, ix] = np.sum(weights[finite] * vals[finite]) / np.sum(weights[finite])

    return z_smooth


def apply_optional_xy_postprocessing(
    te,
    vp,
    vf,
    ies,
    iis,
    n_e=None,
    enable_vp_spike_rejection=True,
    vp_spike_half_window=(1, 1),
    vp_spike_threshold_V=5.0,
    vp_replace_flagged_with_local_median=True,
    enable_neighbor_smoothing=True,
    neighbor_smooth_half_window=(1, 1),
    neighbor_smooth_sigma=1.0,
):
    """
    Optional 2D post-processing of Te(y, x), Vp(y, x), Vf(y, x), I_es(y, x), I_is(y, x), n_e(y, x):
      1) detect/reject Vp spikes using a local 2D median
      2) neighbor-aware 2D smoothing for Te, Vp, Vf, I_es, I_is, and n_e
    """
    te_in = np.asarray(te, dtype=float)
    vp_in = np.asarray(vp, dtype=float)
    vf_in = np.asarray(vf, dtype=float)
    ies_in = np.asarray(ies, dtype=float)
    iis_in = np.asarray(iis, dtype=float)
    n_e_in = np.asarray(n_e, dtype=float) if n_e is not None else None

    te_proc = te_in.copy()
    vp_proc = vp_in.copy()
    vf_proc = vf_in.copy()
    ies_proc = ies_in.copy()
    iis_proc = iis_in.copy()
    n_e_proc = n_e_in.copy() if n_e_in is not None else None

    vp_spike_mask = np.zeros_like(vp_proc, dtype=bool)

    if enable_vp_spike_rejection:
        vp_spike_mask = reject_spikes_by_local_median_2d(
            vp_proc,
            half_window=vp_spike_half_window,
            threshold=vp_spike_threshold_V,
        )
        if vp_replace_flagged_with_local_median:
            vp_proc = replace_flagged_by_local_median_2d(
                vp_proc,
                vp_spike_mask,
                half_window=vp_spike_half_window,
            )

    te_smooth = te_proc.copy()
    vp_smooth = vp_proc.copy()
    vf_smooth = vf_proc.copy()
    ies_smooth = ies_proc.copy()
    iis_smooth = iis_proc.copy()
    n_e_smooth = n_e_proc.copy() if n_e_proc is not None else None

    if enable_neighbor_smoothing:
        te_smooth = gaussian_neighbor_smooth_nanaware_2d(
            te_proc,
            half_window=neighbor_smooth_half_window,
            sigma=neighbor_smooth_sigma,
        )
        vp_smooth = gaussian_neighbor_smooth_nanaware_2d(
            vp_proc,
            half_window=neighbor_smooth_half_window,
            sigma=neighbor_smooth_sigma,
        )
        vf_smooth = gaussian_neighbor_smooth_nanaware_2d(
            vf_proc,
            half_window=neighbor_smooth_half_window,
            sigma=neighbor_smooth_sigma,
        )
        ies_smooth = gaussian_neighbor_smooth_nanaware_2d(
            ies_proc,
            half_window=neighbor_smooth_half_window,
            sigma=neighbor_smooth_sigma,
        )
        iis_smooth = gaussian_neighbor_smooth_nanaware_2d(
            iis_proc,
            half_window=neighbor_smooth_half_window,
            sigma=neighbor_smooth_sigma,
        )
        if n_e_proc is not None:
            n_e_smooth = gaussian_neighbor_smooth_nanaware_2d(
                n_e_proc,
                half_window=neighbor_smooth_half_window,
                sigma=neighbor_smooth_sigma,
            )

    result_dict = {
        "te_raw": te_in,
        "vp_raw": vp_in,
        "vf_raw": vf_in,
        "ies_raw": ies_in,
        "iis_raw": iis_in,
        "te_processed": te_smooth,
        "vp_processed": vp_smooth,
        "vf_processed": vf_smooth,
        "ies_processed": ies_smooth,
        "iis_processed": iis_smooth,
        "vp_spike_mask": vp_spike_mask,
    }
    if n_e_in is not None:
        result_dict["n_e_raw"] = n_e_in
        result_dict["n_e_processed"] = n_e_smooth
    return result_dict


def _plot_map(ax, x_mesh, y_mesh, values, title, cbar_label):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    mesh = None
    if np.any(finite):
        mesh = ax.pcolormesh(x_mesh, y_mesh, values, shading="auto")
        cbar = ax.figure.colorbar(mesh, ax=ax)
        cbar.set_label(cbar_label)
    else:
        finite_x = np.asarray(x_mesh)[np.isfinite(x_mesh)]
        finite_y = np.asarray(y_mesh)[np.isfinite(y_mesh)]
        if finite_x.size:
            ax.set_xlim(np.min(finite_x), np.max(finite_x))
        if finite_y.size:
            ax.set_ylim(np.min(finite_y), np.max(finite_y))
        ax.text(
            0.5,
            0.5,
            "No accepted values",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="0.35",
        )
    ax.set_title(title)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_aspect("equal", adjustable="box")
    return mesh


def render_xy_summary_plot(
    x_mesh,
    y_mesh,
    te,
    vp,
    vf,
    ies,
    iis,
    n_e=None,
    vp_spike_mask=None,
    vp_raw=None,
    te_fit_r2=None,
    te_poor_fit_r2=None,
    analysis_ok=None,
    analysis_status=None,
    shape_factor=None,
    interferometer_profile_y_cm=None,
    title="Langmuir XY-Plane Summary",
):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    _plot_map(axs[0, 0], x_mesh, y_mesh, te.value, "Electron Temperature", "T_e (eV)")
    _plot_map(axs[0, 1], x_mesh, y_mesh, vp.value, "Plasma Potential", "V_p (V)")
    _plot_map(axs[1, 0], x_mesh, y_mesh, vf.value, "Floating Potential", "V_f (V)")
    _plot_map(axs[1, 1], x_mesh, y_mesh, ies.value, "Electron Saturation Current", "I_es (A)")
    _plot_map(axs[2, 0], x_mesh, y_mesh, iis.value, "Ion Saturation Current", "I_is (A)")
    if n_e is not None:
        _plot_map(axs[2, 1], x_mesh, y_mesh, n_e.value, "Electron Density", "n_e (m^-3)")
        if interferometer_profile_y_cm is not None:
            axs[2, 1].axhline(
                interferometer_profile_y_cm,
                color="white",
                linestyle="--",
                linewidth=1.2,
                label=f"interferometer x-line: y={interferometer_profile_y_cm:g} cm",
            )
            axs[2, 1].legend()
        if shape_factor is not None:
            shape_factor_m = u.Quantity(shape_factor).to_value(u.m)
            shape_factor_text = (
                f"X-line shape factor = {shape_factor_m:.4g} m"
                if np.isfinite(shape_factor_m)
                else "X-line shape factor unavailable"
            )
            axs[2, 1].text(
                0.03,
                0.97,
                shape_factor_text,
                transform=axs[2, 1].transAxes,
                ha="left",
                va="top",
                color="black",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
            )
    else:
        axs[2, 1].axis("off")

    # Keep finite estimates visible while marking locations where no shot fit
    # passed the full acceptance policy.
    if analysis_ok is not None:
        fit_accepted = np.asarray(analysis_ok, dtype=bool)
        maps = (
            (axs[0, 0], te.value),
            (axs[0, 1], vp.value),
            (axs[1, 0], vf.value),
            (axs[1, 1], ies.value),
            (axs[2, 0], iis.value),
        )
        if n_e is not None:
            maps += ((axs[2, 1], n_e.value),)
        for ax, values in maps:
            values = np.asarray(values, dtype=float)
            if fit_accepted.shape != values.shape:
                continue
            rejected = np.isfinite(values) & ~fit_accepted
            if np.any(rejected):
                ax.plot(
                    x_mesh[rejected],
                    y_mesh[rejected],
                    "x",
                    color="tab:orange",
                    label="no accepted fits",
                )
                ax.legend()

    if te_fit_r2 is not None and te_poor_fit_r2 is not None:
        poor_te_fit = np.isfinite(te_fit_r2) & (te_fit_r2 < te_poor_fit_r2)
        if np.any(poor_te_fit):
            axs[0, 0].plot(
                x_mesh[poor_te_fit],
                y_mesh[poor_te_fit],
                "rx",
                ms=5,
                mew=1.4,
                label=f"Te fit R^2 < {te_poor_fit_r2:.2f}",
            )
            axs[0, 0].legend()

    if vp_spike_mask is not None and np.any(vp_spike_mask):
        axs[0, 1].plot(
            x_mesh[vp_spike_mask],
            y_mesh[vp_spike_mask],
            "ko",
            ms=3,
            label="flagged Vp spikes",
        )
        axs[0, 1].legend()

    summary_title = title if analysis_status is None else f"{title}\n{analysis_status}"
    fig.suptitle(summary_title, fontsize=16)
    return fig


def render_xy_all_iv_curves_plot(x_coord, y_coord, iv_voltage_grid, iv_current_grid, max_curves=400):
    """Plot a representative subset of interpolated I-V curves colored by y position."""
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    cmap = plt.get_cmap("viridis")
    y_min = np.nanmin(y_coord)
    y_max = np.nanmax(y_coord)

    flat_indices = np.arange(iv_voltage_grid.shape[0] * iv_voltage_grid.shape[1])
    if flat_indices.size > max_curves:
        flat_indices = np.linspace(0, flat_indices.size - 1, max_curves, dtype=int)

    for flat_index in flat_indices:
        iy, ix = np.unravel_index(flat_index, iv_voltage_grid.shape[:2])
        voltage_curve = iv_voltage_grid[iy, ix]
        current_curve = iv_current_grid[iy, ix]
        good = np.isfinite(voltage_curve) & np.isfinite(current_curve)
        if np.count_nonzero(good) < 2:
            continue

        frac = 0.5 if y_max == y_min else (y_coord[iy] - y_min) / (y_max - y_min)
        ax.plot(
            voltage_curve[good],
            current_curve[good],
            color=cmap(frac),
            alpha=0.9,
            lw=1.2,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=y_min, vmax=y_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Y (cm)")

    ax.set_title("Representative Interpolated I-V Curves Across XY Plane")
    ax.set_xlabel("Bias (V)")
    ax.set_ylabel("Current (A)")
    ax.grid(True)

    return fig


# %% Geometry-specific pipeline runners
def run_xline_analysis():
    """Run the configured x-line acquisition and analysis pipeline."""
    # %%
    # Read data

    print(f"Expected Langmuir x-line shots: {n_expected_shots}")
    print(f"shotnum_start = {shotnum_start}, shotnum_end = {shotnum_end}")

    file = lapd.File(filename, silent=True)

    print("Reading voltage data..")
    vsweep_full, dt = read_channel_xline(
        file_obj=file,
        board=board,
        channel=vsweep_channel,
        shotnum_start=shotnum_start,
        shotnum_end=shotnum_end,
        nx=nx,
        nshots=nshots,
        nt_full=nt_full,
        digitizer=digitizer,
        adc=adc,
        config_name=sis_config_name,
        scale_factor=vsweep_attenuation,
        negate=False,
    )
    print(f"Voltage data reshaped to (x, shots, time): {vsweep_full.shape}")

    print("Reading current data..")
    isweep_full, _ = read_channel_xline(
        file_obj=file,
        board=board,
        channel=isweep_channel,
        shotnum_start=shotnum_start,
        shotnum_end=shotnum_end,
        nx=nx,
        nshots=nshots,
        nt_full=nt_full,
        digitizer=digitizer,
        adc=adc,
        config_name=sis_config_name,
        scale_factor=isweep_attenuation / isweep_resistance,
        negate=negate_Isweep_current,
    )
    print(f"Current data reshaped to (x, shots, time): {isweep_full.shape}")

    print("Extracting sweep regions")
    vsweep = vsweep_full[..., sweep_start_index:sweep_end_index + 1]
    isweep = isweep_full[..., sweep_start_index:sweep_end_index + 1]

    if subtract_dc:
        if isweep_dc_offset_start_index is None or isweep_dc_offset_end_index is None:
            raise ValueError(
                "subtract_dc requires an independently measured plasma-off interval"
            )
        if not (
            0
            <= isweep_dc_offset_start_index
            <= isweep_dc_offset_end_index
            < nt_full
        ):
            raise ValueError("the electronics-offset interval is outside the trace")
        if not (
            isweep_dc_offset_end_index < sweep_start_index
            or isweep_dc_offset_start_index > sweep_end_index
        ):
            raise ValueError("the electronics-offset interval must not overlap the I-V sweep")
        print("Subtracting current DC offsets")
        isweep_dc_offset_portion = isweep_full[..., isweep_dc_offset_start_index:isweep_dc_offset_end_index + 1]
        isweep_dc_offsets = np.mean(isweep_dc_offset_portion, axis=-1)
        isweep = isweep - isweep_dc_offsets[..., np.newaxis]

    file.close()
    print("Finished reading and reshaping data.")

    # %%
    # Optional time-domain products for visualization/export. The analysis itself
    # uses unsmoothed samples and performs averaging in voltage bins.

    time = np.arange(nt) * dt

    vsweep_shot = vsweep * u.V
    isweep_shot = isweep * u.A
    vsweep_mean = np.mean(vsweep_shot, axis=1)
    isweep_mean = np.mean(isweep_shot, axis=1)

    isat = np.mean(isweep[..., isat_start_index:isat_end_index + 1], axis=-1)
    isat_mean_values, isat_std_values, isat_count = shot_mean_and_std(isat)
    isat_mean = isat_mean_values * u.A
    isat_std = isat_std_values * u.A

    print("Applying Savitzky-Golay filter along original sweep samples")
    vsweep_shot_smoothed = savgol_filter(
        vsweep_shot.value,
        sg_smooth_bins,
        sg_smooth_order,
        axis=-1
    ) * vsweep_shot.unit

    isweep_shot_smoothed = savgol_filter(
        isweep_shot.value,
        sg_smooth_bins,
        sg_smooth_order,
        axis=-1
    ) * isweep_shot.unit
    vsweep_mean_smoothed = np.mean(vsweep_shot_smoothed, axis=1)
    isweep_mean_smoothed = np.mean(isweep_shot_smoothed, axis=1)
    print("Finished smoothing data.")

    # %%
    # Prepare output arrays

    spatial_shape = (nx,)
    trace_shape = analysis_trace_shape(spatial_shape, nshots, shot_analysis_mode)
    n_traces = int(np.prod(trace_shape))

    te_shot = np.full(trace_shape, np.nan)
    vp_shot = np.full(trace_shape, np.nan)
    vf_shot = np.full(trace_shape, np.nan)
    ies_shot = np.full(trace_shape, np.nan)
    iis_shot = np.full(trace_shape, np.nan)
    n_e_shot = np.full(trace_shape, np.nan)
    analysis_ok_shot = np.zeros(trace_shape, dtype=np.uint8)
    analysis_rejection_counts = Counter()

    te_fit_r2_shot = np.full(trace_shape, np.nan)
    te_fit_rmse_shot = np.full(trace_shape, np.nan)
    te_fit_npts_shot = np.full(trace_shape, np.nan)
    te_fit_vstart_shot = np.full(trace_shape, np.nan)
    te_fit_vstop_shot = np.full(trace_shape, np.nan)
    te_fit_slope_shot = np.full(trace_shape, np.nan)
    te_fit_intercept_shot = np.full(trace_shape, np.nan)
    te_fit_i0_shot = np.full(trace_shape, np.nan)
    te_fit_passed_r2_shot = np.zeros(trace_shape, dtype=np.uint8)
    te_fit_candidate_count_shot = np.zeros(trace_shape, dtype=int)

    iv_voltage_grid_shot = np.full(trace_shape + (iv_npts,), np.nan)
    iv_current_grid_shot = np.full(trace_shape + (iv_npts,), np.nan)
    iv_didv_grid_shot = np.full(trace_shape + (iv_npts,), np.nan)
    iv_fit_mask_grid_shot = np.zeros(trace_shape + (iv_npts,), dtype=np.uint8)

    # %%
    # Robust Langmuir analysis on monotonic I(V) curves

    analysis_tasks = []
    for flat_index in range(n_traces):
        idx = np.unravel_index(flat_index, trace_shape)
        ix, ishot = idx
        bias_values, current_values = prepare_trace_for_analysis(
            vsweep_shot.value,
            isweep_shot.value,
            idx,
            shot_analysis_mode,
            voltage_bin_width,
        )
        include_diagnostic_data = (
            diagnostic_plot_every is not None
            and diagnostic_plot_every > 0
            and ishot == 0
            and ix % diagnostic_plot_every == 0
        )
        analysis_tasks.append(
            (
                flat_index,
                idx,
                bias_values,
                current_values,
                include_diagnostic_data,
                langmuir_analysis_config,
            )
        )

    n_analysis_processes = choose_analysis_process_count(n_traces, analysis_processes)
    use_parallel_analysis = parallel_analysis and n_analysis_processes > 1

    if use_parallel_analysis and "fork" not in mp.get_all_start_methods():
        print("Multiprocessing start method 'fork' is unavailable; falling back to serial analysis.")
        use_parallel_analysis = False

    analysis_processes_used = n_analysis_processes if use_parallel_analysis else 1

    if use_parallel_analysis:
        print(f"Processing {n_traces} Langmuir traces using {analysis_processes_used} worker processes...")
        process_context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=analysis_processes_used,
            mp_context=process_context,
        ) as executor:
            futures = [executor.submit(analyze_trace_worker, task) for task in analysis_tasks]
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                idx = result["idx"]
                ix, ishot = idx
                print(f"Finished trace {result['flat_index'] + 1}/{n_traces} at index {idx} ({completed}/{n_traces} complete)")

                for warning in result["warnings"]:
                    print(f"{warning} at index {idx}")

                if not result["ok"]:
                    reason = (
                        result["warnings"][-1]
                        if result["warnings"]
                        else "unspecified analysis failure"
                    )
                    analysis_rejection_counts[reason] += 1

                if result["iv_voltage_grid"] is None:
                    continue

                analysis_ok_shot[idx] = int(result["ok"])
                # Keep every finite estimate for plotting and export.  The strict
                # quality decision remains available separately in analysis_ok_shot.
                te_shot[idx] = result["te_eV"]
                vp_shot[idx] = result["vp_V"]
                vf_shot[idx] = result["vf_V"]
                ies_shot[idx] = result["ies_A"]
                iis_shot[idx] = result["iis_A"]
                n_e_shot[idx] = result.get("n_e_m3", np.nan)

                te_fit_r2_shot[idx] = result["te_fit_r2"]
                te_fit_rmse_shot[idx] = result["te_fit_rmse"]
                if result["te_fit_candidate_count"]:
                    te_fit_npts_shot[idx] = result["te_fit_npts"]
                te_fit_vstart_shot[idx] = result["te_fit_vstart"]
                te_fit_vstop_shot[idx] = result["te_fit_vstop"]
                te_fit_slope_shot[idx] = result["te_fit_slope"]
                te_fit_intercept_shot[idx] = result["te_fit_intercept"]
                te_fit_i0_shot[idx] = result["te_fit_i0"]
                te_fit_passed_r2_shot[idx] = int(result["te_fit_passed_r2"])
                te_fit_candidate_count_shot[idx] = result["te_fit_candidate_count"]

                iv_voltage_grid_shot[idx + (slice(None),)] = result["iv_voltage_grid"]
                iv_current_grid_shot[idx + (slice(None),)] = result["iv_current_grid"]
                if result["iv_didv_grid"] is not None:
                    iv_didv_grid_shot[idx + (slice(None),)] = result["iv_didv_grid"]
                iv_fit_mask_grid_shot[idx + (slice(None),)] = result["iv_fit_mask"]

                diagnostic_data = result["diagnostic_data"]
                if diagnostic_data is not None:
                    fit_label = (
                        "averaged shots"
                        if shot_analysis_mode == "average"
                        else f"shot {ishot}"
                    )
                    file_label = (
                        "shots_averaged"
                        if shot_analysis_mode == "average"
                        else f"shot{ishot:03d}"
                    )
                    fig = render_analysis_iv_diagnostic_plot(
                        result,
                        f"Diagnostic IV analysis at x index {ix}, {fit_label} "
                        f"(x = {x[ix]:.2f} cm)",
                    )
                    save_diagnostic_figure(
                        fig,
                        diagnostic_plot_output_dir,
                        f"{Path(filename).stem}_iv_diagnostic_ix{ix:03d}_"
                        f"{file_label}_x_{x[ix]:+.2f}cm",
                    )
                    plt.show(block=False)
                    plt.pause(example_pause_seconds)
                    plt.close(fig)
    else:
        print("Processing Langmuir traces serially...")
        for task in analysis_tasks:
            flat_index = task[0]
            idx = task[1]
            ix, ishot = idx
            print(f"Processing trace {flat_index + 1}/{n_traces} at index {idx}...")
            result = analyze_trace_worker(task)

            for warning in result["warnings"]:
                print(f"{warning} at index {idx}")

            if not result["ok"]:
                reason = (
                    result["warnings"][-1]
                    if result["warnings"]
                    else "unspecified analysis failure"
                )
                analysis_rejection_counts[reason] += 1

            if result["iv_voltage_grid"] is None:
                continue

            analysis_ok_shot[idx] = int(result["ok"])
            # Keep every finite estimate for plotting and export.  The strict
            # quality decision remains available separately in analysis_ok_shot.
            te_shot[idx] = result["te_eV"]
            vp_shot[idx] = result["vp_V"]
            vf_shot[idx] = result["vf_V"]
            ies_shot[idx] = result["ies_A"]
            iis_shot[idx] = result["iis_A"]
            n_e_shot[idx] = result.get("n_e_m3", np.nan)

            te_fit_r2_shot[idx] = result["te_fit_r2"]
            te_fit_rmse_shot[idx] = result["te_fit_rmse"]
            if result["te_fit_candidate_count"]:
                te_fit_npts_shot[idx] = result["te_fit_npts"]
            te_fit_vstart_shot[idx] = result["te_fit_vstart"]
            te_fit_vstop_shot[idx] = result["te_fit_vstop"]
            te_fit_slope_shot[idx] = result["te_fit_slope"]
            te_fit_intercept_shot[idx] = result["te_fit_intercept"]
            te_fit_i0_shot[idx] = result["te_fit_i0"]
            te_fit_passed_r2_shot[idx] = int(result["te_fit_passed_r2"])
            te_fit_candidate_count_shot[idx] = result["te_fit_candidate_count"]

            iv_voltage_grid_shot[idx + (slice(None),)] = result["iv_voltage_grid"]
            iv_current_grid_shot[idx + (slice(None),)] = result["iv_current_grid"]
            if result["iv_didv_grid"] is not None:
                iv_didv_grid_shot[idx + (slice(None),)] = result["iv_didv_grid"]
            iv_fit_mask_grid_shot[idx + (slice(None),)] = result["iv_fit_mask"]

            diagnostic_data = result["diagnostic_data"]
            if diagnostic_data is not None:
                fit_label = (
                    "averaged shots"
                    if shot_analysis_mode == "average"
                    else f"shot {ishot}"
                )
                file_label = (
                    "shots_averaged"
                    if shot_analysis_mode == "average"
                    else f"shot{ishot:03d}"
                )
                fig = render_analysis_iv_diagnostic_plot(
                    result,
                    f"Diagnostic IV analysis at x index {ix}, {fit_label} "
                    f"(x = {x[ix]:.2f} cm)",
                )
                save_diagnostic_figure(
                    fig,
                    diagnostic_plot_output_dir,
                    f"{Path(filename).stem}_iv_diagnostic_ix{ix:03d}_"
                    f"{file_label}_x_{x[ix]:+.2f}cm",
                )
                plt.show(block=False)
                plt.pause(example_pause_seconds)
                plt.close(fig)

    # Compatibility export: this mask identifies locations with a reported Te fit.
    # Consult analysis_ok_shot to distinguish quality-accepted and rejected fits.
    statistics_fit_mask = np.isfinite(te_shot)
    te_values, te_std_values, fit_count = shot_mean_and_std(te_shot)
    vp_values, vp_std_values, _ = shot_mean_and_std(vp_shot)
    vf_values, vf_std_values, _ = shot_mean_and_std(vf_shot)
    ies_values, ies_std_values, _ = shot_mean_and_std(ies_shot)
    iis_values, iis_std_values, _ = shot_mean_and_std(iis_shot)
    n_e_values, n_e_std_values, _ = shot_mean_and_std(n_e_shot)

    te = te_values * u.eV
    vp = vp_values * u.V
    vf = vf_values * u.V
    ies = ies_values * u.A
    iis = iis_values * u.A
    n_e = n_e_values * u.m**-3
    te_std = te_std_values * u.eV
    vp_std = vp_std_values * u.V
    vf_std = vf_std_values * u.V
    ies_std = ies_std_values * u.A
    iis_std = iis_std_values * u.A
    n_e_std = n_e_std_values * u.m**-3

    te_fit_r2, te_fit_r2_std, te_fit_count = shot_mean_and_std(te_fit_r2_shot)
    te_fit_rmse, te_fit_rmse_std, _ = shot_mean_and_std(te_fit_rmse_shot)
    te_fit_npts, te_fit_npts_std, _ = shot_mean_and_std(te_fit_npts_shot)
    te_fit_vstart, te_fit_vstart_std, _ = shot_mean_and_std(te_fit_vstart_shot)
    te_fit_vstop, te_fit_vstop_std, _ = shot_mean_and_std(te_fit_vstop_shot)
    te_fit_slope, te_fit_slope_std, _ = shot_mean_and_std(te_fit_slope_shot)
    te_fit_intercept, te_fit_intercept_std, _ = shot_mean_and_std(te_fit_intercept_shot)
    te_fit_i0, te_fit_i0_std, _ = shot_mean_and_std(te_fit_i0_shot)
    te_fit_passed_r2 = np.any(te_fit_passed_r2_shot.astype(bool), axis=-1).astype(np.uint8)
    te_fit_candidate_count = np.sum(te_fit_candidate_count_shot, axis=-1)
    analysis_valid_count = np.sum(analysis_ok_shot, axis=-1)
    analysis_ok = (analysis_valid_count > 0).astype(np.uint8)
    accepted_trace_count = int(np.sum(analysis_ok_shot))
    reported_trace_count = int(np.count_nonzero(np.isfinite(te_shot)))
    fit_description = (
        "shot-averaged fits"
        if shot_analysis_mode == "average"
        else "per-shot fits"
    )
    analysis_status = (
        f"Reported {reported_trace_count}/{n_traces} finite {fit_description}; "
        f"accepted {accepted_trace_count}/{n_traces} {fit_description}"
    )
    if analysis_rejection_counts:
        common_reason, common_count = analysis_rejection_counts.most_common(1)[0]
        analysis_status += f"; top rejection ({common_count}): {common_reason}"
    print(f"Analysis summary: {analysis_status}")

    # Backward-compatible representative I-V products are visualization-only shot
    # means.  They are not used for physics, because individual sweep endpoints
    # need not coincide.  Full per-shot grids are exported separately below.
    iv_voltage_grid = np.nanmean(iv_voltage_grid_shot, axis=-2)
    iv_current_grid = np.nanmean(iv_current_grid_shot, axis=-2)
    iv_didv_grid = np.nanmean(iv_didv_grid_shot, axis=-2)
    iv_fit_mask_grid = np.any(iv_fit_mask_grid_shot.astype(bool), axis=-2).astype(np.uint8)

    shot_statistics_available, shot_statistics_stage = shot_statistics_metadata(
        shot_analysis_mode,
        nshots,
    )
    if shot_analysis_mode == "average":
        isat_std = np.full(spatial_shape, np.nan) * u.A
        unavailable = unavailable_per_shot_fit_products(
            spatial_shape,
            nshots,
            iv_npts,
        )
        te_shot = unavailable["te_shot"]
        vp_shot = unavailable["vp_shot"]
        vf_shot = unavailable["vf_shot"]
        ies_shot = unavailable["ies_shot"]
        iis_shot = unavailable["iis_shot"]
        n_e_shot = unavailable["n_e_shot"]
        analysis_ok_shot = unavailable["analysis_ok_shot"]
        statistics_fit_mask = unavailable["statistics_fit_mask"]
        te_fit_r2_shot = unavailable["te_fit_r2_shot"]
        te_fit_rmse_shot = unavailable["te_fit_rmse_shot"]
        te_fit_npts_shot = unavailable["te_fit_npts_shot"]
        te_fit_vstart_shot = unavailable["te_fit_vstart_shot"]
        te_fit_vstop_shot = unavailable["te_fit_vstop_shot"]
        te_fit_slope_shot = unavailable["te_fit_slope_shot"]
        te_fit_intercept_shot = unavailable["te_fit_intercept_shot"]
        te_fit_i0_shot = unavailable["te_fit_i0_shot"]
        te_fit_passed_r2_shot = unavailable["te_fit_passed_r2_shot"]
        te_fit_candidate_count_shot = unavailable["te_fit_candidate_count_shot"]
        iv_voltage_grid_shot = unavailable["iv_voltage_grid_shot"]
        iv_current_grid_shot = unavailable["iv_current_grid_shot"]
        iv_didv_grid_shot = unavailable["iv_didv_grid_shot"]
        iv_fit_mask_grid_shot = unavailable["iv_fit_mask_grid_shot"]

    print(
        "Finished "
        + (
            "shot-averaged Langmuir probe analysis"
            if shot_analysis_mode == "average"
            else "per-shot Langmuir probe analysis and spatial statistics"
        )
    )

    # %%
    # Optional post-processing along x

    post = apply_optional_x_postprocessing(
        x_coord=x,
        te=te.value,
        vp=vp.value,
        vf=vf.value,
        ies=ies.value,
        iis=iis.value,
        n_e=n_e.value,
        enable_vp_spike_rejection=enable_vp_spike_rejection,
        vp_spike_half_window=vp_spike_half_window,
        vp_spike_threshold_V=vp_spike_threshold_V,
        vp_replace_flagged_with_local_interp=vp_replace_flagged_with_local_interp,
        enable_neighbor_smoothing=enable_neighbor_smoothing,
        neighbor_smooth_half_window=neighbor_smooth_half_window,
        neighbor_smooth_sigma=neighbor_smooth_sigma,
    )

    te_raw = post["te_raw"] * u.eV
    vp_raw = post["vp_raw"] * u.V
    vf_raw = post["vf_raw"] * u.V
    ies_raw = post["ies_raw"] * u.A if "ies_raw" in post else None
    iis_raw = post["iis_raw"] * u.A if "iis_raw" in post else None
    n_e_raw = post["n_e_raw"] * u.m**-3 if "n_e_raw" in post else None

    te_processed = post["te_processed"] * u.eV
    vp_processed = post["vp_processed"] * u.V
    vf_processed = post["vf_processed"] * u.V
    ies_processed = post["ies_processed"] * u.A if "ies_processed" in post else None
    iis_processed = post["iis_processed"] * u.A if "iis_processed" in post else None
    n_e_processed = post["n_e_processed"] * u.m**-3 if "n_e_processed" in post else None

    vp_spike_mask = post["vp_spike_mask"]
    # Plots choose processed or raw results
    te_plot = te_processed if enable_neighbor_smoothing else te_raw
    vp_plot = vp_processed if (enable_neighbor_smoothing or enable_vp_spike_rejection) else vp_raw
    vf_plot = vf_processed if enable_neighbor_smoothing else vf_raw
    ies_plot = ies_processed if (ies_processed is not None and enable_neighbor_smoothing) else ies_raw
    iis_plot = iis_processed if (iis_processed is not None and enable_neighbor_smoothing) else iis_raw
    n_e_plot = n_e_processed if (n_e_processed is not None and enable_neighbor_smoothing) else n_e_raw

    # Use the same raw or post-processed density profile displayed in the summary.
    shape_factor = None
    electron_density_normalized = None
    if calculate_shape_factor:
        shape_factor = electron_density_profile_shape_factor(x, n_e_plot)
        (
            electron_density_normalized,
            n_e_plot,
            density_scale,
        ) = scale_density_profile_to_interferometer(
            n_e_plot,
            interferometer_scaling,
            shape_factor,
        )
        if np.isfinite(density_scale):
            # Preserve consistency between the calibrated density and its
            # shot-to-shot uncertainty in the summary plot.
            n_e_std = n_e_std * density_scale
            print(
                "Scaled normalized electron density to the interferometer "
                f"line-integrated density; peak n_e = {np.nanmax(n_e_plot):.4g}."
            )
        else:
            print(
                "Warning: electron density could not be normalized and scaled "
                "because its maximum, shape factor, or interferometer scaling "
                "is not finite and positive."
            )
            n_e_std = np.full(n_e_std.shape, np.nan) * u.m**-3
    # %%
    # Plot results

    plot_png_bytes = None
    all_iv_plot_png_bytes = None
    summary_plot_path = None
    all_iv_plot_path = None

    if plot_results:
        fig = render_xline_summary_plot(
            x,
            te_plot,
            vp_plot,
            vf_plot,
            ies_plot,
            iis_plot,
            n_e_plot,
            te_std=te_std,
            vp_std=vp_std,
            vf_std=vf_std,
            ies_std=ies_std,
            iis_std=iis_std,
            n_e_std=n_e_std,
            plot_stds=plot_summary_stds and shot_statistics_available,
            vp_spike_mask=vp_spike_mask,
            vp_raw=vp_raw,
            te_fit_r2=te_fit_r2,
            te_poor_fit_r2=te_min_r2,
            analysis_ok=analysis_ok,
            analysis_status=analysis_status,
            shape_factor=shape_factor,
        )

        plot_buffer = io.BytesIO()
        fig.savefig(plot_buffer, format="png", dpi=600)
        plot_png_bytes = plot_buffer.getvalue()
        plot_buffer.close()

        summary_plot_path = save_diagnostic_figure(
            fig,
            diagnostic_plot_output_dir,
            f"{Path(filename).stem}_langmuir_xline_summary",
        )
        plt.show()

    if make_all_iv_diagnostic_plot:
        fig_iv = render_xline_all_iv_curves_plot(x, iv_voltage_grid, iv_current_grid)

        iv_plot_buffer = io.BytesIO()
        fig_iv.savefig(iv_plot_buffer, format="png", dpi=600)
        all_iv_plot_png_bytes = iv_plot_buffer.getvalue()
        iv_plot_buffer.close()

        all_iv_plot_path = save_diagnostic_figure(
            fig_iv,
            diagnostic_plot_output_dir,
            f"{Path(filename).stem}_all_interpolated_iv_curves",
        )
        plt.show()

    # %%
    # Save processed Langmuir x-line results

    output_h5_path = Path(filename)
    output_npz_path = output_h5_path.with_name(f"{output_h5_path.stem}_langmuir_xline.npz")

    if save_results:
        with open_h5_for_update(output_h5_path) as h5f:
            if "langmuir_xline" in h5f:
                del h5f["langmuir_xline"]

            grp = h5f.create_group("langmuir_xline")

            grp.create_dataset("x_cm", data=x)
            grp.create_dataset("time_s", data=time)

            # Primary exported profiles reflect current optional processing choices
            grp.create_dataset("te_eV", data=te_plot.value)
            grp.create_dataset("vp_V", data=vp_plot.value)
            grp.create_dataset("vf_V", data=vf_plot.value)
            grp.create_dataset("ies_A", data=ies_plot.value)
            grp.create_dataset("iis_A", data=iis_plot.value)
            # Electron density map (m^-3) if available
            try:
                if n_e_plot is not None:
                    grp.create_dataset("n_e_m3", data=getattr(n_e_plot, "value", n_e_plot))
            except Exception:
                pass
            if calculate_shape_factor and electron_density_normalized is not None:
                grp.create_dataset(
                    "n_e_normalized",
                    data=electron_density_normalized,
                )
            grp.create_dataset("isat_A", data=isat_mean.value)

            # Per-shot fitted quantities and their per-location sample deviations
            grp.create_dataset("te_shot_eV", data=te_shot)
            grp.create_dataset("vp_shot_V", data=vp_shot)
            grp.create_dataset("vf_shot_V", data=vf_shot)
            grp.create_dataset("ies_shot_A", data=ies_shot)
            grp.create_dataset("iis_shot_A", data=iis_shot)
            grp.create_dataset("n_e_shot_m3", data=n_e_shot)
            grp.create_dataset("isat_shot_A", data=isat)
            grp.create_dataset("te_std_eV", data=te_std.value)
            grp.create_dataset("vp_std_V", data=vp_std.value)
            grp.create_dataset("vf_std_V", data=vf_std.value)
            grp.create_dataset("ies_std_A", data=ies_std.value)
            grp.create_dataset("iis_std_A", data=iis_std.value)
            grp.create_dataset("n_e_std_m3", data=n_e_std.value)
            grp.create_dataset("isat_std_A", data=isat_std.value)
            grp.create_dataset("fit_count", data=fit_count)
            grp.create_dataset("analysis_ok", data=analysis_ok)
            grp.create_dataset("analysis_valid_count", data=analysis_valid_count)
            grp.create_dataset("analysis_ok_shot", data=analysis_ok_shot)
            grp.create_dataset("statistics_fit_mask", data=statistics_fit_mask.astype(np.uint8))

            # Raw extracted profiles
            grp.create_dataset("te_raw_eV", data=te_raw.value)
            grp.create_dataset("vp_raw_V", data=vp_raw.value)
            grp.create_dataset("vf_raw_V", data=vf_raw.value)
            grp.create_dataset("ies_raw_A", data=ies_raw.value if ies_raw is not None else np.full(spatial_shape, np.nan))
            grp.create_dataset("iis_raw_A", data=iis_raw.value if iis_raw is not None else np.full(spatial_shape, np.nan))
            grp.create_dataset("n_e_raw_m3", data=n_e_raw.value if n_e_raw is not None else np.full(spatial_shape, np.nan))

            # Post-processed profiles
            grp.create_dataset("te_processed_eV", data=te_processed.value)
            grp.create_dataset("vp_processed_V", data=vp_processed.value)
            grp.create_dataset("vf_processed_V", data=vf_processed.value)
            grp.create_dataset("ies_processed_A", data=ies_processed.value if ies_processed is not None else np.full(spatial_shape, np.nan))
            grp.create_dataset("iis_processed_A", data=iis_processed.value if iis_processed is not None else np.full(spatial_shape, np.nan))
            grp.create_dataset("n_e_processed_m3", data=n_e_processed.value if n_e_processed is not None else np.full(spatial_shape, np.nan))

            # Spike diagnostics
            grp.create_dataset("vp_spike_mask", data=vp_spike_mask.astype(np.uint8))

            # Shot-averaged original sweep curves
            grp.create_dataset("vsweep_mean_V", data=vsweep_mean.value)
            grp.create_dataset("isweep_mean_A", data=isweep_mean.value)
            grp.create_dataset("vsweep_mean_smoothed_V", data=vsweep_mean_smoothed.value)
            grp.create_dataset("isweep_mean_smoothed_A", data=isweep_mean_smoothed.value)
            grp.create_dataset("vsweep_shot_smoothed_V", data=vsweep_shot_smoothed.value)
            grp.create_dataset("isweep_shot_smoothed_A", data=isweep_shot_smoothed.value)

            # Interpolated monotonic IV analysis products
            grp.create_dataset("iv_voltage_grid_V", data=iv_voltage_grid)
            grp.create_dataset("iv_current_grid_A", data=iv_current_grid)
            grp.create_dataset("iv_didv_grid_A_per_V", data=iv_didv_grid)
            grp.create_dataset("iv_te_fit_mask", data=iv_fit_mask_grid)
            grp.create_dataset("iv_voltage_grid_shot_V", data=iv_voltage_grid_shot)
            grp.create_dataset("iv_current_grid_shot_A", data=iv_current_grid_shot)
            grp.create_dataset("iv_didv_grid_shot_A_per_V", data=iv_didv_grid_shot)
            grp.create_dataset("iv_te_fit_mask_shot", data=iv_fit_mask_grid_shot)

            # Fit diagnostics
            grp.create_dataset("te_fit_r2", data=te_fit_r2)
            grp.create_dataset("te_fit_rmse_logI", data=te_fit_rmse)
            grp.create_dataset("te_fit_npts", data=te_fit_npts)
            grp.create_dataset("te_fit_vstart_V", data=te_fit_vstart)
            grp.create_dataset("te_fit_vstop_V", data=te_fit_vstop)
            grp.create_dataset("te_fit_slope_logI_per_V", data=te_fit_slope)
            grp.create_dataset("te_fit_intercept_logI", data=te_fit_intercept)
            grp.create_dataset("te_fit_i0_A", data=te_fit_i0)
            grp.create_dataset("te_fit_passed_r2", data=te_fit_passed_r2)
            grp.create_dataset("te_fit_candidate_count", data=te_fit_candidate_count)
            grp.create_dataset("te_fit_r2_std", data=te_fit_r2_std)
            grp.create_dataset("te_fit_rmse_logI_std", data=te_fit_rmse_std)
            grp.create_dataset("te_fit_npts_std", data=te_fit_npts_std)
            grp.create_dataset("te_fit_vstart_std_V", data=te_fit_vstart_std)
            grp.create_dataset("te_fit_vstop_std_V", data=te_fit_vstop_std)
            grp.create_dataset("te_fit_slope_std_logI_per_V", data=te_fit_slope_std)
            grp.create_dataset("te_fit_intercept_std_logI", data=te_fit_intercept_std)
            grp.create_dataset("te_fit_i0_std_A", data=te_fit_i0_std)
            grp.create_dataset("te_fit_count", data=te_fit_count)
            grp.create_dataset("te_fit_r2_shot", data=te_fit_r2_shot)
            grp.create_dataset("te_fit_rmse_logI_shot", data=te_fit_rmse_shot)
            grp.create_dataset("te_fit_npts_shot", data=te_fit_npts_shot)
            grp.create_dataset("te_fit_vstart_shot_V", data=te_fit_vstart_shot)
            grp.create_dataset("te_fit_vstop_shot_V", data=te_fit_vstop_shot)
            grp.create_dataset("te_fit_slope_shot_logI_per_V", data=te_fit_slope_shot)
            grp.create_dataset("te_fit_intercept_shot_logI", data=te_fit_intercept_shot)
            grp.create_dataset("te_fit_i0_shot_A", data=te_fit_i0_shot)
            grp.create_dataset("te_fit_passed_r2_shot", data=te_fit_passed_r2_shot)
            grp.create_dataset("te_fit_candidate_count_shot", data=te_fit_candidate_count_shot)

            if subtract_dc:
                grp.create_dataset("isweep_dc_offsets_A", data=isweep_dc_offsets)
            grp.create_dataset("xline_shape_info", data=np.array([nx, nshots, nt_full, nt, iv_npts], dtype=np.int64))
            grp.create_dataset("trace_spatial_shape", data=np.array(spatial_shape, dtype=np.int64))

            if plot_results and plot_png_bytes is None:
                fig = render_xline_summary_plot(
                    x,
                    te_plot,
                    vp_plot,
                    vf_plot,
                    ies_plot,
                    iis_plot,
                    n_e_plot,
                    te_std=te_std,
                    vp_std=vp_std,
                    vf_std=vf_std,
                    ies_std=ies_std,
                    iis_std=iis_std,
                    n_e_std=n_e_std,
                    plot_stds=plot_summary_stds and shot_statistics_available,
                    vp_spike_mask=vp_spike_mask,
                    vp_raw=vp_raw,
                    te_fit_r2=te_fit_r2,
                    te_poor_fit_r2=te_min_r2,
                    analysis_ok=analysis_ok,
                    analysis_status=analysis_status,
                    shape_factor=shape_factor,
                )
                plot_buffer = io.BytesIO()
                fig.savefig(plot_buffer, format="png", dpi=600)
                plot_png_bytes = plot_buffer.getvalue()
                plot_buffer.close()
                plt.close(fig)

            if plot_png_bytes is not None:
                grp.create_dataset(
                    "summary_plot_png",
                    data=np.frombuffer(plot_png_bytes, dtype=np.uint8)
                )
                grp["summary_plot_png"].attrs["mime_type"] = "image/png"
                grp["summary_plot_png"].attrs["description"] = "Rendered Langmuir x-line summary plot."

            if all_iv_plot_png_bytes is not None:
                grp.create_dataset(
                    "all_iv_curves_plot_png",
                    data=np.frombuffer(all_iv_plot_png_bytes, dtype=np.uint8)
                )
                grp["all_iv_curves_plot_png"].attrs["mime_type"] = "image/png"
                grp["all_iv_curves_plot_png"].attrs["description"] = (
                    "Diagnostic plot of all interpolated I-V curves colored by x position."
                )

            grp.attrs["source_file"] = str(filename)
            grp.attrs["geometry"] = "x_line"
            grp.attrs["fixed_y_cm"] = 0.0

            grp.attrs["nx"] = nx
            grp.attrs["n_traces"] = nx * nshots
            grp.attrs["n_analysis_traces"] = n_traces
            grp.attrs["n_acquired_traces"] = nx * nshots
            grp.attrs["trace_spatial_ndim"] = len(spatial_shape)
            grp.attrs["nshots"] = nshots
            grp.attrs["shot_analysis_mode"] = shot_analysis_mode
            grp.attrs["nt_full"] = nt_full
            grp.attrs["nt_sweep"] = nt
            grp.attrs["dt_s"] = dt

            grp.attrs["digitizer"] = digitizer
            grp.attrs["adc"] = adc
            grp.attrs["config_name"] = sis_config_name
            grp.attrs["board"] = board
            grp.attrs["vsweep_channel"] = vsweep_channel
            grp.attrs["isweep_channel"] = isweep_channel
            grp.attrs["vsweep_attenuation"] = vsweep_attenuation
            grp.attrs["isweep_attenuation"] = isweep_attenuation
            grp.attrs["isweep_resistance_ohm"] = isweep_resistance

            # Record probe geometry used for density calculation if available
            try:
                grp.attrs["probe_area_m2"] = float(probe_area.to(u.m**2).value)
            except Exception:
                pass

            grp.attrs["sweep_start_index"] = sweep_start_index
            grp.attrs["sweep_end_index"] = sweep_end_index
            grp.attrs["isat_start_index"] = isat_start_index
            grp.attrs["isat_end_index"] = isat_end_index

            if subtract_dc:
                grp.attrs["dc_offset_start_index"] = isweep_dc_offset_start_index
                grp.attrs["dc_offset_end_index"] = isweep_dc_offset_end_index

            grp.attrs["sg_smooth_bins"] = sg_smooth_bins
            grp.attrs["sg_smooth_order"] = sg_smooth_order
            grp.attrs["iv_npts"] = iv_npts
            grp.attrs["voltage_bin_width_V"] = voltage_bin_width
            grp.attrs["vp_smoothing"] = "none" if vp_smoothing is None else str(vp_smoothing)
            grp.attrs["vp_smoothing_width_V"] = vp_smoothing_width_V
            grp.attrs["vp_savgol_order"] = vp_savgol_order
            grp.attrs["ies_method"] = ies_method
            grp.attrs["ies_definition"] = describe_ies_method(ies_method)
            grp.attrs["parallel_analysis"] = int(parallel_analysis)
            grp.attrs["analysis_processes_requested"] = (
                -1 if analysis_processes is None else int(analysis_processes)
            )
            grp.attrs["analysis_processes_used"] = analysis_processes_used
            grp.attrs["analysis_start_method"] = "fork" if use_parallel_analysis else "serial"

            grp.attrs["te_min_points"] = te_min_points
            grp.attrs["te_margin_from_vp_V"] = te_margin_from_vp
            grp.attrs["te_current_floor_frac"] = te_current_floor_frac
            grp.attrs["te_subtract_i0"] = int(te_subtract_i0)
            grp.attrs["te_min_eV"] = te_min_eV
            grp.attrs["te_max_eV"] = te_max_eV
            grp.attrs["te_min_r2"] = te_min_r2
            grp.attrs["ion_min_snr"] = ion_min_snr
            grp.attrs["analysis_model"] = (
                "single-temperature semilog fit; median low-bias ion estimate; "
                + describe_ies_method(ies_method)
            )
            grp.attrs["te_fit_i0_definition"] = "median low-bias ion-region current"
            grp.attrs["iv_npts_role"] = "fixed-size diagnostics only"
            grp.attrs["current_zero_calibrated"] = int(current_zero_calibrated)
            grp.attrs["negate_Isweep_current"] = int(negate_Isweep_current)
            grp.attrs["calculate_shape_factor"] = int(calculate_shape_factor)
            grp.attrs["shape_factor_m"] = (
                np.nan if shape_factor is None else shape_factor.to_value(u.m)
            )
            grp.attrs["interferometer_line_integrated_density_m2"] = (
                interferometer_scaling.to_value(u.m**-2)
            )
            grp.attrs["enforce_ideal_model_checks"] = int(enforce_ideal_model_checks)
            grp.attrs["subtract_dc"] = int(subtract_dc)
            grp.attrs["shot_standard_deviation_ddof"] = 1
            grp.attrs["shot_statistics_available"] = int(
                shot_statistics_available
            )
            grp.attrs["per_shot_axis_order"] = "x,shot"
            grp.attrs["shot_statistics_stage"] = shot_statistics_stage
            grp.attrs["profile_value_policy"] = (
                "finite estimates retained; analysis_ok records fit acceptance"
            )

            grp.attrs["enable_vp_spike_rejection"] = int(enable_vp_spike_rejection)
            grp.attrs["vp_spike_half_window"] = vp_spike_half_window
            grp.attrs["vp_spike_threshold_V"] = vp_spike_threshold_V
            grp.attrs["vp_replace_flagged_with_local_interp"] = int(vp_replace_flagged_with_local_interp)

            grp.attrs["enable_neighbor_smoothing"] = int(enable_neighbor_smoothing)
            grp.attrs["neighbor_smooth_half_window"] = neighbor_smooth_half_window
            grp.attrs["neighbor_smooth_sigma"] = neighbor_smooth_sigma

            grp.attrs["diagnostic_plot_every"] = diagnostic_plot_every
            grp.attrs["diagnostic_plot_output_dir"] = str(diagnostic_plot_output_dir)
            if summary_plot_path is not None:
                grp.attrs["summary_plot_path"] = str(summary_plot_path)
            if all_iv_plot_path is not None:
                grp.attrs["all_iv_plot_path"] = str(all_iv_plot_path)

            grp.attrs["make_all_iv_diagnostic_plot"] = int(make_all_iv_diagnostic_plot)
            grp.attrs["data_offset_shots"] = data_offset

        xline_y = np.zeros_like(x)
        xline_npz_data = {
            "source_file": np.array(str(filename)),
            "geometry": np.array("x_line"),
            "shot_analysis_mode": np.array(shot_analysis_mode),
            "shot_statistics_available": np.array(
                int(shot_statistics_available)
            ),
            "n_analysis_traces": np.array(n_traces),
            "n_acquired_traces": np.array(nx * nshots),
            "fixed_y_cm": np.array(0.0),
            "x_cm": x,
            "y_cm": xline_y,
            "X_cm": x,
            "Y_cm": xline_y,
            "time_s": time,
            "te_eV": te_plot.value,
            "vp_V": vp_plot.value,
            "vf_V": vf_plot.value,
            "ies_A": ies_plot.value,
            "iis_A": iis_plot.value,
            "isat_A": isat_mean.value,
            "te_shot_eV": te_shot,
            "vp_shot_V": vp_shot,
            "vf_shot_V": vf_shot,
            "ies_shot_A": ies_shot,
            "iis_shot_A": iis_shot,
            "n_e_shot_m3": n_e_shot,
            "isat_shot_A": isat,
            "te_std_eV": te_std.value,
            "vp_std_V": vp_std.value,
            "vf_std_V": vf_std.value,
            "ies_std_A": ies_std.value,
            "iis_std_A": iis_std.value,
            "n_e_std_m3": n_e_std.value,
            "isat_std_A": isat_std.value,
            "fit_count": fit_count,
            "analysis_ok": analysis_ok,
            "analysis_valid_count": analysis_valid_count,
            "analysis_ok_shot": analysis_ok_shot,
            "statistics_fit_mask": statistics_fit_mask.astype(np.uint8),
            "n_e_m3": (n_e_plot.value if hasattr(n_e_plot, "value") else n_e_plot) if n_e_plot is not None else np.full_like(x, np.nan),
            "n_e_normalized": (
                electron_density_normalized
                if electron_density_normalized is not None
                else np.full_like(x, np.nan, dtype=float)
            ),
            "te_raw_eV": te_raw.value,
            "vp_raw_V": vp_raw.value,
            "vf_raw_V": vf_raw.value,
            "ies_raw_A": ies_raw.value if ies_raw is not None else np.full(spatial_shape, np.nan),
            "iis_raw_A": iis_raw.value if iis_raw is not None else np.full(spatial_shape, np.nan),
            "n_e_raw_m3": n_e_raw.value if n_e_raw is not None else np.full(spatial_shape, np.nan),
            "te_processed_eV": te_processed.value,
            "vp_processed_V": vp_processed.value,
            "vf_processed_V": vf_processed.value,
            "ies_processed_A": ies_processed.value if ies_processed is not None else np.full(spatial_shape, np.nan),
            "iis_processed_A": iis_processed.value if iis_processed is not None else np.full(spatial_shape, np.nan),
            "n_e_processed_m3": n_e_processed.value if n_e_processed is not None else np.full(spatial_shape, np.nan),
            "vp_spike_mask": vp_spike_mask.astype(np.uint8),
            "vsweep_mean_V": vsweep_mean.value,
            "isweep_mean_A": isweep_mean.value,
            "vsweep_mean_smoothed_V": vsweep_mean_smoothed.value,
            "isweep_mean_smoothed_A": isweep_mean_smoothed.value,
            "vsweep_shot_smoothed_V": vsweep_shot_smoothed.value,
            "isweep_shot_smoothed_A": isweep_shot_smoothed.value,
            "iv_voltage_grid_V": iv_voltage_grid,
            "iv_current_grid_A": iv_current_grid,
            "iv_didv_grid_A_per_V": iv_didv_grid,
            "iv_te_fit_mask": iv_fit_mask_grid,
            "iv_voltage_grid_shot_V": iv_voltage_grid_shot,
            "iv_current_grid_shot_A": iv_current_grid_shot,
            "iv_didv_grid_shot_A_per_V": iv_didv_grid_shot,
            "iv_te_fit_mask_shot": iv_fit_mask_grid_shot,
            "te_fit_r2": te_fit_r2,
            "te_fit_rmse_logI": te_fit_rmse,
            "te_fit_npts": te_fit_npts,
            "te_fit_vstart_V": te_fit_vstart,
            "te_fit_vstop_V": te_fit_vstop,
            "te_fit_slope_logI_per_V": te_fit_slope,
            "te_fit_intercept_logI": te_fit_intercept,
            "te_fit_i0_A": te_fit_i0,
            "te_subtract_i0": np.array(int(te_subtract_i0)),
            "te_fit_passed_r2": te_fit_passed_r2,
            "te_fit_candidate_count": te_fit_candidate_count,
            "te_fit_r2_std": te_fit_r2_std,
            "te_fit_rmse_logI_std": te_fit_rmse_std,
            "te_fit_npts_std": te_fit_npts_std,
            "te_fit_vstart_std_V": te_fit_vstart_std,
            "te_fit_vstop_std_V": te_fit_vstop_std,
            "te_fit_slope_std_logI_per_V": te_fit_slope_std,
            "te_fit_intercept_std_logI": te_fit_intercept_std,
            "te_fit_i0_std_A": te_fit_i0_std,
            "te_fit_count": te_fit_count,
            "te_fit_r2_shot": te_fit_r2_shot,
            "te_fit_rmse_logI_shot": te_fit_rmse_shot,
            "te_fit_npts_shot": te_fit_npts_shot,
            "te_fit_vstart_shot_V": te_fit_vstart_shot,
            "te_fit_vstop_shot_V": te_fit_vstop_shot,
            "te_fit_slope_shot_logI_per_V": te_fit_slope_shot,
            "te_fit_intercept_shot_logI": te_fit_intercept_shot,
            "te_fit_i0_shot_A": te_fit_i0_shot,
            "te_fit_passed_r2_shot": te_fit_passed_r2_shot,
            "te_fit_candidate_count_shot": te_fit_candidate_count_shot,
            "shot_standard_deviation_ddof": np.array(1),
            "per_shot_axis_order": np.array("x,shot"),
            "shot_statistics_stage": np.array(shot_statistics_stage),
            "profile_value_policy": np.array(
                "finite estimates retained; analysis_ok records fit acceptance"
            ),
            "xline_shape_info": np.array([nx, nshots, nt_full, nt, iv_npts], dtype=np.int64),
            "trace_spatial_shape": np.array(spatial_shape, dtype=np.int64),
            "dt_s": np.array(dt),
            "te_min_r2": np.array(te_min_r2),
            "ion_min_snr": np.array(ion_min_snr),
            "ies_method": np.array(ies_method),
            "ies_definition": np.array(describe_ies_method(ies_method)),
            "analysis_model": np.array(
                "single-temperature semilog fit; median low-bias ion estimate; "
                + describe_ies_method(ies_method)
            ),
            "te_fit_i0_definition": np.array("median low-bias ion-region current"),
            "iv_npts_role": np.array("fixed-size diagnostics only"),
            "current_zero_calibrated": np.array(int(current_zero_calibrated)),
            "negate_Isweep_current": np.array(int(negate_Isweep_current)),
            "calculate_shape_factor": np.array(int(calculate_shape_factor)),
            "shape_factor_m": np.array(
                np.nan if shape_factor is None else shape_factor.to_value(u.m)
            ),
            "interferometer_line_integrated_density_m2": np.array(
                interferometer_scaling.to_value(u.m**-2)
            ),
            "enforce_ideal_model_checks": np.array(int(enforce_ideal_model_checks)),
            "subtract_dc": np.array(int(subtract_dc)),
            "summary_plot_path": np.array("" if summary_plot_path is None else str(summary_plot_path)),
            "all_iv_plot_path": np.array("" if all_iv_plot_path is None else str(all_iv_plot_path)),
        }
        if subtract_dc:
            xline_npz_data["isweep_dc_offsets_A"] = isweep_dc_offsets
        if plot_png_bytes is not None:
            xline_npz_data["summary_plot_png"] = np.frombuffer(plot_png_bytes, dtype=np.uint8)
        if all_iv_plot_png_bytes is not None:
            xline_npz_data["all_iv_curves_plot_png"] = np.frombuffer(all_iv_plot_png_bytes, dtype=np.uint8)

        np.savez_compressed(output_npz_path, **xline_npz_data)

        print(f"Saved Langmuir x-line results to: {output_h5_path}")
        print(f"Saved Langmuir x-line NPZ results to: {output_npz_path}")

    # %%


def run_xy_analysis():
    """Run the configured xy-plane acquisition and analysis pipeline."""
    # %%
    # Read data

    print(f"Expected Langmuir XY-plane shots: {n_expected_shots}")
    print(f"shotnum_start = {shotnum_start}, shotnum_end = {shotnum_end}")

    file = lapd.File(filename, silent=True)

    print("Reading voltage data..")
    vsweep_full, dt = read_channel_xy(
        file_obj=file,
        board=board,
        channel=vsweep_channel,
        shotnum_start=shotnum_start,
        shotnum_end=shotnum_end,
        ny=ny,
        nx=nx,
        nshots=nshots,
        nt_full=nt_full,
        digitizer=digitizer,
        adc=adc,
        config_name=sis_config_name,
        scale_factor=vsweep_attenuation,
        flipup=True,
        negate=False,
    )
    print(f"Voltage data reshaped to (y, x, shots, time): {vsweep_full.shape}")

    print("Reading current data..")
    isweep_full, _ = read_channel_xy(
        file_obj=file,
        board=board,
        channel=isweep_channel,
        shotnum_start=shotnum_start,
        shotnum_end=shotnum_end,
        ny=ny,
        nx=nx,
        nshots=nshots,
        nt_full=nt_full,
        digitizer=digitizer,
        adc=adc,
        config_name=sis_config_name,
        scale_factor=isweep_attenuation / isweep_resistance,
        flipup=True,
        negate=negate_Isweep_current,
    )
    print(f"Current data reshaped to (y, x, shots, time): {isweep_full.shape}")

    print("Extracting sweep regions")
    vsweep = vsweep_full[..., sweep_start_index:sweep_end_index + 1]
    isweep = isweep_full[..., sweep_start_index:sweep_end_index + 1]

    if subtract_dc:
        if isweep_dc_offset_start_index is None or isweep_dc_offset_end_index is None:
            raise ValueError(
                "subtract_dc requires an independently measured plasma-off interval"
            )
        if not (
            0
            <= isweep_dc_offset_start_index
            <= isweep_dc_offset_end_index
            < nt_full
        ):
            raise ValueError("the electronics-offset interval is outside the trace")
        if not (
            isweep_dc_offset_end_index < sweep_start_index
            or isweep_dc_offset_start_index > sweep_end_index
        ):
            raise ValueError("the electronics-offset interval must not overlap the I-V sweep")
        print("Subtracting current DC offsets")
        isweep_dc_offset_portion = isweep_full[..., isweep_dc_offset_start_index:isweep_dc_offset_end_index + 1]
        isweep_dc_offsets = np.mean(isweep_dc_offset_portion, axis=-1)
        isweep = isweep - isweep_dc_offsets[..., np.newaxis]

    file.close()
    print("Finished reading and reshaping data.")

    # %%
    # Optional time-domain products for visualization/export. The analysis itself
    # uses unsmoothed samples and performs averaging in voltage bins.

    time = np.arange(nt) * dt

    vsweep_shot = vsweep * u.V
    isweep_shot = isweep * u.A
    vsweep_mean = np.mean(vsweep_shot, axis=2)
    isweep_mean = np.mean(isweep_shot, axis=2)

    isat = np.mean(isweep[..., isat_start_index:isat_end_index + 1], axis=-1)
    isat_mean_values, isat_std_values, isat_count = shot_mean_and_std(isat)
    isat_mean = isat_mean_values * u.A
    isat_std = isat_std_values * u.A

    print("Applying Savitzky-Golay filter along original sweep samples")
    vsweep_shot_smoothed = savgol_filter(
        vsweep_shot.value,
        sg_smooth_bins,
        sg_smooth_order,
        axis=-1
    ) * vsweep_shot.unit

    isweep_shot_smoothed = savgol_filter(
        isweep_shot.value,
        sg_smooth_bins,
        sg_smooth_order,
        axis=-1
    ) * isweep_shot.unit
    vsweep_mean_smoothed = np.mean(vsweep_shot_smoothed, axis=2)
    isweep_mean_smoothed = np.mean(isweep_shot_smoothed, axis=2)
    print("Finished smoothing data.")

    # %%
    # Prepare output arrays

    spatial_shape = (ny, nx)
    trace_shape = analysis_trace_shape(spatial_shape, nshots, shot_analysis_mode)
    n_traces = int(np.prod(trace_shape))

    te_shot = np.full(trace_shape, np.nan)
    vp_shot = np.full(trace_shape, np.nan)
    vf_shot = np.full(trace_shape, np.nan)
    ies_shot = np.full(trace_shape, np.nan)
    iis_shot = np.full(trace_shape, np.nan)
    n_e_shot = np.full(trace_shape, np.nan)
    analysis_ok_shot = np.zeros(trace_shape, dtype=np.uint8)
    analysis_rejection_counts = Counter()

    te_fit_r2_shot = np.full(trace_shape, np.nan)
    te_fit_rmse_shot = np.full(trace_shape, np.nan)
    te_fit_npts_shot = np.full(trace_shape, np.nan)
    te_fit_vstart_shot = np.full(trace_shape, np.nan)
    te_fit_vstop_shot = np.full(trace_shape, np.nan)
    te_fit_slope_shot = np.full(trace_shape, np.nan)
    te_fit_intercept_shot = np.full(trace_shape, np.nan)
    te_fit_i0_shot = np.full(trace_shape, np.nan)
    te_fit_passed_r2_shot = np.zeros(trace_shape, dtype=np.uint8)
    te_fit_candidate_count_shot = np.zeros(trace_shape, dtype=int)

    iv_voltage_grid_shot = np.full(trace_shape + (iv_npts,), np.nan)
    iv_current_grid_shot = np.full(trace_shape + (iv_npts,), np.nan)
    iv_didv_grid_shot = np.full(trace_shape + (iv_npts,), np.nan)
    iv_fit_mask_grid_shot = np.zeros(trace_shape + (iv_npts,), dtype=np.uint8)

    # %%
    # Robust Langmuir analysis on monotonic I(V) curves

    analysis_tasks = []
    for flat_index in range(n_traces):
        idx = np.unravel_index(flat_index, trace_shape)
        bias_values, current_values = prepare_trace_for_analysis(
            vsweep_shot.value,
            isweep_shot.value,
            idx,
            shot_analysis_mode,
            voltage_bin_width,
        )
        include_diagnostic_data = (
            diagnostic_plot_every is not None
            and diagnostic_plot_every > 0
            and flat_index % diagnostic_plot_every == 0
        )
        analysis_tasks.append(
            (
                flat_index,
                idx,
                bias_values,
                current_values,
                include_diagnostic_data,
                langmuir_analysis_config,
            )
        )

    n_analysis_processes = choose_analysis_process_count(n_traces, analysis_processes)
    use_parallel_analysis = parallel_analysis and n_analysis_processes > 1

    if use_parallel_analysis and "fork" not in mp.get_all_start_methods():
        print("Multiprocessing start method 'fork' is unavailable; falling back to serial analysis.")
        use_parallel_analysis = False

    analysis_processes_used = n_analysis_processes if use_parallel_analysis else 1

    if use_parallel_analysis:
        print(f"Processing {n_traces} Langmuir traces using {analysis_processes_used} worker processes...")
        process_context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=analysis_processes_used,
            mp_context=process_context,
        ) as executor:
            futures = [executor.submit(analyze_trace_worker, task) for task in analysis_tasks]
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                idx = result["idx"]
                iy, ix, ishot = idx
                print(f"Finished trace {result['flat_index'] + 1}/{n_traces} at index {idx} ({completed}/{n_traces} complete)")

                for warning in result["warnings"]:
                    print(f"{warning} at index {idx}")

                if not result["ok"]:
                    reason = (
                        result["warnings"][-1]
                        if result["warnings"]
                        else "unspecified analysis failure"
                    )
                    analysis_rejection_counts[reason] += 1

                if result["iv_voltage_grid"] is None:
                    continue

                analysis_ok_shot[idx] = int(result["ok"])
                # Keep every finite estimate for plotting and export.  The strict
                # quality decision remains available separately in analysis_ok_shot.
                te_shot[idx] = result["te_eV"]
                vp_shot[idx] = result["vp_V"]
                vf_shot[idx] = result["vf_V"]
                ies_shot[idx] = result["ies_A"]
                iis_shot[idx] = result["iis_A"]
                n_e_shot[idx] = result.get("n_e_m3", np.nan)

                te_fit_r2_shot[idx] = result["te_fit_r2"]
                te_fit_rmse_shot[idx] = result["te_fit_rmse"]
                if result["te_fit_candidate_count"]:
                    te_fit_npts_shot[idx] = result["te_fit_npts"]
                te_fit_vstart_shot[idx] = result["te_fit_vstart"]
                te_fit_vstop_shot[idx] = result["te_fit_vstop"]
                te_fit_slope_shot[idx] = result["te_fit_slope"]
                te_fit_intercept_shot[idx] = result["te_fit_intercept"]
                te_fit_i0_shot[idx] = result["te_fit_i0"]
                te_fit_passed_r2_shot[idx] = int(result["te_fit_passed_r2"])
                te_fit_candidate_count_shot[idx] = result["te_fit_candidate_count"]

                iv_voltage_grid_shot[idx + (slice(None),)] = result["iv_voltage_grid"]
                iv_current_grid_shot[idx + (slice(None),)] = result["iv_current_grid"]
                if result["iv_didv_grid"] is not None:
                    iv_didv_grid_shot[idx + (slice(None),)] = result["iv_didv_grid"]
                iv_fit_mask_grid_shot[idx + (slice(None),)] = result["iv_fit_mask"]

                diagnostic_data = result["diagnostic_data"]
                if diagnostic_data is not None:
                    fit_label = (
                        "averaged shots"
                        if shot_analysis_mode == "average"
                        else f"shot {ishot}"
                    )
                    file_label = (
                        "shots_averaged"
                        if shot_analysis_mode == "average"
                        else f"shot{ishot:03d}"
                    )
                    fig = render_analysis_iv_diagnostic_plot(
                        result,
                        f"Diagnostic IV analysis at (y, x) index ({iy}, {ix}), "
                        f"{fit_label} (x = {x[ix]:.2f} cm, y = {y[iy]:.2f} cm)",
                    )
                    save_diagnostic_figure(
                        fig,
                        diagnostic_plot_output_dir,
                        f"{Path(filename).stem}_iv_diagnostic_iy{iy:03d}_ix{ix:03d}_"
                        f"{file_label}_y_{y[iy]:+.2f}cm_x_{x[ix]:+.2f}cm",
                    )
                    plt.show(block=False)
                    plt.pause(example_pause_seconds)
                    plt.close(fig)
    else:
        print("Processing Langmuir traces serially...")
        for task in analysis_tasks:
            flat_index = task[0]
            idx = task[1]
            iy, ix, ishot = idx
            print(f"Processing trace {flat_index + 1}/{n_traces} at index {idx}...")
            result = analyze_trace_worker(task)

            for warning in result["warnings"]:
                print(f"{warning} at index {idx}")

            if not result["ok"]:
                reason = (
                    result["warnings"][-1]
                    if result["warnings"]
                    else "unspecified analysis failure"
                )
                analysis_rejection_counts[reason] += 1

            if result["iv_voltage_grid"] is None:
                continue

            analysis_ok_shot[idx] = int(result["ok"])
            # Keep every finite estimate for plotting and export.  The strict
            # quality decision remains available separately in analysis_ok_shot.
            te_shot[idx] = result["te_eV"]
            vp_shot[idx] = result["vp_V"]
            vf_shot[idx] = result["vf_V"]
            ies_shot[idx] = result["ies_A"]
            iis_shot[idx] = result["iis_A"]
            n_e_shot[idx] = result.get("n_e_m3", np.nan)

            te_fit_r2_shot[idx] = result["te_fit_r2"]
            te_fit_rmse_shot[idx] = result["te_fit_rmse"]
            if result["te_fit_candidate_count"]:
                te_fit_npts_shot[idx] = result["te_fit_npts"]
            te_fit_vstart_shot[idx] = result["te_fit_vstart"]
            te_fit_vstop_shot[idx] = result["te_fit_vstop"]
            te_fit_slope_shot[idx] = result["te_fit_slope"]
            te_fit_intercept_shot[idx] = result["te_fit_intercept"]
            te_fit_i0_shot[idx] = result["te_fit_i0"]
            te_fit_passed_r2_shot[idx] = int(result["te_fit_passed_r2"])
            te_fit_candidate_count_shot[idx] = result["te_fit_candidate_count"]

            iv_voltage_grid_shot[idx + (slice(None),)] = result["iv_voltage_grid"]
            iv_current_grid_shot[idx + (slice(None),)] = result["iv_current_grid"]
            if result["iv_didv_grid"] is not None:
                iv_didv_grid_shot[idx + (slice(None),)] = result["iv_didv_grid"]
            iv_fit_mask_grid_shot[idx + (slice(None),)] = result["iv_fit_mask"]

            diagnostic_data = result["diagnostic_data"]
            if diagnostic_data is not None:
                fit_label = (
                    "averaged shots"
                    if shot_analysis_mode == "average"
                    else f"shot {ishot}"
                )
                file_label = (
                    "shots_averaged"
                    if shot_analysis_mode == "average"
                    else f"shot{ishot:03d}"
                )
                fig = render_analysis_iv_diagnostic_plot(
                    result,
                    f"Diagnostic IV analysis at (y, x) index ({iy}, {ix}), "
                    f"{fit_label} (x = {x[ix]:.2f} cm, y = {y[iy]:.2f} cm)",
                )
                save_diagnostic_figure(
                    fig,
                    diagnostic_plot_output_dir,
                    f"{Path(filename).stem}_iv_diagnostic_iy{iy:03d}_ix{ix:03d}_"
                    f"{file_label}_y_{y[iy]:+.2f}cm_x_{x[ix]:+.2f}cm",
                )
                plt.show(block=False)
                plt.pause(example_pause_seconds)
                plt.close(fig)
    # Compatibility export: this mask identifies locations with a reported Te fit.
    # Consult analysis_ok_shot to distinguish quality-accepted and rejected fits.
    statistics_fit_mask = np.isfinite(te_shot)
    te_values, te_std_values, fit_count = shot_mean_and_std(te_shot)
    vp_values, vp_std_values, _ = shot_mean_and_std(vp_shot)
    vf_values, vf_std_values, _ = shot_mean_and_std(vf_shot)
    ies_values, ies_std_values, _ = shot_mean_and_std(ies_shot)
    iis_values, iis_std_values, _ = shot_mean_and_std(iis_shot)
    n_e_values, n_e_std_values, _ = shot_mean_and_std(n_e_shot)

    te = te_values * u.eV
    vp = vp_values * u.V
    vf = vf_values * u.V
    ies = ies_values * u.A
    iis = iis_values * u.A
    n_e = n_e_values * u.m**-3
    te_std = te_std_values * u.eV
    vp_std = vp_std_values * u.V
    vf_std = vf_std_values * u.V
    ies_std = ies_std_values * u.A
    iis_std = iis_std_values * u.A
    n_e_std = n_e_std_values * u.m**-3

    te_fit_r2, te_fit_r2_std, te_fit_count = shot_mean_and_std(te_fit_r2_shot)
    te_fit_rmse, te_fit_rmse_std, _ = shot_mean_and_std(te_fit_rmse_shot)
    te_fit_npts, te_fit_npts_std, _ = shot_mean_and_std(te_fit_npts_shot)
    te_fit_vstart, te_fit_vstart_std, _ = shot_mean_and_std(te_fit_vstart_shot)
    te_fit_vstop, te_fit_vstop_std, _ = shot_mean_and_std(te_fit_vstop_shot)
    te_fit_slope, te_fit_slope_std, _ = shot_mean_and_std(te_fit_slope_shot)
    te_fit_intercept, te_fit_intercept_std, _ = shot_mean_and_std(te_fit_intercept_shot)
    te_fit_i0, te_fit_i0_std, _ = shot_mean_and_std(te_fit_i0_shot)
    te_fit_passed_r2 = np.any(te_fit_passed_r2_shot.astype(bool), axis=-1).astype(np.uint8)
    te_fit_candidate_count = np.sum(te_fit_candidate_count_shot, axis=-1)
    analysis_valid_count = np.sum(analysis_ok_shot, axis=-1)
    analysis_ok = (analysis_valid_count > 0).astype(np.uint8)
    accepted_trace_count = int(np.sum(analysis_ok_shot))
    reported_trace_count = int(np.count_nonzero(np.isfinite(te_shot)))
    fit_description = (
        "shot-averaged fits"
        if shot_analysis_mode == "average"
        else "per-shot fits"
    )
    analysis_status = (
        f"Reported {reported_trace_count}/{n_traces} finite {fit_description}; "
        f"accepted {accepted_trace_count}/{n_traces} {fit_description}"
    )
    if analysis_rejection_counts:
        common_reason, common_count = analysis_rejection_counts.most_common(1)[0]
        analysis_status += f"; top rejection ({common_count}): {common_reason}"
    print(f"Analysis summary: {analysis_status}")

    # Representative I-V products are visualization-only shot means.  They are
    # not used for physics, because individual sweep endpoints need not coincide.
    # Full per-shot grids are exported separately below.
    iv_voltage_grid = np.nanmean(iv_voltage_grid_shot, axis=-2)
    iv_current_grid = np.nanmean(iv_current_grid_shot, axis=-2)
    iv_didv_grid = np.nanmean(iv_didv_grid_shot, axis=-2)
    iv_fit_mask_grid = np.any(iv_fit_mask_grid_shot.astype(bool), axis=-2).astype(np.uint8)

    shot_statistics_available, shot_statistics_stage = shot_statistics_metadata(
        shot_analysis_mode,
        nshots,
    )
    if shot_analysis_mode == "average":
        isat_std = np.full(spatial_shape, np.nan) * u.A
        unavailable = unavailable_per_shot_fit_products(
            spatial_shape,
            nshots,
            iv_npts,
        )
        te_shot = unavailable["te_shot"]
        vp_shot = unavailable["vp_shot"]
        vf_shot = unavailable["vf_shot"]
        ies_shot = unavailable["ies_shot"]
        iis_shot = unavailable["iis_shot"]
        n_e_shot = unavailable["n_e_shot"]
        analysis_ok_shot = unavailable["analysis_ok_shot"]
        statistics_fit_mask = unavailable["statistics_fit_mask"]
        te_fit_r2_shot = unavailable["te_fit_r2_shot"]
        te_fit_rmse_shot = unavailable["te_fit_rmse_shot"]
        te_fit_npts_shot = unavailable["te_fit_npts_shot"]
        te_fit_vstart_shot = unavailable["te_fit_vstart_shot"]
        te_fit_vstop_shot = unavailable["te_fit_vstop_shot"]
        te_fit_slope_shot = unavailable["te_fit_slope_shot"]
        te_fit_intercept_shot = unavailable["te_fit_intercept_shot"]
        te_fit_i0_shot = unavailable["te_fit_i0_shot"]
        te_fit_passed_r2_shot = unavailable["te_fit_passed_r2_shot"]
        te_fit_candidate_count_shot = unavailable["te_fit_candidate_count_shot"]
        iv_voltage_grid_shot = unavailable["iv_voltage_grid_shot"]
        iv_current_grid_shot = unavailable["iv_current_grid_shot"]
        iv_didv_grid_shot = unavailable["iv_didv_grid_shot"]
        iv_fit_mask_grid_shot = unavailable["iv_fit_mask_grid_shot"]

    print(
        "Finished "
        + (
            "shot-averaged Langmuir probe analysis"
            if shot_analysis_mode == "average"
            else "per-shot Langmuir probe analysis and spatial statistics"
        )
    )

    # %%
    # Optional post-processing on XY maps

    post = apply_optional_xy_postprocessing(
        te=te.value,
        vp=vp.value,
        vf=vf.value,
        ies=ies.value,
        iis=iis.value,
        n_e=n_e.value,
        enable_vp_spike_rejection=enable_vp_spike_rejection,
        vp_spike_half_window=vp_spike_half_window,
        vp_spike_threshold_V=vp_spike_threshold_V,
        vp_replace_flagged_with_local_median=vp_replace_flagged_with_local_median,
        enable_neighbor_smoothing=enable_neighbor_smoothing,
        neighbor_smooth_half_window=neighbor_smooth_half_window,
        neighbor_smooth_sigma=neighbor_smooth_sigma,
    )

    te_raw = post["te_raw"] * u.eV
    vp_raw = post["vp_raw"] * u.V
    vf_raw = post["vf_raw"] * u.V
    ies_raw = post["ies_raw"] * u.A
    iis_raw = post["iis_raw"] * u.A
    n_e_raw = post["n_e_raw"] * u.m**-3 if "n_e_raw" in post else None

    te_processed = post["te_processed"] * u.eV
    vp_processed = post["vp_processed"] * u.V
    vf_processed = post["vf_processed"] * u.V
    ies_processed = post["ies_processed"] * u.A
    iis_processed = post["iis_processed"] * u.A
    n_e_processed = post["n_e_processed"] * u.m**-3 if "n_e_processed" in post else None

    vp_spike_mask = post["vp_spike_mask"]

    te_plot = te_processed if enable_neighbor_smoothing else te_raw
    vp_plot = vp_processed if (enable_neighbor_smoothing or enable_vp_spike_rejection) else vp_raw
    vf_plot = vf_processed if enable_neighbor_smoothing else vf_raw
    ies_plot = ies_processed if enable_neighbor_smoothing else ies_raw
    iis_plot = iis_processed if enable_neighbor_smoothing else iis_raw
    n_e_plot = n_e_processed if enable_neighbor_smoothing else n_e_raw

    # Calibrate the xy density map using the same normalized-profile spatial
    # integral as the x-line pipeline.  One measured x-line determines a
    # single scale factor, which is applied uniformly to the full map.
    shape_factor = None
    density_scale = np.nan
    electron_density_normalized_xline = None
    electron_density_scaled_xline = None
    (
        interferometer_profile_y_index,
        interferometer_profile_y_selected_cm,
        electron_density_xline,
    ) = select_xline_from_xy_map(
        y,
        n_e_plot,
        requested_y_cm=interferometer_profile_y_cm,
    )
    if calculate_shape_factor:
        interferometer_result = scale_xy_density_map_to_interferometer(
            x,
            y,
            n_e_plot,
            interferometer_scaling,
            requested_y_cm=interferometer_profile_y_cm,
        )
        shape_factor = interferometer_result["shape_factor"]
        density_scale = interferometer_result["density_scale"]
        electron_density_normalized_xline = interferometer_result[
            "normalized_xline"
        ]
        electron_density_scaled_xline = interferometer_result["scaled_xline"]
        n_e_plot = interferometer_result["scaled_density_map"]

        if np.isfinite(density_scale):
            n_e_std = n_e_std * density_scale
            print(
                "Scaled the xy electron-density map using the x-line at "
                f"y={interferometer_profile_y_selected_cm:g} cm; "
                f"peak n_e = {np.nanmax(n_e_plot):.4g}."
            )
        else:
            print(
                "Warning: the xy electron-density map could not be normalized "
                "and scaled because the selected x-line maximum, shape factor, "
                "or interferometer scaling is not finite and positive."
            )
            n_e_std = np.full(n_e_std.shape, np.nan) * u.m**-3

    # %%
    # Plot results

    plot_png_bytes = None
    all_iv_plot_png_bytes = None
    summary_plot_path = None
    all_iv_plot_path = None

    if plot_results:
        fig = render_xy_summary_plot(
            X,
            Y,
            te_plot,
            vp_plot,
            vf_plot,
            ies_plot,
            iis_plot,
            n_e_plot,
            vp_spike_mask=vp_spike_mask,
            vp_raw=vp_raw,
            te_fit_r2=te_fit_r2,
            te_poor_fit_r2=te_min_r2,
            analysis_ok=analysis_ok,
            analysis_status=analysis_status,
            shape_factor=shape_factor,
            interferometer_profile_y_cm=interferometer_profile_y_selected_cm,
            title=f"{Path(filename).stem} - Langmuir XY-plane Summary",
        )

        plot_buffer = io.BytesIO()
        fig.savefig(plot_buffer, format="png", dpi=600)
        plot_png_bytes = plot_buffer.getvalue()
        plot_buffer.close()

        summary_plot_path = save_diagnostic_figure(
            fig,
            diagnostic_plot_output_dir,
            f"{Path(filename).stem}_langmuir_xy_summary",
        )
        plt.show()

    if make_all_iv_diagnostic_plot:
        fig_iv = render_xy_all_iv_curves_plot(x, y, iv_voltage_grid, iv_current_grid)

        iv_plot_buffer = io.BytesIO()
        fig_iv.savefig(iv_plot_buffer, format="png", dpi=600)
        all_iv_plot_png_bytes = iv_plot_buffer.getvalue()
        iv_plot_buffer.close()

        all_iv_plot_path = save_diagnostic_figure(
            fig_iv,
            diagnostic_plot_output_dir,
            f"{Path(filename).stem}_all_interpolated_iv_curves",
        )
        plt.show()

    # %%
    # Save processed Langmuir XY-plane results

    output_h5_path = Path(filename)
    output_npz_path = output_h5_path.with_name(f"{output_h5_path.stem}_langmuir_xy.npz")

    if save_results:
        with open_h5_for_update(output_h5_path) as h5f:
            if "langmuir_xy" in h5f:
                del h5f["langmuir_xy"]

            grp = h5f.create_group("langmuir_xy")

            grp.create_dataset("x_cm", data=x)
            grp.create_dataset("y_cm", data=y)
            grp.create_dataset("X_cm", data=X)
            grp.create_dataset("Y_cm", data=Y)
            grp.create_dataset("time_s", data=time)

            # Primary exported profiles reflect current optional processing choices
            grp.create_dataset("te_eV", data=te_plot.value)
            grp.create_dataset("vp_V", data=vp_plot.value)
            grp.create_dataset("vf_V", data=vf_plot.value)
            grp.create_dataset("ies_A", data=ies_plot.value)
            grp.create_dataset("iis_A", data=iis_plot.value)
            grp.create_dataset("isat_A", data=isat_mean.value)
            grp.create_dataset("n_e_m3", data=n_e_plot.value)
            grp.create_dataset(
                "interferometer_xline_n_e_raw_m3",
                data=u.Quantity(electron_density_xline).to_value(u.m**-3),
            )
            if electron_density_normalized_xline is not None:
                grp.create_dataset(
                    "interferometer_xline_n_e_normalized",
                    data=electron_density_normalized_xline,
                )
            if electron_density_scaled_xline is not None:
                grp.create_dataset(
                    "interferometer_xline_n_e_m3",
                    data=electron_density_scaled_xline.to_value(u.m**-3),
                )

            # Per-shot fitted quantities and their per-location sample deviations
            grp.create_dataset("te_shot_eV", data=te_shot)
            grp.create_dataset("vp_shot_V", data=vp_shot)
            grp.create_dataset("vf_shot_V", data=vf_shot)
            grp.create_dataset("ies_shot_A", data=ies_shot)
            grp.create_dataset("iis_shot_A", data=iis_shot)
            grp.create_dataset("n_e_shot_m3", data=n_e_shot)
            grp.create_dataset("isat_shot_A", data=isat)
            grp.create_dataset("te_std_eV", data=te_std.value)
            grp.create_dataset("vp_std_V", data=vp_std.value)
            grp.create_dataset("vf_std_V", data=vf_std.value)
            grp.create_dataset("ies_std_A", data=ies_std.value)
            grp.create_dataset("iis_std_A", data=iis_std.value)
            grp.create_dataset("n_e_std_m3", data=n_e_std.value)
            grp.create_dataset("isat_std_A", data=isat_std.value)
            grp.create_dataset("fit_count", data=fit_count)
            grp.create_dataset("analysis_ok", data=analysis_ok)
            grp.create_dataset("analysis_valid_count", data=analysis_valid_count)
            grp.create_dataset("analysis_ok_shot", data=analysis_ok_shot)
            grp.create_dataset("statistics_fit_mask", data=statistics_fit_mask.astype(np.uint8))

            # Raw extracted profiles
            grp.create_dataset("te_raw_eV", data=te_raw.value)
            grp.create_dataset("vp_raw_V", data=vp_raw.value)
            grp.create_dataset("vf_raw_V", data=vf_raw.value)
            grp.create_dataset("ies_raw_A", data=ies_raw.value)
            grp.create_dataset("iis_raw_A", data=iis_raw.value)
            grp.create_dataset("n_e_raw_m3", data=n_e_raw.value if n_e_raw is not None else np.full(spatial_shape, np.nan))

            # Post-processed profiles
            grp.create_dataset("te_processed_eV", data=te_processed.value)
            grp.create_dataset("vp_processed_V", data=vp_processed.value)
            grp.create_dataset("vf_processed_V", data=vf_processed.value)
            grp.create_dataset("ies_processed_A", data=ies_processed.value)
            grp.create_dataset("iis_processed_A", data=iis_processed.value)
            grp.create_dataset("n_e_processed_m3", data=n_e_processed.value if n_e_processed is not None else np.full(spatial_shape, np.nan))

            # Spike diagnostics
            grp.create_dataset("vp_spike_mask", data=vp_spike_mask.astype(np.uint8))

            # Shot-averaged original sweep curves
            grp.create_dataset("vsweep_mean_V", data=vsweep_mean.value)
            grp.create_dataset("isweep_mean_A", data=isweep_mean.value)
            grp.create_dataset("vsweep_mean_smoothed_V", data=vsweep_mean_smoothed.value)
            grp.create_dataset("isweep_mean_smoothed_A", data=isweep_mean_smoothed.value)
            grp.create_dataset("vsweep_shot_smoothed_V", data=vsweep_shot_smoothed.value)
            grp.create_dataset("isweep_shot_smoothed_A", data=isweep_shot_smoothed.value)

            # Interpolated monotonic IV analysis products
            grp.create_dataset("iv_voltage_grid_V", data=iv_voltage_grid)
            grp.create_dataset("iv_current_grid_A", data=iv_current_grid)
            grp.create_dataset("iv_didv_grid_A_per_V", data=iv_didv_grid)
            grp.create_dataset("iv_te_fit_mask", data=iv_fit_mask_grid)
            grp.create_dataset("iv_voltage_grid_shot_V", data=iv_voltage_grid_shot)
            grp.create_dataset("iv_current_grid_shot_A", data=iv_current_grid_shot)
            grp.create_dataset("iv_didv_grid_shot_A_per_V", data=iv_didv_grid_shot)
            grp.create_dataset("iv_te_fit_mask_shot", data=iv_fit_mask_grid_shot)

            # Fit diagnostics
            grp.create_dataset("te_fit_r2", data=te_fit_r2)
            grp.create_dataset("te_fit_rmse_logI", data=te_fit_rmse)
            grp.create_dataset("te_fit_npts", data=te_fit_npts)
            grp.create_dataset("te_fit_vstart_V", data=te_fit_vstart)
            grp.create_dataset("te_fit_vstop_V", data=te_fit_vstop)
            grp.create_dataset("te_fit_slope_logI_per_V", data=te_fit_slope)
            grp.create_dataset("te_fit_intercept_logI", data=te_fit_intercept)
            grp.create_dataset("te_fit_i0_A", data=te_fit_i0)
            grp.create_dataset("te_fit_passed_r2", data=te_fit_passed_r2)
            grp.create_dataset("te_fit_candidate_count", data=te_fit_candidate_count)
            grp.create_dataset("te_fit_r2_std", data=te_fit_r2_std)
            grp.create_dataset("te_fit_rmse_logI_std", data=te_fit_rmse_std)
            grp.create_dataset("te_fit_npts_std", data=te_fit_npts_std)
            grp.create_dataset("te_fit_vstart_std_V", data=te_fit_vstart_std)
            grp.create_dataset("te_fit_vstop_std_V", data=te_fit_vstop_std)
            grp.create_dataset("te_fit_slope_std_logI_per_V", data=te_fit_slope_std)
            grp.create_dataset("te_fit_intercept_std_logI", data=te_fit_intercept_std)
            grp.create_dataset("te_fit_i0_std_A", data=te_fit_i0_std)
            grp.create_dataset("te_fit_count", data=te_fit_count)
            grp.create_dataset("te_fit_r2_shot", data=te_fit_r2_shot)
            grp.create_dataset("te_fit_rmse_logI_shot", data=te_fit_rmse_shot)
            grp.create_dataset("te_fit_npts_shot", data=te_fit_npts_shot)
            grp.create_dataset("te_fit_vstart_shot_V", data=te_fit_vstart_shot)
            grp.create_dataset("te_fit_vstop_shot_V", data=te_fit_vstop_shot)
            grp.create_dataset("te_fit_slope_shot_logI_per_V", data=te_fit_slope_shot)
            grp.create_dataset("te_fit_intercept_shot_logI", data=te_fit_intercept_shot)
            grp.create_dataset("te_fit_i0_shot_A", data=te_fit_i0_shot)
            grp.create_dataset("te_fit_passed_r2_shot", data=te_fit_passed_r2_shot)
            grp.create_dataset("te_fit_candidate_count_shot", data=te_fit_candidate_count_shot)

            if subtract_dc:
                grp.create_dataset("isweep_dc_offsets_A", data=isweep_dc_offsets)
            grp.create_dataset("xy_shape_info", data=np.array([ny, nx, nshots, nt_full, nt, iv_npts], dtype=np.int64))
            grp.create_dataset("trace_spatial_shape", data=np.array(spatial_shape, dtype=np.int64))

            if plot_results and plot_png_bytes is None:
                fig = render_xy_summary_plot(
                    X,
                    Y,
                    te_plot,
                    vp_plot,
                    vf_plot,
                    ies_plot,
                    iis_plot,
                    n_e_plot,
                    vp_spike_mask=vp_spike_mask,
                    vp_raw=vp_raw,
                    te_fit_r2=te_fit_r2,
                    te_poor_fit_r2=te_min_r2,
                    analysis_ok=analysis_ok,
                    analysis_status=analysis_status,
                    shape_factor=shape_factor,
                    interferometer_profile_y_cm=(
                        interferometer_profile_y_selected_cm
                    ),
                )
                plot_buffer = io.BytesIO()
                fig.savefig(plot_buffer, format="png", dpi=600)
                plot_png_bytes = plot_buffer.getvalue()
                plot_buffer.close()
                plt.close(fig)

            if plot_png_bytes is not None:
                grp.create_dataset(
                    "summary_plot_png",
                    data=np.frombuffer(plot_png_bytes, dtype=np.uint8)
                )
                grp["summary_plot_png"].attrs["mime_type"] = "image/png"
                grp["summary_plot_png"].attrs["description"] = "Rendered Langmuir XY-plane summary plot."

            if all_iv_plot_png_bytes is not None:
                grp.create_dataset(
                    "all_iv_curves_plot_png",
                    data=np.frombuffer(all_iv_plot_png_bytes, dtype=np.uint8)
                )
                grp["all_iv_curves_plot_png"].attrs["mime_type"] = "image/png"
                grp["all_iv_curves_plot_png"].attrs["description"] = (
                    "Diagnostic plot of representative interpolated I-V curves colored by y position."
                )

            grp.attrs["source_file"] = str(filename)
            grp.attrs["geometry"] = "xy_plane"

            grp.attrs["ny"] = ny
            grp.attrs["nx"] = nx
            grp.attrs["n_traces"] = ny * nx * nshots
            grp.attrs["n_analysis_traces"] = n_traces
            grp.attrs["n_acquired_traces"] = ny * nx * nshots
            grp.attrs["trace_spatial_ndim"] = len(spatial_shape)
            grp.attrs["nshots"] = nshots
            grp.attrs["shot_analysis_mode"] = shot_analysis_mode
            grp.attrs["nt_full"] = nt_full
            grp.attrs["nt_sweep"] = nt
            grp.attrs["dt_s"] = dt

            grp.attrs["digitizer"] = digitizer
            grp.attrs["adc"] = adc
            grp.attrs["config_name"] = sis_config_name
            grp.attrs["board"] = board
            grp.attrs["vsweep_channel"] = vsweep_channel
            grp.attrs["isweep_channel"] = isweep_channel
            grp.attrs["vsweep_attenuation"] = vsweep_attenuation
            grp.attrs["isweep_attenuation"] = isweep_attenuation
            grp.attrs["isweep_resistance_ohm"] = isweep_resistance
            grp.attrs["calculate_shape_factor"] = int(calculate_shape_factor)
            grp.attrs["shape_factor_m"] = (
                np.nan if shape_factor is None else shape_factor.to_value(u.m)
            )
            grp.attrs["density_scale"] = density_scale
            grp.attrs["interferometer_line_integrated_density_m2"] = (
                interferometer_scaling.to_value(u.m**-2)
            )
            grp.attrs["interferometer_profile_y_requested_cm"] = (
                interferometer_profile_y_cm
            )
            grp.attrs["interferometer_profile_y_index"] = (
                interferometer_profile_y_index
            )
            grp.attrs["interferometer_profile_y_selected_cm"] = (
                interferometer_profile_y_selected_cm
            )

            grp.attrs["sweep_start_index"] = sweep_start_index
            grp.attrs["sweep_end_index"] = sweep_end_index
            grp.attrs["isat_start_index"] = isat_start_index
            grp.attrs["isat_end_index"] = isat_end_index

            if subtract_dc:
                grp.attrs["dc_offset_start_index"] = isweep_dc_offset_start_index
                grp.attrs["dc_offset_end_index"] = isweep_dc_offset_end_index

            grp.attrs["sg_smooth_bins"] = sg_smooth_bins
            grp.attrs["sg_smooth_order"] = sg_smooth_order
            grp.attrs["iv_npts"] = iv_npts
            grp.attrs["voltage_bin_width_V"] = voltage_bin_width
            grp.attrs["vp_smoothing"] = "none" if vp_smoothing is None else str(vp_smoothing)
            grp.attrs["vp_smoothing_width_V"] = vp_smoothing_width_V
            grp.attrs["vp_savgol_order"] = vp_savgol_order
            grp.attrs["ies_method"] = ies_method
            grp.attrs["ies_definition"] = describe_ies_method(ies_method)
            grp.attrs["parallel_analysis"] = int(parallel_analysis)
            grp.attrs["analysis_processes_requested"] = (
                -1 if analysis_processes is None else int(analysis_processes)
            )
            grp.attrs["analysis_processes_used"] = analysis_processes_used
            grp.attrs["analysis_start_method"] = "fork" if use_parallel_analysis else "serial"

            grp.attrs["te_min_points"] = te_min_points
            grp.attrs["te_margin_from_vp_V"] = te_margin_from_vp
            grp.attrs["te_current_floor_frac"] = te_current_floor_frac
            grp.attrs["te_subtract_i0"] = int(te_subtract_i0)
            grp.attrs["te_min_eV"] = te_min_eV
            grp.attrs["te_max_eV"] = te_max_eV
            grp.attrs["te_min_r2"] = te_min_r2
            grp.attrs["ion_min_snr"] = ion_min_snr
            grp.attrs["analysis_model"] = (
                "single-temperature semilog fit; median low-bias ion estimate; "
                + describe_ies_method(ies_method)
            )
            grp.attrs["te_fit_i0_definition"] = "median low-bias ion-region current"
            grp.attrs["iv_npts_role"] = "fixed-size diagnostics only"
            grp.attrs["current_zero_calibrated"] = int(current_zero_calibrated)
            grp.attrs["negate_Isweep_current"] = int(negate_Isweep_current)
            grp.attrs["enforce_ideal_model_checks"] = int(enforce_ideal_model_checks)
            grp.attrs["subtract_dc"] = int(subtract_dc)
            grp.attrs["shot_standard_deviation_ddof"] = 1
            grp.attrs["shot_statistics_available"] = int(
                shot_statistics_available
            )
            grp.attrs["per_shot_axis_order"] = "y,x,shot"
            grp.attrs["shot_statistics_stage"] = shot_statistics_stage
            grp.attrs["profile_value_policy"] = (
                "finite estimates retained; analysis_ok records fit acceptance"
            )

            grp.attrs["enable_vp_spike_rejection"] = int(enable_vp_spike_rejection)
            grp.attrs["vp_spike_half_window_y"] = _as_yx_half_window(vp_spike_half_window)[0]
            grp.attrs["vp_spike_half_window_x"] = _as_yx_half_window(vp_spike_half_window)[1]
            grp.attrs["vp_spike_threshold_V"] = vp_spike_threshold_V
            grp.attrs["vp_replace_flagged_with_local_median"] = int(vp_replace_flagged_with_local_median)

            grp.attrs["enable_neighbor_smoothing"] = int(enable_neighbor_smoothing)
            grp.attrs["neighbor_smooth_half_window_y"] = _as_yx_half_window(neighbor_smooth_half_window)[0]
            grp.attrs["neighbor_smooth_half_window_x"] = _as_yx_half_window(neighbor_smooth_half_window)[1]
            grp.attrs["neighbor_smooth_sigma"] = neighbor_smooth_sigma

            grp.attrs["diagnostic_plot_every"] = diagnostic_plot_every
            grp.attrs["diagnostic_plot_output_dir"] = str(diagnostic_plot_output_dir)
            if summary_plot_path is not None:
                grp.attrs["summary_plot_path"] = str(summary_plot_path)
            if all_iv_plot_path is not None:
                grp.attrs["all_iv_plot_path"] = str(all_iv_plot_path)

            grp.attrs["make_all_iv_diagnostic_plot"] = int(make_all_iv_diagnostic_plot)
            grp.attrs["data_offset_shots"] = data_offset

        xy_npz_data = {
            "source_file": np.array(str(filename)),
            "geometry": np.array("xy_plane"),
            "shot_analysis_mode": np.array(shot_analysis_mode),
            "shot_statistics_available": np.array(
                int(shot_statistics_available)
            ),
            "n_analysis_traces": np.array(n_traces),
            "n_acquired_traces": np.array(ny * nx * nshots),
            "calculate_shape_factor": np.array(int(calculate_shape_factor)),
            "shape_factor_m": np.array(
                np.nan if shape_factor is None else shape_factor.to_value(u.m)
            ),
            "density_scale": np.array(density_scale),
            "interferometer_line_integrated_density_m2": np.array(
                interferometer_scaling.to_value(u.m**-2)
            ),
            "interferometer_profile_y_requested_cm": np.array(
                interferometer_profile_y_cm
            ),
            "interferometer_profile_y_index": np.array(
                interferometer_profile_y_index
            ),
            "interferometer_profile_y_selected_cm": np.array(
                interferometer_profile_y_selected_cm
            ),
            "interferometer_xline_n_e_raw_m3": u.Quantity(
                electron_density_xline
            ).to_value(u.m**-3),
            "interferometer_xline_n_e_normalized": (
                np.full(nx, np.nan)
                if electron_density_normalized_xline is None
                else electron_density_normalized_xline
            ),
            "interferometer_xline_n_e_m3": (
                np.full(nx, np.nan)
                if electron_density_scaled_xline is None
                else electron_density_scaled_xline.to_value(u.m**-3)
            ),
            "x_cm": x,
            "y_cm": y,
            "X_cm": X,
            "Y_cm": Y,
            "time_s": time,
            "te_eV": te_plot.value,
            "vp_V": vp_plot.value,
            "vf_V": vf_plot.value,
            "ies_A": ies_plot.value,
            "iis_A": iis_plot.value,
            "n_e_m3": n_e_plot.value,
            "isat_A": isat_mean.value,
            "te_shot_eV": te_shot,
            "vp_shot_V": vp_shot,
            "vf_shot_V": vf_shot,
            "ies_shot_A": ies_shot,
            "iis_shot_A": iis_shot,
            "n_e_shot_m3": n_e_shot,
            "isat_shot_A": isat,
            "te_std_eV": te_std.value,
            "vp_std_V": vp_std.value,
            "vf_std_V": vf_std.value,
            "ies_std_A": ies_std.value,
            "iis_std_A": iis_std.value,
            "n_e_std_m3": n_e_std.value,
            "isat_std_A": isat_std.value,
            "fit_count": fit_count,
            "analysis_ok": analysis_ok,
            "analysis_valid_count": analysis_valid_count,
            "analysis_ok_shot": analysis_ok_shot,
            "statistics_fit_mask": statistics_fit_mask.astype(np.uint8),
            "te_raw_eV": te_raw.value,
            "vp_raw_V": vp_raw.value,
            "vf_raw_V": vf_raw.value,
            "ies_raw_A": ies_raw.value,
            "iis_raw_A": iis_raw.value,
            "n_e_raw_m3": n_e_raw.value if n_e_raw is not None else np.full(spatial_shape, np.nan),
            "te_processed_eV": te_processed.value,
            "vp_processed_V": vp_processed.value,
            "vf_processed_V": vf_processed.value,
            "ies_processed_A": ies_processed.value,
            "iis_processed_A": iis_processed.value,
            "n_e_processed_m3": n_e_processed.value if n_e_processed is not None else np.full(spatial_shape, np.nan),
            "vp_spike_mask": vp_spike_mask.astype(np.uint8),
            "vsweep_mean_V": vsweep_mean.value,
            "isweep_mean_A": isweep_mean.value,
            "vsweep_mean_smoothed_V": vsweep_mean_smoothed.value,
            "isweep_mean_smoothed_A": isweep_mean_smoothed.value,
            "vsweep_shot_smoothed_V": vsweep_shot_smoothed.value,
            "isweep_shot_smoothed_A": isweep_shot_smoothed.value,
            "iv_voltage_grid_V": iv_voltage_grid,
            "iv_current_grid_A": iv_current_grid,
            "iv_didv_grid_A_per_V": iv_didv_grid,
            "iv_te_fit_mask": iv_fit_mask_grid,
            "iv_voltage_grid_shot_V": iv_voltage_grid_shot,
            "iv_current_grid_shot_A": iv_current_grid_shot,
            "iv_didv_grid_shot_A_per_V": iv_didv_grid_shot,
            "iv_te_fit_mask_shot": iv_fit_mask_grid_shot,
            "te_fit_r2": te_fit_r2,
            "te_fit_rmse_logI": te_fit_rmse,
            "te_fit_npts": te_fit_npts,
            "te_fit_vstart_V": te_fit_vstart,
            "te_fit_vstop_V": te_fit_vstop,
            "te_fit_slope_logI_per_V": te_fit_slope,
            "te_fit_intercept_logI": te_fit_intercept,
            "te_fit_i0_A": te_fit_i0,
            "te_subtract_i0": np.array(int(te_subtract_i0)),
            "te_fit_passed_r2": te_fit_passed_r2,
            "te_fit_candidate_count": te_fit_candidate_count,
            "te_fit_r2_std": te_fit_r2_std,
            "te_fit_rmse_logI_std": te_fit_rmse_std,
            "te_fit_npts_std": te_fit_npts_std,
            "te_fit_vstart_std_V": te_fit_vstart_std,
            "te_fit_vstop_std_V": te_fit_vstop_std,
            "te_fit_slope_std_logI_per_V": te_fit_slope_std,
            "te_fit_intercept_std_logI": te_fit_intercept_std,
            "te_fit_i0_std_A": te_fit_i0_std,
            "te_fit_count": te_fit_count,
            "te_fit_r2_shot": te_fit_r2_shot,
            "te_fit_rmse_logI_shot": te_fit_rmse_shot,
            "te_fit_npts_shot": te_fit_npts_shot,
            "te_fit_vstart_shot_V": te_fit_vstart_shot,
            "te_fit_vstop_shot_V": te_fit_vstop_shot,
            "te_fit_slope_shot_logI_per_V": te_fit_slope_shot,
            "te_fit_intercept_shot_logI": te_fit_intercept_shot,
            "te_fit_i0_shot_A": te_fit_i0_shot,
            "te_fit_passed_r2_shot": te_fit_passed_r2_shot,
            "te_fit_candidate_count_shot": te_fit_candidate_count_shot,
            "shot_standard_deviation_ddof": np.array(1),
            "per_shot_axis_order": np.array("y,x,shot"),
            "shot_statistics_stage": np.array(shot_statistics_stage),
            "profile_value_policy": np.array(
                "finite estimates retained; analysis_ok records fit acceptance"
            ),
            "xy_shape_info": np.array([ny, nx, nshots, nt_full, nt, iv_npts], dtype=np.int64),
            "trace_spatial_shape": np.array(spatial_shape, dtype=np.int64),
            "dt_s": np.array(dt),
            "te_min_r2": np.array(te_min_r2),
            "ion_min_snr": np.array(ion_min_snr),
            "ies_method": np.array(ies_method),
            "ies_definition": np.array(describe_ies_method(ies_method)),
            "analysis_model": np.array(
                "single-temperature semilog fit; median low-bias ion estimate; "
                + describe_ies_method(ies_method)
            ),
            "te_fit_i0_definition": np.array("median low-bias ion-region current"),
            "iv_npts_role": np.array("fixed-size diagnostics only"),
            "current_zero_calibrated": np.array(int(current_zero_calibrated)),
            "negate_Isweep_current": np.array(int(negate_Isweep_current)),
            "enforce_ideal_model_checks": np.array(int(enforce_ideal_model_checks)),
            "subtract_dc": np.array(int(subtract_dc)),
            "summary_plot_path": np.array("" if summary_plot_path is None else str(summary_plot_path)),
            "all_iv_plot_path": np.array("" if all_iv_plot_path is None else str(all_iv_plot_path)),
        }
        if subtract_dc:
            xy_npz_data["isweep_dc_offsets_A"] = isweep_dc_offsets
        if plot_png_bytes is not None:
            xy_npz_data["summary_plot_png"] = np.frombuffer(plot_png_bytes, dtype=np.uint8)
        if all_iv_plot_png_bytes is not None:
            xy_npz_data["all_iv_curves_plot_png"] = np.frombuffer(all_iv_plot_png_bytes, dtype=np.uint8)

        np.savez_compressed(output_npz_path, **xy_npz_data)

        print(f"Saved Langmuir XY-plane results to: {output_h5_path}")
        print(f"Saved Langmuir XY-plane NPZ results to: {output_npz_path}")

    # %%


def _build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-parameters",
        metavar="FILE",
        help="Run one saved GUI configuration instead of opening the GUI.",
    )
    parser.add_argument(
        "--geometry",
        choices=SUPPORTED_GEOMETRIES,
        help="Geometry to run with --run-parameters.",
    )
    return parser


def main(argv=None):
    """Open the GUI, or execute the private saved-parameter worker mode."""
    arguments = _build_argument_parser().parse_args(argv)
    if arguments.run_parameters:
        if arguments.geometry is None:
            raise SystemExit("--geometry is required with --run-parameters")
        parameters, _active_geometry, warning = load_last_parameters(
            arguments.run_parameters
        )
        if warning:
            raise SystemExit(warning)
        return run_analysis(arguments.geometry, parameters[arguments.geometry])

    if arguments.geometry is not None:
        raise SystemExit("--geometry can only be used with --run-parameters")
    from langmuir_analysis_gui import launch_gui

    return launch_gui()


if __name__ == "__main__":
    main()
