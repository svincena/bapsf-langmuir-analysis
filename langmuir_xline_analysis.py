# %%
import numpy as np
import matplotlib
# %matplotlib qt
matplotlib.use('Qt5Agg')  # or 'TkAgg'
# matplotlib.use('TkAgg')   # or 'Qt5Agg'
import matplotlib.pyplot as plt
from bapsflib import lapd
import h5py
import astropy.units as u
import io
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import savgol_filter
from langmuir_analysis import analyze_iv_trace
from langmuir_diagnostics import (
    render_iv_diagnostic_plot as render_analysis_iv_diagnostic_plot,
)


save_results = True
plot_results = True

# Draw shot-to-shot +/-1 sigma error bars when more than one shot is available.
plot_summary_stds = True

# Integrate the normalized electron-density profile across x and report the
# resulting effective profile width in the summary plot.
calculate_shape_factor = True

# The first 128 samples provide a plasma-off electronics baseline, independent
# of the later I-V sweep.  Do not replace this with low-bias sweep samples;
# doing so would erase the physical ion current and bias Vf.
subtract_dc = False
current_zero_calibrated = True
# Set this to match the current polarity from the acquisition electronics. The
# analysis expects negative ion current and positive electron current.
negate_Isweep_current = True

# Real probe saturation branches are normally sloped. Keep ideal planar-
# Maxwellian consistency checks as diagnostics instead of rejecting good data.
enforce_ideal_model_checks = False

# Use multiprocessing to analyze independent traces in parallel.
parallel_analysis = True
analysis_processes = None  # None uses one fewer than the detected CPU count

filename = '/Users/vincena/data/Mini_Magnetospheres/August2026/run_06_langmuir_xline_dipole_removed 2026-08-26 09.33.06.hdf5'
# digitizer = "SIS 3301" # for 3301, this is also the name of the adc
# adc = "SIS 3301" #8-channel 14-bit 100 MS/s digitizer
adc = "SIS 3302" #8-channel 16-bit 100 MS/s digitizer
digitizer = "SIS crate"
sis_config_name = "Isat_Isweep_Vsweep_282624S_50MHz"



nshots = 5
nt_full = 282_624

nx = 81
x_min = -37.5
x_max = 37.5
x = np.linspace(x_min, x_max, nx)

board = 2
vsweep_channel = 3
isweep_channel = 2

vsweep_attenuation = 100.0
isweep_attenuation = 1.0 #4.0
isweep_resistance = 3.1 #1.0
probe_area = 4.0 * u.mm**2



# data_offset used if, say, skipping half the data where 2 probes move but only
# one at a time and half the data is being taken on a probe when it's sitting at
# the start or end of its motion list
data_offset = 0

shotnum_start = data_offset + 1
shotnum_end = nx * nshots + shotnum_start   # exclusive upper bound in slice(...)
n_expected_shots = shotnum_end - shotnum_start


first_sweep_index = 5010
sweep_start_index = first_sweep_index
sweep_end_index = 15000
nt = sweep_end_index - sweep_start_index + 1

isweep_dc_offset_start_index = 0
isweep_dc_offset_end_index = 127

isat_start_index = 0
isat_end_index = 127

# Smoothing along original time-ordered sweep
# sg_smooth_bins = 8
# sg_smooth_order = 1
sg_smooth_bins = 32
sg_smooth_order = 2


# Example diagnostic plot controls
diagnostic_plot_every = 2       # make an IV diagnostic plot every N x indices; 0 disables these plots
example_pause_seconds = 1.0
diagnostic_plot_output_dir = Path("output_diagnostic_plots")

# Interpolated monotonic I-V grid parameters
iv_npts = 2048
voltage_bin_width = 0.05  # volts; merges near-duplicate voltages before interpolation

# Derivative smoothing is specified in volts and therefore does not change when
# the fixed-size diagnostic grid resolution changes.
vp_smoothing = "savgol"
vp_smoothing_width_V = 2.5
vp_savgol_order = 2

# One Maxwellian fit on independent voltage-bin means.
te_min_points = 12
te_margin_from_vp = 0.2  # volts
te_current_floor_frac = 0.03
te_min_eV = 0.05
te_max_eV = 20.0
te_min_r2 = 0.98
# Uses autocorrelation-adjusted uncertainty of the regional median, not the
# point-to-point residual scatter.
ion_min_snr = 3.0
te_subtract_i0 = True  # legacy export name; I0 is now the median ion-region current

langmuir_analysis_config = {
    "iv_npts": iv_npts,
    "voltage_bin_width": voltage_bin_width,
    "vp_smoothing": vp_smoothing,
    "vp_smoothing_width_V": vp_smoothing_width_V,
    "vp_savgol_order": vp_savgol_order,
    "te_min_points": te_min_points,
    "te_margin_from_vp": te_margin_from_vp,
    "te_current_floor_frac": te_current_floor_frac,
    "te_min_eV": te_min_eV,
    "te_max_eV": te_max_eV,
    "te_min_r2": te_min_r2,
    "ion_min_snr": ion_min_snr,
    "probe_area": probe_area,
    "current_zero_calibrated": current_zero_calibrated,
    "enforce_ideal_model_checks": enforce_ideal_model_checks,
}

# Optional post-processing controls along x
enable_vp_spike_rejection = True
vp_spike_half_window = 2          # neighbors on each side for local median
vp_spike_threshold_V = 5.0        # flag if |Vp - local median| exceeds this
vp_replace_flagged_with_local_interp = True

enable_neighbor_smoothing = True
neighbor_smooth_half_window = 1   # 1 means 3-point neighborhood, 2 means 5-point
neighbor_smooth_sigma = 1.0       # gaussian-like weighting in index space

make_all_iv_diagnostic_plot = True


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


def render_summary_plot(
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


def render_all_iv_curves_plot(x, iv_voltage_grid, iv_current_grid):
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
trace_shape = spatial_shape + (nshots,)
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
            vsweep_shot.value[idx + (slice(None),)],
            isweep_shot.value[idx + (slice(None),)],
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
                fig = render_analysis_iv_diagnostic_plot(
                    result,
                    f"Diagnostic IV analysis at x index {ix}, shot {ishot} (x = {x[ix]:.2f} cm)",
                )
                save_diagnostic_figure(
                    fig,
                    diagnostic_plot_output_dir,
                    f"{Path(filename).stem}_iv_diagnostic_ix{ix:03d}_shot{ishot:03d}_x_{x[ix]:+.2f}cm",
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
            fig = render_analysis_iv_diagnostic_plot(
                result,
                f"Diagnostic IV analysis at x index {ix}, shot {ishot} (x = {x[ix]:.2f} cm)",
            )
            save_diagnostic_figure(
                fig,
                diagnostic_plot_output_dir,
                f"{Path(filename).stem}_iv_diagnostic_ix{ix:03d}_shot{ishot:03d}_x_{x[ix]:+.2f}cm",
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
analysis_status = (
    f"Reported {reported_trace_count}/{n_traces} finite fits; "
    f"accepted {accepted_trace_count}/{n_traces} fits"
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

print("Finished per-shot Langmuir probe analysis and spatial statistics")

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
shape_factor = (
    electron_density_profile_shape_factor(x, n_e_plot)
    if calculate_shape_factor
    else None
)
# %%
# Plot results

plot_png_bytes = None
all_iv_plot_png_bytes = None
summary_plot_path = None
all_iv_plot_path = None

if plot_results:
    fig = render_summary_plot(
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
        plot_stds=plot_summary_stds and nshots > 1,
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
    fig_iv = render_all_iv_curves_plot(x, iv_voltage_grid, iv_current_grid)

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
            fig = render_summary_plot(
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
                plot_stds=plot_summary_stds and nshots > 1,
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
        grp.attrs["n_traces"] = n_traces
        grp.attrs["trace_spatial_ndim"] = len(spatial_shape)
        grp.attrs["nshots"] = nshots
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
            "single-temperature semilog fit; median low-bias ion and high-bias "
            "electron-saturation region estimates"
        )
        grp.attrs["te_fit_i0_definition"] = "median low-bias ion-region current"
        grp.attrs["iv_npts_role"] = "fixed-size diagnostics only"
        grp.attrs["current_zero_calibrated"] = int(current_zero_calibrated)
        grp.attrs["negate_Isweep_current"] = int(negate_Isweep_current)
        grp.attrs["enforce_ideal_model_checks"] = int(enforce_ideal_model_checks)
        grp.attrs["subtract_dc"] = int(subtract_dc)
        grp.attrs["shot_standard_deviation_ddof"] = 1
        grp.attrs["per_shot_axis_order"] = "x,shot"
        grp.attrs["shot_statistics_stage"] = "individual fits before spatial post-processing"
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
        "shot_statistics_stage": np.array("individual fits before spatial post-processing"),
        "profile_value_policy": np.array(
            "finite estimates retained; analysis_ok records fit acceptance"
        ),
        "xline_shape_info": np.array([nx, nshots, nt_full, nt, iv_npts], dtype=np.int64),
        "trace_spatial_shape": np.array(spatial_shape, dtype=np.int64),
        "dt_s": np.array(dt),
        "te_min_r2": np.array(te_min_r2),
        "ion_min_snr": np.array(ion_min_snr),
        "analysis_model": np.array(
            "single-temperature semilog fit; median low-bias ion and high-bias "
            "electron-saturation region estimates"
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
        xline_npz_data["isweep_dc_offsets_A"] = isweep_dc_offsets
    if plot_png_bytes is not None:
        xline_npz_data["summary_plot_png"] = np.frombuffer(plot_png_bytes, dtype=np.uint8)
    if all_iv_plot_png_bytes is not None:
        xline_npz_data["all_iv_curves_plot_png"] = np.frombuffer(all_iv_plot_png_bytes, dtype=np.uint8)

    np.savez_compressed(output_npz_path, **xline_npz_data)

    print(f"Saved Langmuir x-line results to: {output_h5_path}")
    print(f"Saved Langmuir x-line NPZ results to: {output_npz_path}")

# %%
