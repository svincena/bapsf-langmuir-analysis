# %%
import numpy as np
import matplotlib
%matplotlib qt
# matplotlib.use('Qt5Agg')  # or 'TkAgg'
# matplotlib.use('TkAgg')   # or 'Qt5Agg'
import matplotlib.pyplot as plt
from bapsflib import lapd
import h5py
import astropy.units as u
import io
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import savgol_filter
from langmuir_analysis import (
    analyze_iv_trace,
    choose_analysis_process_count,
    render_iv_diagnostic_plot as render_analysis_iv_diagnostic_plot,
)


save_results = True
plot_results = True
subtract_dc = True
parallel_analysis = True
analysis_processes = None  # None uses one fewer than the detected CPU count

# Shot statistics.  Every shot is always fitted independently.  Leave this
# False to include every finite fit in the per-location statistics; it can be
# enabled later to exclude shots whose Te fit did not meet te_min_r2.
exclude_poor_fits_from_statistics = False

# filename = "/Users/vincena/data/Vincena/ions01/05_Lang_x_p29_4mm2_facing_antenna_2_2026-04-04_18.19.56.hdf5"
filename = "/Users/vincena/data/ie01/hdf/p37x_lang_35V_redo_fine.hdf5"
digitizer = "SIS crate"
digitizer = "SIS 3301" # for 3301, this is also the name of the adc
# adc = "SIS 3302" #8-channel 16-bit 100 MS/s digitizer
adc = "SIS 3301" #8-channel 14-bit 100 MS/s digitizer
sis_config_name = "ions01_Lang_really_really"
sis_config_name = "fast_lang_board3"

# nx = 163
# nshots = 5
# nt_full = 712704
nx = 81
nshots = 1
nt_full = 32768

board = 3
# vsweep_channel = 7
# isweep_channel = 8
vsweep_channel = 0
isweep_channel = 1

vsweep_attenuation = 100.0
isweep_attenuation = 2.0 #4.0
isweep_resistance = 10.0 #1.0
probe_area = 4.0 * u.mm**2

x = np.linspace(-31.0, 50.0, nx)
x = np.linspace(-10.0, 10.0, nx)

# data_offset used if, say, skipping half the data where 2 probes move but only
# one at a time and half the data is being taken on a probe when it's sitting at
# the start or end of its motion list
data_offset = 0

shotnum_start = data_offset + 1
shotnum_end = nx * nshots + shotnum_start   # exclusive upper bound in slice(...)
n_expected_shots = shotnum_end - shotnum_start

# first_sweep_index = 550_000
# sweep_start_index = first_sweep_index + 1_500
# sweep_end_index = first_sweep_index + 2_500
# nt = sweep_end_index - sweep_start_index + 1
first_sweep_index = 15000
sweep_start_index = first_sweep_index
sweep_end_index = 21000
nt = sweep_end_index - sweep_start_index + 1

isweep_dc_offset_start_index = first_sweep_index
isweep_dc_offset_end_index = isweep_dc_offset_start_index + 128 - 1

isat_start_index = 0
isat_end_index = 127

# Smoothing along original time-ordered sweep
# sg_smooth_bins = 8
# sg_smooth_order = 1
sg_smooth_bins = 32
sg_smooth_order = 2


# Example diagnostic plot controls
diagnostic_plot_every = 8       # make an IV diagnostic plot every N x indices; 0 disables these plots
example_pause_seconds = 1.0
diagnostic_plot_output_dir = Path("output_diagnostic_plots")

# Interpolated monotonic I-V grid parameters
iv_npts = 1024
voltage_bin_width = 0.05  # volts; merges near-duplicate voltages before interpolation

# Derivative smoothing on interpolated IV curve for plasma-potential detection.
# Use None, "moving", or "savgol".
vp_smoothing = "savgol"
vp_moving_average_width = 10
iv_savgol_window = 31
iv_savgol_order = 2

# Adaptive Te fit controls
te_min_points = int(0.05 * iv_npts)
te_max_points = int(0.2* iv_npts)
te_margin_from_vp = 0.1  # eV; require Te fit range to be at least this far from detected Vp
te_current_floor_frac = 0.03
te_min_eV = 0.05
te_max_eV = 30.0
te_min_r2 = 0.90
te_subtract_i0 = True

langmuir_analysis_config = {
    "iv_npts": iv_npts,
    "voltage_bin_width": voltage_bin_width,
    "vp_smoothing": vp_smoothing,
    "vp_moving_average_width": vp_moving_average_width,
    "iv_savgol_window": iv_savgol_window,
    "iv_savgol_order": iv_savgol_order,
    "te_min_points": te_min_points,
    "te_max_points": te_max_points,
    "te_margin_from_vp": te_margin_from_vp,
    "te_current_floor_frac": te_current_floor_frac,
    "te_min_eV": te_min_eV,
    "te_max_eV": te_max_eV,
    "te_min_r2": te_min_r2,
    "te_subtract_i0": te_subtract_i0,
    "probe_area": probe_area,
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
    vp_spike_mask=None,
    vp_raw=None,
    te_fit_r2=None,
    te_poor_fit_r2=None,
    title="Langmuir X-Line Summary",
):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    def _plot_profile(ax, values, plot_title, y_label):
        ax.plot(x, values.value)
        ax.set_title(plot_title)
        ax.set_xlabel("X (cm)")
        ax.set_ylabel(y_label)
        ax.grid(True)

    _plot_profile(axs[0, 0], te, "Electron Temperature", "T_e (eV)")
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

    axs[0, 1].plot(x, vp.value, label="Vp")
    if vp_raw is not None:
        axs[0, 1].plot(x, vp_raw.value, alpha=0.35, label="Vp raw")
    if vp_spike_mask is not None and vp_raw is not None:
        m = vp_spike_mask.astype(bool)
        if np.any(m):
            axs[0, 1].plot(x[m], vp_raw.value[m], "o", label="flagged spikes")
    axs[0, 1].set_title("Plasma Potential")
    axs[0, 1].set_xlabel("X (cm)")
    axs[0, 1].set_ylabel("V_p (V)")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    _plot_profile(axs[1, 0], vf, "Floating Potential", "V_f (V)")
    _plot_profile(axs[1, 1], ies, "Electron Saturation Current", "I_es (A)")
    _plot_profile(axs[2, 0], iis, "Ion Saturation Current", "I_is (A)")
    if n_e is not None:
        _plot_profile(axs[2, 1], n_e, "Electron Density", "n_e (m^-3)")
    else:
        axs[2, 1].axis("off")

    fig.suptitle(title, fontsize=16)
    return fig


def render_all_iv_curves_plot(x, iv_voltage_grid, iv_current_grid):
    """Plot all interpolated I-V curves colored by x position."""
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    cmap = plt.get_cmap("viridis")
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)

    for i in range(len(x)):
        V = iv_voltage_grid[i]
        I = iv_current_grid[i]
        good = np.isfinite(V) & np.isfinite(I)
        if np.count_nonzero(good) < 2:
            continue

        frac = 0.5 if x_max == x_min else (x[i] - x_min) / (x_max - x_min)
        ax.plot(V[good], I[good], color=cmap(frac), alpha=0.9, lw=1.2)

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
    negate=True,
)
print(f"Current data reshaped to (x, shots, time): {isweep_full.shape}")

print("Extracting sweep regions")
vsweep = vsweep_full[..., sweep_start_index:sweep_end_index + 1]
isweep = isweep_full[..., sweep_start_index:sweep_end_index + 1]

if subtract_dc:
    print("Subtracting current DC offsets")
    isweep_dc_offset_portion = isweep_full[..., isweep_dc_offset_start_index:isweep_dc_offset_end_index + 1]
    isweep_dc_offsets = np.mean(isweep_dc_offset_portion, axis=-1)
    isweep = isweep - isweep_dc_offsets[..., np.newaxis]

file.close()
print("Finished reading and reshaping data.")

# %%
# Lightly smooth every shot along the original sweep index

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

te_fit_r2_shot = np.full(trace_shape, np.nan)
te_fit_rmse_shot = np.full(trace_shape, np.nan)
te_fit_npts_shot = np.full(trace_shape, 0, dtype=int)
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
    include_diagnostic_data = (
        diagnostic_plot_every is not None
        and diagnostic_plot_every > 0
        and flat_index % diagnostic_plot_every == 0
    )
    analysis_tasks.append(
        (
            flat_index,
            idx,
            vsweep_shot_smoothed.value[idx + (slice(None),)],
            isweep_shot_smoothed.value[idx + (slice(None),)],
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

            if result["iv_voltage_grid"] is None:
                continue

            te_shot[idx] = result["te_eV"]
            vp_shot[idx] = result["vp_V"]
            vf_shot[idx] = result["vf_V"]
            ies_shot[idx] = result["ies_A"]
            iis_shot[idx] = result["iis_A"]
            n_e_shot[idx] = result.get("n_e_m3", np.nan)

            te_fit_r2_shot[idx] = result["te_fit_r2"]
            te_fit_rmse_shot[idx] = result["te_fit_rmse"]
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

        if result["iv_voltage_grid"] is None:
            continue

        te_shot[idx] = result["te_eV"]
        vp_shot[idx] = result["vp_V"]
        vf_shot[idx] = result["vf_V"]
        ies_shot[idx] = result["ies_A"]
        iis_shot[idx] = result["iis_A"]
        n_e_shot[idx] = result.get("n_e_m3", np.nan)

        te_fit_r2_shot[idx] = result["te_fit_r2"]
        te_fit_rmse_shot[idx] = result["te_fit_rmse"]
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

statistics_fit_mask = (
    te_fit_passed_r2_shot.astype(bool)
    if exclude_poor_fits_from_statistics
    else np.ones(trace_shape, dtype=bool)
)

te_values, te_std_values, fit_count = shot_mean_and_std(te_shot, statistics_fit_mask)
vp_values, vp_std_values, _ = shot_mean_and_std(vp_shot, statistics_fit_mask)
vf_values, vf_std_values, _ = shot_mean_and_std(vf_shot, statistics_fit_mask)
ies_values, ies_std_values, _ = shot_mean_and_std(ies_shot, statistics_fit_mask)
iis_values, iis_std_values, _ = shot_mean_and_std(iis_shot, statistics_fit_mask)
n_e_values, n_e_std_values, _ = shot_mean_and_std(n_e_shot, statistics_fit_mask)

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
te_fit_passed_r2 = np.all(te_fit_passed_r2_shot.astype(bool), axis=-1).astype(np.uint8)
te_fit_candidate_count = np.sum(te_fit_candidate_count_shot, axis=-1)

# Backward-compatible representative I-V products are the shot means.  The
# full per-shot grids are exported separately below.
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
        vp_spike_mask=vp_spike_mask,
        vp_raw=vp_raw,
        te_fit_r2=te_fit_r2,
        te_poor_fit_r2=te_min_r2,
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
                vp_spike_mask=vp_spike_mask,
                vp_raw=vp_raw,
                te_fit_r2=te_fit_r2,
                te_poor_fit_r2=te_min_r2,
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
        grp.attrs["vp_moving_average_width"] = vp_moving_average_width
        grp.attrs["iv_savgol_window"] = iv_savgol_window
        grp.attrs["iv_savgol_order"] = iv_savgol_order
        grp.attrs["parallel_analysis"] = int(parallel_analysis)
        grp.attrs["analysis_processes_requested"] = (
            -1 if analysis_processes is None else int(analysis_processes)
        )
        grp.attrs["analysis_processes_used"] = analysis_processes_used
        grp.attrs["analysis_start_method"] = "fork" if use_parallel_analysis else "serial"

        grp.attrs["te_min_points"] = te_min_points
        grp.attrs["te_max_points"] = te_max_points
        grp.attrs["te_margin_from_vp_V"] = te_margin_from_vp
        grp.attrs["te_current_floor_frac"] = te_current_floor_frac
        grp.attrs["te_subtract_i0"] = int(te_subtract_i0)
        grp.attrs["te_min_eV"] = te_min_eV
        grp.attrs["te_max_eV"] = te_max_eV
        grp.attrs["te_min_r2"] = te_min_r2
        grp.attrs["exclude_poor_fits_from_statistics"] = int(exclude_poor_fits_from_statistics)
        grp.attrs["shot_standard_deviation_ddof"] = 1
        grp.attrs["per_shot_axis_order"] = "x,shot"
        grp.attrs["shot_statistics_stage"] = "individual fits before spatial post-processing"

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
        "exclude_poor_fits_from_statistics": np.array(int(exclude_poor_fits_from_statistics)),
        "shot_standard_deviation_ddof": np.array(1),
        "per_shot_axis_order": np.array("x,shot"),
        "shot_statistics_stage": np.array("individual fits before spatial post-processing"),
        "xline_shape_info": np.array([nx, nshots, nt_full, nt, iv_npts], dtype=np.int64),
        "trace_spatial_shape": np.array(spatial_shape, dtype=np.int64),
        "dt_s": np.array(dt),
        "te_min_r2": np.array(te_min_r2),
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
