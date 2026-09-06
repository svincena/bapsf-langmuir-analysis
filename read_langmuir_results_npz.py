"""Read and optionally plot x-line or xy-plane Langmuir NPZ results.

The result geometry is detected from the scalar ``geometry`` field stored in
the NPZ file.  Both result types use the same loader, schema validation,
metadata summary, and command-line interface; only the spatial plot is
geometry-specific.

Examples
--------
Print a result inventory::

    python read_langmuir_results_npz.py path/to/run_langmuir_xline.npz
    python read_langmuir_results_npz.py path/to/run_langmuir_xy.npz

Show the matching one- or two-dimensional summary plot::

    python read_langmuir_results_npz.py path/to/results.npz --plot
"""

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np


SUPPORTED_LANGMUIR_GEOMETRIES = {"x_line", "xy_plane"}

_COMMON_EXPECTED_NPZ_KEYS = {
    "source_file",
    "geometry",
    "x_cm",
    "time_s",
    "te_eV",
    "vp_V",
    "vf_V",
    "ies_A",
    "iis_A",
    "isat_A",
    "n_e_m3",
    "te_raw_eV",
    "vp_raw_V",
    "vf_raw_V",
    "ies_raw_A",
    "iis_raw_A",
    "n_e_raw_m3",
    "te_processed_eV",
    "vp_processed_V",
    "vf_processed_V",
    "ies_processed_A",
    "iis_processed_A",
    "n_e_processed_m3",
    "vp_spike_mask",
    "vsweep_mean_V",
    "isweep_mean_A",
    "vsweep_mean_smoothed_V",
    "isweep_mean_smoothed_A",
    "iv_voltage_grid_V",
    "iv_current_grid_A",
    "iv_didv_grid_A_per_V",
    "iv_te_fit_mask",
    "te_fit_r2",
    "te_fit_rmse_logI",
    "te_fit_npts",
    "te_fit_vstart_V",
    "te_fit_vstop_V",
    "te_fit_slope_logI_per_V",
    "te_fit_intercept_logI",
    "te_fit_i0_A",
    "te_subtract_i0",
    "te_fit_passed_r2",
    "te_fit_candidate_count",
    "trace_spatial_shape",
    "dt_s",
    "te_min_r2",
    "summary_plot_path",
    "all_iv_plot_path",
}

EXPECTED_LANGMUIR_XLINE_NPZ_KEYS = _COMMON_EXPECTED_NPZ_KEYS | {
    "xline_shape_info",
}
EXPECTED_LANGMUIR_XY_NPZ_KEYS = _COMMON_EXPECTED_NPZ_KEYS | {
    "y_cm",
    "X_cm",
    "Y_cm",
    "xy_shape_info",
}
EXPECTED_NPZ_KEYS_BY_GEOMETRY = {
    "x_line": EXPECTED_LANGMUIR_XLINE_NPZ_KEYS,
    "xy_plane": EXPECTED_LANGMUIR_XY_NPZ_KEYS,
}


def _scalar_value(value, name):
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar; got shape {array.shape}.")
    value = array.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _scalar_text(value):
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array)


def get_langmuir_result_geometry(data):
    """Return and validate the scalar geometry marker in a result mapping."""
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping of NPZ result arrays.")
    if "geometry" not in data:
        raise ValueError("Langmuir result is missing the required geometry field.")

    geometry = str(_scalar_value(data["geometry"], "geometry"))
    if geometry not in SUPPORTED_LANGMUIR_GEOMETRIES:
        choices = ", ".join(sorted(SUPPORTED_LANGMUIR_GEOMETRIES))
        raise ValueError(
            f"Unsupported Langmuir result geometry {geometry!r}; "
            f"expected one of {{{choices}}}."
        )
    return geometry


def validate_langmuir_results_npz(data, expected_geometry=None):
    """Validate the geometry and stable schema while allowing extra fields."""
    geometry = get_langmuir_result_geometry(data)
    if expected_geometry is not None:
        if expected_geometry not in SUPPORTED_LANGMUIR_GEOMETRIES:
            raise ValueError(f"Unknown expected geometry {expected_geometry!r}.")
        if geometry != expected_geometry:
            raise ValueError(
                f"Expected {expected_geometry!r} Langmuir results, "
                f"but the file contains {geometry!r} results."
            )

    missing = EXPECTED_NPZ_KEYS_BY_GEOMETRY[geometry] - set(data)
    if missing:
        raise ValueError(
            f"Loaded NPZ file does not conform to expected {geometry} "
            f"Langmuir schema: missing keys: {sorted(missing)}"
        )
    return data


def validate_langmuir_xline_npz(data):
    """Validate an x-line result mapping."""
    return validate_langmuir_results_npz(data, expected_geometry="x_line")


def validate_langmuir_xy_npz(data):
    """Validate an xy-plane result mapping."""
    return validate_langmuir_results_npz(data, expected_geometry="xy_plane")


def load_langmuir_results_npz(path, expected_geometry=None):
    """Load and validate either geometry from a Langmuir result NPZ file."""
    with np.load(path, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    return validate_langmuir_results_npz(
        data,
        expected_geometry=expected_geometry,
    )


def load_langmuir_xline_npz(path):
    """Compatibility loader that requires x-line results."""
    return load_langmuir_results_npz(path, expected_geometry="x_line")


def load_langmuir_xy_npz(path):
    """Compatibility loader that requires xy-plane results."""
    return load_langmuir_results_npz(path, expected_geometry="xy_plane")


def electron_density_profile_shape_factor_m(x_cm, electron_density_profile):
    """Integrate peak-normalized electron density over x in meters."""
    x_values_cm = np.asarray(x_cm, dtype=float)
    profile_values = np.asarray(electron_density_profile, dtype=float)
    if x_values_cm.shape != profile_values.shape:
        raise ValueError(
            "x coordinates and electron-density profile must have the same shape; "
            f"got {x_values_cm.shape} and {profile_values.shape}."
        )

    finite = np.isfinite(x_values_cm) & np.isfinite(profile_values)
    if np.count_nonzero(finite) < 2:
        return np.nan

    finite_profile = profile_values[finite]
    profile_maximum = np.max(finite_profile)
    if not np.isfinite(profile_maximum) or profile_maximum == 0.0:
        return np.nan

    x_values_m = x_values_cm[finite] * 1.0e-2
    normalized_profile = finite_profile / profile_maximum
    return float(
        np.sum(
            np.diff(x_values_m)
            * 0.5
            * (normalized_profile[:-1] + normalized_profile[1:])
        )
    )


def _shape_factor_m(data):
    if "shape_factor_m" in data:
        stored = np.asarray(data["shape_factor_m"], dtype=float)
        if stored.shape == ():
            value = float(stored)
            if np.isfinite(value):
                return value
    if get_langmuir_result_geometry(data) == "x_line":
        if np.asarray(data["n_e_m3"]).ndim != 1:
            return np.nan
        return electron_density_profile_shape_factor_m(
            data["x_cm"],
            data["n_e_m3"],
        )
    return np.nan


def print_summary(data):
    """Print a compact inventory and fit-quality summary for either geometry."""
    geometry = get_langmuir_result_geometry(data)
    print(f"source_file: {_scalar_text(data.get('source_file', ''))}")
    print(f"geometry: {geometry}")

    for key in (
        "shot_analysis_mode",
        "shot_statistics_available",
        "calculate_shape_factor",
        "shape_factor_m",
        "density_scale",
        "interferometer_line_integrated_density_m2",
        "interferometer_profile_y_requested_cm",
        "interferometer_profile_y_selected_cm",
        "interferometer_profile_y_index",
        # Legacy XY calibration fields remain readable.
        "scale_to_interferometer_density",
        "interferometer_line_averaged_density_m3",
        "current_zero_calibrated",
        "negate_Isweep_current",
        "subtract_dc",
    ):
        if key in data:
            print(f"{key}: {_scalar_text(data[key])}")

    coordinate_keys = (
        ("x_cm", "X_cm", "Y_cm")
        if geometry == "x_line"
        else ("x_cm", "y_cm", "X_cm", "Y_cm")
    )
    for key in coordinate_keys + (
        "te_eV",
        "vp_V",
        "vf_V",
        "ies_A",
        "iis_A",
        "n_e_m3",
        "te_fit_r2",
    ):
        if key in data:
            array = np.asarray(data[key])
            print(f"{key}: shape={array.shape}, dtype={array.dtype}")

    shape_factor_m = _shape_factor_m(data)
    if np.isfinite(shape_factor_m):
        label = "electron-density shape factor"
        if geometry == "xy_plane":
            label = "interferometer x-line shape factor"
        print(f"{label}: {shape_factor_m:.4g} m")

    if "analysis_valid_count" in data:
        analysis_label = (
            "accepted averaged analyses"
            if _scalar_text(data.get("shot_analysis_mode", "individual"))
            == "average"
            else "accepted shot analyses"
        )
        print(
            f"{analysis_label}: "
            f"{int(np.sum(np.asarray(data['analysis_valid_count'])))}"
        )
    elif "te_raw_eV" in data:
        print(
            "accepted spatial results: "
            f"{np.count_nonzero(np.isfinite(data['te_raw_eV']))}"
        )

    if "te_fit_r2" in data:
        min_r2 = float(np.asarray(data.get("te_min_r2", 0.90)))
        fit_r2 = np.asarray(data["te_fit_r2"], dtype=float)
        finite = np.isfinite(fit_r2)
        poor = finite & (fit_r2 < min_r2)
        print(f"finite Te fit diagnostics: {np.count_nonzero(finite)}")
        print(f"Te fits below R^2 {min_r2:.2f}: {np.count_nonzero(poor)}")


def plot_xline_summary(data):
    """Make the six-panel profile summary for x-line results."""
    import matplotlib.pyplot as plt

    if np.asarray(data["te_eV"]).ndim == 2:
        return plot_xline_multi_ramp_summary(data)

    x = np.asarray(data["x_cm"], dtype=float)
    panels = [
        ((0, 0), "te_eV", "te_std_eV", "Electron Temperature", "T_e (eV)"),
        ((0, 1), "vp_V", "vp_std_V", "Plasma Potential", "V_p (V)"),
        ((1, 0), "vf_V", "vf_std_V", "Floating Potential", "V_f (V)"),
        ((1, 1), "ies_A", "ies_std_A", "Electron Saturation Current", "I_es (A)"),
        ((2, 0), "iis_A", "iis_std_A", "Ion Saturation Current", "I_is (A)"),
        ((2, 1), "n_e_m3", "n_e_std_m3", "Electron Density", "n_e (m^-3)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    finite_x = x[np.isfinite(x)]
    analysis_ok = (
        np.asarray(data["analysis_ok"], dtype=bool)
        if "analysis_ok" in data
        else None
    )

    for (row, column), key, std_key, title, y_label in panels:
        axis = axes[row, column]
        if key not in data:
            axis.axis("off")
            continue

        values = np.asarray(data[key], dtype=float)
        finite = np.isfinite(x) & np.isfinite(values)
        if np.any(finite):
            line = axis.plot(x[finite], values[finite])[0]
            if std_key in data:
                std = np.asarray(data[std_key], dtype=float)
                finite_std = finite & np.isfinite(std)
                if np.any(finite_std):
                    axis.errorbar(
                        x[finite_std],
                        values[finite_std],
                        yerr=std[finite_std],
                        fmt="none",
                        ecolor=line.get_color(),
                        elinewidth=1.0,
                        capsize=3,
                        alpha=0.7,
                        label="shot-to-shot ±1σ",
                    )
            if analysis_ok is not None and analysis_ok.shape == finite.shape:
                rejected = finite & ~analysis_ok
                if np.any(rejected):
                    axis.plot(
                        x[rejected],
                        values[rejected],
                        "x",
                        color="tab:orange",
                        label="no accepted fits",
                    )
        else:
            axis.text(
                0.5,
                0.5,
                "No accepted values",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="0.35",
            )

        axis.set_title(title)
        axis.set_xlabel("X (cm)")
        axis.set_ylabel(y_label)
        if finite_x.size:
            axis.set_xlim(np.min(finite_x), np.max(finite_x))
        axis.grid(True)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend()

    if "te_fit_r2" in data and "te_min_r2" in data:
        min_r2 = float(np.asarray(data["te_min_r2"]))
        poor = (
            np.isfinite(data["te_fit_r2"])
            & np.isfinite(data["te_eV"])
            & (data["te_fit_r2"] < min_r2)
        )
        if np.any(poor):
            axes[0, 0].plot(
                x[poor],
                data["te_eV"][poor],
                "rx",
                ms=6,
                mew=1.5,
                label=f"Te fit R^2 < {min_r2:.2f}",
            )
            axes[0, 0].legend()

    if "vp_raw_V" in data:
        vp_raw = np.asarray(data["vp_raw_V"], dtype=float)
        vp_raw_finite = np.isfinite(x) & np.isfinite(vp_raw)
        if np.any(vp_raw_finite):
            axes[0, 1].plot(
                x[vp_raw_finite],
                vp_raw[vp_raw_finite],
                alpha=0.35,
                label="Vp raw",
            )
        if "vp_spike_mask" in data:
            spike_mask = np.asarray(data["vp_spike_mask"], dtype=bool)
            if spike_mask.shape == vp_raw.shape:
                spikes = spike_mask & vp_raw_finite
                if np.any(spikes):
                    axes[0, 1].plot(
                        x[spikes],
                        vp_raw[spikes],
                        "o",
                        label="flagged spikes",
                    )
        handles, _ = axes[0, 1].get_legend_handles_labels()
        if handles:
            axes[0, 1].legend()

    shape_factor_m = _shape_factor_m(data)
    shape_factor_text = (
        f"Shape factor = {shape_factor_m:.4g} m"
        if np.isfinite(shape_factor_m)
        else "Shape factor unavailable"
    )
    axes[2, 1].text(
        0.03,
        0.95,
        shape_factor_text,
        transform=axes[2, 1].transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )

    source_file = _scalar_text(data.get("source_file", "")).strip()
    title = Path(source_file).name if source_file else "Langmuir X-Line NPZ Results"
    fig.suptitle(title)
    return fig


def plot_xline_multi_ramp_summary(data):
    """Plot stored multi-ramp x-line profiles or x/ramp-time maps."""
    import matplotlib.pyplot as plt

    x = np.asarray(data["x_cm"], dtype=float)
    nramps = np.asarray(data["te_eV"]).shape[1]
    ramp_times = np.asarray(
        data.get("ramp_center_time_s", np.arange(nramps)),
        dtype=float,
    )
    display_mode = _scalar_text(
        data.get("ramp_display_mode", "separate_profiles")
    )
    panels = [
        ((0, 0), "te_eV", "te_std_eV", "Electron Temperature", "T_e (eV)"),
        ((0, 1), "vp_V", "vp_std_V", "Plasma Potential", "V_p (V)"),
        ((1, 0), "vf_V", "vf_std_V", "Floating Potential", "V_f (V)"),
        ((1, 1), "ies_A", "ies_std_A", "Electron Saturation Current", "I_es (A)"),
        ((2, 0), "iis_A", "iis_std_A", "Ion Saturation Current", "I_is (A)"),
        ((2, 1), "n_e_m3", "n_e_std_m3", "Electron Density", "n_e (m^-3)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, nramps))
    for (row, column), key, std_key, title, value_label in panels:
        axis = axes[row, column]
        if key not in data:
            axis.axis("off")
            continue
        values = np.asarray(data[key], dtype=float)
        if display_mode == "ramp_time_map":
            image = axis.pcolormesh(
                x,
                ramp_times,
                values.T,
                shading="auto",
            )
            fig.colorbar(image, ax=axis, label=value_label)
            axis.set_ylabel("Ramp center time (s)")
        else:
            std = (
                np.asarray(data[std_key], dtype=float)
                if std_key in data
                else None
            )
            for ramp_index, color in enumerate(colors):
                finite = np.isfinite(x) & np.isfinite(values[:, ramp_index])
                if not np.any(finite):
                    continue
                axis.plot(
                    x[finite],
                    values[finite, ramp_index],
                    color=color,
                    label=(
                        f"ramp {ramp_index + 1}, "
                        f"t={ramp_times[ramp_index]:.4g} s"
                    ),
                )
                if std is not None:
                    finite_std = finite & np.isfinite(std[:, ramp_index])
                    if np.any(finite_std):
                        axis.errorbar(
                            x[finite_std],
                            values[finite_std, ramp_index],
                            yerr=std[finite_std, ramp_index],
                            fmt="none",
                            ecolor=color,
                            alpha=0.55,
                            capsize=2,
                        )
            axis.set_ylabel(value_label)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize="small")
        axis.set_title(title)
        axis.set_xlabel("X (cm)")
        axis.grid(True)

    source_file = _scalar_text(data.get("source_file", "")).strip()
    title = Path(source_file).name if source_file else "Langmuir X-Line Results"
    fig.suptitle(title)
    return fig


def plot_xy_summary(data):
    """Make the six-panel spatial-map summary for xy-plane results."""
    import matplotlib.pyplot as plt

    x_mesh = np.asarray(data["X_cm"], dtype=float)
    y_mesh = np.asarray(data["Y_cm"], dtype=float)
    panels = [
        ((0, 0), "te_eV", "Electron Temperature", "T_e (eV)"),
        ((0, 1), "vp_V", "Plasma Potential", "V_p (V)"),
        ((1, 0), "vf_V", "Floating Potential", "V_f (V)"),
        ((1, 1), "ies_A", "Electron Saturation Current", "I_es (A)"),
        ((2, 0), "iis_A", "Ion Saturation Current", "I_is (A)"),
        ((2, 1), "n_e_m3", "Electron Density", "n_e (m^-3)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    analysis_ok = (
        np.asarray(data["analysis_ok"], dtype=bool)
        if "analysis_ok" in data
        else None
    )

    for (row, column), key, title, colorbar_label in panels:
        axis = axes[row, column]
        if key not in data:
            axis.axis("off")
            continue

        values = np.asarray(data[key], dtype=float)
        finite = np.isfinite(values)
        if np.any(finite):
            mesh = axis.pcolormesh(x_mesh, y_mesh, values, shading="auto")
            colorbar = fig.colorbar(mesh, ax=axis)
            colorbar.set_label(colorbar_label)
        else:
            finite_x = x_mesh[np.isfinite(x_mesh)]
            finite_y = y_mesh[np.isfinite(y_mesh)]
            if finite_x.size:
                axis.set_xlim(np.min(finite_x), np.max(finite_x))
            if finite_y.size:
                axis.set_ylim(np.min(finite_y), np.max(finite_y))
            axis.text(
                0.5,
                0.5,
                "No accepted values",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="0.35",
            )

        if analysis_ok is not None and analysis_ok.shape == values.shape:
            rejected = finite & ~analysis_ok
            if np.any(rejected):
                axis.plot(
                    x_mesh[rejected],
                    y_mesh[rejected],
                    "x",
                    color="tab:orange",
                    label="no accepted fits",
                )
                axis.legend()

        axis.set_title(title)
        axis.set_xlabel("X (cm)")
        axis.set_ylabel("Y (cm)")
        axis.set_aspect("equal", adjustable="box")

    if "te_fit_r2" in data and "te_min_r2" in data:
        min_r2 = float(np.asarray(data["te_min_r2"]))
        poor = np.isfinite(data["te_fit_r2"]) & (data["te_fit_r2"] < min_r2)
        if np.any(poor):
            axes[0, 0].plot(
                x_mesh[poor],
                y_mesh[poor],
                "rx",
                ms=5,
                mew=1.4,
                label=f"Te fit R^2 < {min_r2:.2f}",
            )
            axes[0, 0].legend()

    if "vp_spike_mask" in data:
        spike_mask = np.asarray(data["vp_spike_mask"], dtype=bool)
        if np.any(spike_mask):
            axes[0, 1].plot(
                x_mesh[spike_mask],
                y_mesh[spike_mask],
                "ko",
                ms=3,
                label="flagged Vp spikes",
            )
            axes[0, 1].legend()

    if "interferometer_profile_y_selected_cm" in data:
        selected_y = float(np.asarray(data["interferometer_profile_y_selected_cm"]))
        axes[2, 1].axhline(
            selected_y,
            color="white",
            linestyle="--",
            linewidth=1.2,
            label=f"interferometer x-line: y={selected_y:g} cm",
        )
        axes[2, 1].legend()

    shape_factor_m = _shape_factor_m(data)
    if np.isfinite(shape_factor_m):
        axes[2, 1].text(
            0.03,
            0.97,
            f"X-line shape factor = {shape_factor_m:.4g} m",
            transform=axes[2, 1].transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )

    source_file = _scalar_text(data.get("source_file", "")).strip()
    title = Path(source_file).name if source_file else "Langmuir XY-Plane NPZ Results"
    fig.suptitle(title)
    return fig


def plot_summary(data):
    """Dispatch to the summary plot matching the stored result geometry."""
    geometry = get_langmuir_result_geometry(data)
    if geometry == "x_line":
        return plot_xline_summary(data)
    return plot_xy_summary(data)


def main(argv=None, expected_geometry=None):
    """Run the combined command-line reader."""
    parser = argparse.ArgumentParser(
        description=(
            "Read and optionally plot x-line or xy-plane Langmuir NPZ results. "
            "The geometry is detected from the file."
        ),
        epilog=(
            "Examples:\n"
            "  python read_langmuir_results_npz.py results.npz\n"
            "  python read_langmuir_results_npz.py results.npz --plot"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("npz_path", help="Path to a Langmuir result NPZ file.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show the geometry-appropriate summary plot.",
    )
    args = parser.parse_args(argv)

    data = load_langmuir_results_npz(
        args.npz_path,
        expected_geometry=expected_geometry,
    )
    print_summary(data)

    if args.plot:
        import matplotlib.pyplot as plt

        plot_summary(data)
        plt.show()
    return data


if __name__ == "__main__":
    main()
