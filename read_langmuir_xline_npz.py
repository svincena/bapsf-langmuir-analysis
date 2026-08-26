"""Read and optionally plot Langmuir x-line NPZ output files.

Invocation examples:

    python read_langmuir_xline_npz.py path/to/run_langmuir_xline.npz
    python read_langmuir_xline_npz.py path/to/run_langmuir_xline.npz --plot

The first form prints a compact inventory, acquisition-polarity settings, and
Te-fit quality counts. The ``--plot`` form opens a six-panel summary matching
the plot produced by ``langmuir_xline_analysis.py``.

Useful exported arrays:

    source_file, geometry
        Input filename and geometry marker.
    scale_to_interferometer_density, interferometer_line_averaged_density_m3
        Optional interferometer density scaling settings used by the analysis.
    x_cm
        One-dimensional x coordinate.
    X_cm, Y_cm
        Line-coordinate arrays. ``Y_cm`` is normally all zeros for the x-line.
    te_eV, vp_V, vf_V, ies_A, iis_A, n_e_m3, isat_A
        Primary exported profiles. These reflect the current processing choices
        in ``langmuir_xline_analysis.py``.
    te_raw_eV, vp_raw_V, vf_raw_V, ies_raw_A, iis_raw_A, n_e_raw_m3
        Raw extracted profiles before optional smoothing or spike cleanup.
    te_processed_eV, vp_processed_V, vf_processed_V, ies_processed_A,
    iis_processed_A, n_e_processed_m3
        Post-processed profiles.
    te_std_eV, vp_std_V, vf_std_V, ies_std_A, iis_std_A, n_e_std_m3
        Shot-to-shot sample standard deviations used as summary error bars.
    te_fit_r2, te_fit_rmse_logI, te_fit_npts, te_fit_vstart_V, te_fit_vstop_V
        Te fit quality and fit-window diagnostics.
    te_fit_i0_A
        Legacy key containing the measured constant ion-saturation plateau
        subtracted before the Maxwellian fit.
    te_fit_passed_r2
        Integer mask where 1 means the Te fit met ``te_min_r2``.
    analysis_ok, analysis_valid_count, analysis_ok_shot
        Overall acceptance masks and accepted-shot counts. Finite estimates
        are retained for inspection while these arrays record fit acceptance.
    current_zero_calibrated, negate_Isweep_current, subtract_dc
        Current calibration, acquisition-polarity, and offset-subtraction
        settings used to produce the exported profiles.
    iv_voltage_grid_V, iv_current_grid_A, iv_didv_grid_A_per_V, iv_te_fit_mask
        Per-location interpolated I-V curves and Te fit masks.
    summary_plot_png, all_iv_curves_plot_png
        Optional PNG bytes, present when plot generation was enabled.

Profiles use ``(nx,)`` indexing; scientific per-shot arrays use
``(nx, nshots)``. Representative display I-V arrays use ``(nx, iv_npts)``,
while their ``*_shot`` counterparts use ``(nx, nshots, iv_npts)``.

From another Python script:

    from read_langmuir_xline_npz import load_langmuir_xline_npz

    data = load_langmuir_xline_npz("path/to/run_langmuir_xline.npz")
    x = data["x_cm"]
    te = data["te_eV"]
    vp = data["vp_V"]
    vf = data["vf_V"]
"""

import argparse
from pathlib import Path

import numpy as np


def load_langmuir_xline_npz(path):
    """Load a Langmuir x-line .npz file into an in-memory dictionary."""
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def _scalar_text(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr)


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

    # Convert the coordinate samples to meters before trapezoidal integration.
    x_values_m = x_values_cm[finite] * 1.0e-2
    normalized_profile = finite_profile / profile_maximum
    return float(
        np.sum(
            np.diff(x_values_m)
            * 0.5
            * (normalized_profile[:-1] + normalized_profile[1:])
        )
    )


def print_summary(data):
    """Print a compact summary of a loaded Langmuir x-line NPZ result."""
    print(f"source_file: {_scalar_text(data.get('source_file', ''))}")
    print(f"geometry: {_scalar_text(data.get('geometry', ''))}")
    if "scale_to_interferometer_density" in data:
        print(
            "scale_to_interferometer_density: "
            f"{_scalar_text(data['scale_to_interferometer_density'])}"
        )
    if "interferometer_line_averaged_density_m3" in data:
        print(
            "interferometer_line_averaged_density_m3: "
            f"{_scalar_text(data['interferometer_line_averaged_density_m3'])}"
        )

    for key in (
        "current_zero_calibrated",
        "negate_Isweep_current",
        "subtract_dc",
    ):
        if key in data:
            print(f"{key}: {_scalar_text(data[key])}")

    for key in (
        "x_cm",
        "X_cm",
        "Y_cm",
        "te_eV",
        "vp_V",
        "vf_V",
        "ies_A",
        "iis_A",
        "n_e_m3",
        "te_fit_r2",
    ):
        if key in data:
            arr = data[key]
            print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

    if "x_cm" in data and "n_e_m3" in data:
        shape_factor_m = electron_density_profile_shape_factor_m(
            data["x_cm"], data["n_e_m3"]
        )
        if np.isfinite(shape_factor_m):
            print(f"electron-density shape factor: {shape_factor_m:.4g} m")
        else:
            print("electron-density shape factor: unavailable")

    if "analysis_valid_count" in data:
        print(
            "accepted shot analyses: "
            f"{int(np.sum(np.asarray(data['analysis_valid_count'])))}"
        )
    elif "te_raw_eV" in data:
        print(
            f"accepted spatial results: {np.count_nonzero(np.isfinite(data['te_raw_eV']))}"
        )

    if "te_fit_r2" in data:
        min_r2 = float(np.asarray(data.get("te_min_r2", 0.90)))
        good = np.isfinite(data["te_fit_r2"])
        poor = good & (data["te_fit_r2"] < min_r2)
        print(f"finite Te fit diagnostics: {np.count_nonzero(good)}")
        print(f"Te fits below R^2 {min_r2:.2f}: {np.count_nonzero(poor)}")


def plot_summary(data):
    """Make a summary plot matching langmuir_xline_analysis.render_summary_plot."""
    import matplotlib.pyplot as plt

    x = np.asarray(data["x_cm"], dtype=float)
    panels = [
        ((0, 0), "te_eV", "te_std_eV", "Electron Temperature", "T_e (eV)"),
        ((0, 1), "vp_V", "vp_std_V", "Plasma Potential", "V_p (V)"),
        ((1, 0), "vf_V", "vf_std_V", "Floating Potential", "V_f (V)"),
        (
            (1, 1),
            "ies_A",
            "ies_std_A",
            "Electron Saturation Current",
            "I_es (A)",
        ),
        ((2, 0), "iis_A", "iis_std_A", "Ion Saturation Current", "I_is (A)"),
        ((2, 1), "n_e_m3", "n_e_std_m3", "Electron Density", "n_e (m^-3)"),
    ]

    fig, axs = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    finite_x = x[np.isfinite(x)]
    analysis_ok = (
        np.asarray(data["analysis_ok"], dtype=bool)
        if "analysis_ok" in data
        else None
    )

    for (row, col), key, std_key, title, ylabel in panels:
        ax = axs[row, col]
        if key not in data:
            ax.axis("off")
            continue

        values = np.asarray(data[key], dtype=float)
        finite = np.isfinite(x) & np.isfinite(values)
        if np.any(finite):
            line = ax.plot(x[finite], values[finite])[0]
            if std_key in data:
                std = np.asarray(data[std_key], dtype=float)
                finite_std = finite & np.isfinite(std)
                if np.any(finite_std):
                    ax.errorbar(
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
                    ax.plot(
                        x[rejected],
                        values[rejected],
                        "x",
                        color="tab:orange",
                        label="no accepted fits",
                    )
        else:
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
        ax.set_ylabel(ylabel)
        if finite_x.size:
            ax.set_xlim(np.min(finite_x), np.max(finite_x))
        ax.grid(True)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    if "te_fit_r2" in data and "te_min_r2" in data:
        min_r2 = float(np.asarray(data["te_min_r2"]))
        poor = (
            np.isfinite(data["te_fit_r2"])
            & np.isfinite(data["te_eV"])
            & (data["te_fit_r2"] < min_r2)
        )
        if np.any(poor):
            axs[0, 0].plot(
                x[poor],
                data["te_eV"][poor],
                "rx",
                ms=6,
                mew=1.5,
                label=f"Te fit R^2 < {min_r2:.2f}",
            )
            axs[0, 0].legend()

    if "vp_raw_V" in data:
        vp_raw = np.asarray(data["vp_raw_V"], dtype=float)
        vp_raw_finite = np.isfinite(x) & np.isfinite(vp_raw)
        if np.any(vp_raw_finite):
            axs[0, 1].plot(
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
                    axs[0, 1].plot(
                        x[spikes],
                        vp_raw[spikes],
                        "o",
                        label="flagged spikes",
                    )
        handles, _ = axs[0, 1].get_legend_handles_labels()
        if handles:
            axs[0, 1].legend()

    if "n_e_m3" in data:
        shape_factor_m = electron_density_profile_shape_factor_m(
            x, data["n_e_m3"]
        )
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

    source_file = _scalar_text(data.get("source_file", "")).strip()
    title = Path(source_file).name if source_file else "Langmuir X-Line NPZ Results"
    fig.suptitle(title)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Read and optionally plot Langmuir x-line NPZ output files.",
        epilog=(
            "Examples:\n"
            "  python read_langmuir_xline_npz.py path/to/run_langmuir_xline.npz\n"
            "  python read_langmuir_xline_npz.py path/to/run_langmuir_xline.npz --plot"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("npz_path", help="Path to a *_langmuir_xline.npz file.")
    parser.add_argument(
        "--plot", action="store_true", help="Show a quick summary plot."
    )
    args = parser.parse_args()

    data = load_langmuir_xline_npz(args.npz_path)
    print_summary(data)

    if args.plot:
        import matplotlib.pyplot as plt

        plot_summary(data)
        plt.show()


if __name__ == "__main__":
    main()
