"""Read and optionally plot Langmuir XY-plane NPZ output files.

Invocation examples:

    python read_langmuir_xy_npz.py path/to/run_langmuir_xy.npz
    python read_langmuir_xy_npz.py path/to/run_langmuir_xy.npz --plot

The first form prints a compact inventory, acquisition-polarity settings, and
Te-fit quality counts. The ``--plot`` form opens a six-panel summary matching
the plot produced by ``langmuir_xy_analysis.py``.

Useful exported arrays:

    source_file, geometry
        Input filename and geometry marker.
    scale_to_interferometer_density, interferometer_line_averaged_density_m3
        Interferometer density scaling settings used by the analysis.
    x_cm, y_cm
        One-dimensional coordinate axes.
    X_cm, Y_cm
        Two-dimensional spatial coordinate meshes with the same shape as the
        XY maps.
    time_s
        Time axis for the XY-plane sequence.

    te_eV, vp_V, vf_V, ies_A, iis_A, isat_A, n_e_m3
        Primary exported maps reflecting the processing choices in
        ``langmuir_xy_analysis.py``. Here ``ies`` is electron saturation current and
        ``iis`` is ion saturation current.
    te_raw_eV, vp_raw_V, vf_raw_V, ies_raw_A, iis_raw_A, n_e_raw_m3
        Raw extracted maps before optional neighbor smoothing or spike cleanup.
    te_processed_eV, vp_processed_V, vf_processed_V, ies_processed_A,
    iis_processed_A, n_e_processed_m3
        Post-processed maps.
    te_std_eV, vp_std_V, vf_std_V, ies_std_A, iis_std_A, n_e_std_m3
        Shot-to-shot sample standard deviations for the primary maps.
    vp_spike_mask
        Binary spike rejection mask for plasma potential.

    vsweep_mean_V, isweep_mean_A
        Mean voltage/current traces for the XY-plane dataset.
    vsweep_mean_smoothed_V, isweep_mean_smoothed_A
        Smoothed mean traces.
    isweep_dc_offsets_A
        DC offsets applied to each sweep when ``subtract_dc`` was enabled.

    iv_voltage_grid_V, iv_current_grid_A, iv_didv_grid_A_per_V, iv_te_fit_mask
        Per-location interpolated I-V curves and Te fit masks.

    te_fit_r2, te_fit_rmse_logI, te_fit_npts,
    te_fit_vstart_V, te_fit_vstop_V, te_fit_slope_logI_per_V,
    te_fit_intercept_logI, te_fit_i0_A, te_subtract_i0,
    te_fit_passed_r2, te_fit_candidate_count
        Te fit diagnostics and selection metadata.
        ``te_fit_i0_A`` is retained as a legacy key but now contains the
        measured constant ion-saturation plateau subtracted from total current.
        It is no longer an artificial positivity offset.
    analysis_ok, analysis_valid_count, analysis_ok_shot
        Overall acceptance masks and accepted-shot counts. Finite estimates
        are retained for inspection while these arrays record fit acceptance.
    current_zero_calibrated, negate_Isweep_current, subtract_dc
        Current calibration, acquisition-polarity, and offset-subtraction
        settings used to produce the exported maps.

    xy_shape_info, trace_spatial_shape, dt_s, te_min_r2
        Geometry and diagnostic metadata.

    summary_plot_path, all_iv_plot_path
        Saved file paths for diagnostic plots.
    summary_plot_png, all_iv_curves_plot_png
        Optional PNG bytes when plot generation was enabled.

Maps use ``(ny, nx)`` indexing; scientific per-shot arrays use
``(ny, nx, nshots)``. Representative display I-V arrays use
``(ny, nx, iv_npts)``, while their ``*_shot`` counterparts use
``(ny, nx, nshots, iv_npts)``.

From another Python script:

    from read_langmuir_xy_npz import load_langmuir_xy_npz

    data = load_langmuir_xy_npz("path/to/run_langmuir_xy.npz")
    X = data["X_cm"]
    Y = data["Y_cm"]
    te = data["te_eV"]
    vp = data["vp_V"]
    vf = data["vf_V"]
    # etc.
"""

import argparse
from pathlib import Path

import numpy as np

EXPECTED_LANGMUIR_XY_NPZ_KEYS = {
    "source_file",
    "geometry",
    "x_cm",
    "y_cm",
    "X_cm",
    "Y_cm",
    "time_s",
    "te_eV",
    "vp_V",
    "vf_V",
    "ies_A",
    "iis_A",
    "isat_A",
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
    "xy_shape_info",
    "n_e_m3",
    "trace_spatial_shape",
    "dt_s",
    "te_min_r2",
    "summary_plot_path",
    "all_iv_plot_path",
}


def validate_langmuir_xy_npz(data):
    """Require the stable schema while allowing forward-compatible extras."""
    keys = set(data.keys())
    missing = EXPECTED_LANGMUIR_XY_NPZ_KEYS - keys
    if missing:
        raise ValueError(
            "Loaded NPZ file does not conform to expected Langmuir XY-plane schema: "
            f"missing keys: {sorted(missing)}"
        )
    return data


def load_langmuir_xy_npz(path):
    """Load a Langmuir XY-plane .npz file into an in-memory dictionary."""
    with np.load(path, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    return validate_langmuir_xy_npz(data)


def _scalar_text(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr)


def print_summary(data):
    """Print a compact summary of a loaded Langmuir XY-plane NPZ result."""
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
    """Make a summary plot matching langmuir_xy_analysis.render_summary_plot."""
    import matplotlib.pyplot as plt

    X = data["X_cm"]
    Y = data["Y_cm"]
    panels = [
        ((0, 0), "te_eV", "Electron Temperature", "T_e (eV)"),
        ((0, 1), "vp_V", "Plasma Potential", "V_p (V)"),
        ((1, 0), "vf_V", "Floating Potential", "V_f (V)"),
        ((1, 1), "ies_A", "Electron Saturation Current", "I_es (A)"),
        ((2, 0), "iis_A", "Ion Saturation Current", "I_is (A)"),
        ((2, 1), "n_e_m3", "Electron Density", "n_e (m^-3)"),
    ]

    fig, axs = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    analysis_ok = (
        np.asarray(data["analysis_ok"], dtype=bool)
        if "analysis_ok" in data
        else None
    )

    for (row, col), key, title, label in panels:
        ax = axs[row, col]
        if key not in data:
            ax.axis("off")
            continue

        values = np.asarray(data[key], dtype=float)
        finite = np.isfinite(values)
        if np.any(finite):
            mesh = ax.pcolormesh(X, Y, values, shading="auto")
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label(label)
        else:
            finite_x = np.asarray(X)[np.isfinite(X)]
            finite_y = np.asarray(Y)[np.isfinite(Y)]
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

        if analysis_ok is not None and analysis_ok.shape == values.shape:
            rejected = finite & ~analysis_ok
            if np.any(rejected):
                ax.plot(
                    X[rejected],
                    Y[rejected],
                    "x",
                    color="tab:orange",
                    label="no accepted fits",
                )
                ax.legend()

        ax.set_title(title)
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_aspect("equal", adjustable="box")

    if "te_fit_r2" in data and "te_min_r2" in data:
        min_r2 = float(np.asarray(data["te_min_r2"]))
        poor = np.isfinite(data["te_fit_r2"]) & (data["te_fit_r2"] < min_r2)
        if np.any(poor):
            ax = axs[0, 0]
            ax.plot(
                X[poor],
                Y[poor],
                "rx",
                ms=5,
                mew=1.4,
                label=f"Te fit R^2 < {min_r2:.2f}",
            )
            ax.legend()

    if "vp_spike_mask" in data:
        vp_spike_mask = np.asarray(data["vp_spike_mask"]).astype(bool)
        if np.any(vp_spike_mask):
            axs[0, 1].plot(
                X[vp_spike_mask],
                Y[vp_spike_mask],
                "ko",
                ms=3,
                label="flagged Vp spikes",
            )
            axs[0, 1].legend()

    source_file = _scalar_text(data.get("source_file", "")).strip()
    title = Path(source_file).name if source_file else "Langmuir XY-Plane NPZ Results"
    fig.suptitle(title)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Read and optionally plot Langmuir XY-plane NPZ output files.",
        epilog=(
            "Examples:\n"
            "  python read_langmuir_xy_npz.py path/to/run_langmuir_xy.npz\n"
            "  python read_langmuir_xy_npz.py path/to/run_langmuir_xy.npz --plot"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "npz_path",
        nargs="?",
        default=None,
        help="Path to a *_langmuir_xy.npz file.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show a quick summary plot."
    )
    args, extra = parser.parse_known_args()
    if args.npz_path is None:
        parser.print_help()
        return

    data = load_langmuir_xy_npz(args.npz_path)
    print_summary(data)

    if args.plot:
        import matplotlib.pyplot as plt

        plot_summary(data)
        plt.show()


if __name__ == "__main__":
    main()
