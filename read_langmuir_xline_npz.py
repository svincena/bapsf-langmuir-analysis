"""Read and optionally plot Langmuir x-line NPZ output files.

Invocation examples:

    python read_langmuir_xline_npz.py path/to/run_langmuir_xline.npz
    python read_langmuir_xline_npz.py path/to/run_langmuir_xline.npz --plot

The first form prints a compact inventory and Te-fit quality counts. The
``--plot`` form also opens a quick four-panel summary plot with poor Te fits
marked on the Te trace.

Useful exported arrays:

    source_file, geometry
        Input filename and geometry marker.
    scale_to_interferometer_density, interferometer_line_averaged_density_m3
        Optional interferometer density scaling settings used by the analysis.
    x_cm
        One-dimensional x coordinate.
    X_cm, Y_cm
        Line-coordinate arrays. ``Y_cm`` is normally all zeros for the x-line.
    te_eV, vp_V, vf_V, ies_A, isat_A
        Primary exported profiles. These reflect the current processing choices
        in ``langmuir_xline_analysis_complete.py``.
    te_raw_eV, vp_raw_V, vf_raw_V
        Raw extracted profiles before optional smoothing or spike cleanup.
    te_processed_eV, vp_processed_V, vf_processed_V
        Post-processed profiles.
    te_fit_r2, te_fit_rmse_logI, te_fit_npts, te_fit_vstart_V, te_fit_vstop_V
        Te fit quality and fit-window diagnostics.
    te_fit_passed_r2
        Integer mask where 1 means the Te fit met ``te_min_r2``.
    iv_voltage_grid_V, iv_current_grid_A, iv_didv_grid_A_per_V, iv_te_fit_mask
        Per-location interpolated I-V curves and Te fit masks.
    summary_plot_png, all_iv_curves_plot_png
        Optional PNG bytes, present when plot generation was enabled.

All profile arrays use ``(nx,)`` indexing. Per-trace I-V arrays use
``(nx, iv_npts)`` indexing.

From another Python script:

    from read_langmuir_xline_npz import load_langmuir_xline_npz

    data = load_langmuir_xline_npz("path/to/run_langmuir_xline.npz")
    x = data["x_cm"]
    te = data["te_eV"]
    vp = data["vp_V"]
    vf = data["vf_V"]
"""

import argparse

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

    for key in ("x_cm", "X_cm", "Y_cm", "te_eV", "vp_V", "vf_V", "ies_A", "te_fit_r2"):
        if key in data:
            arr = data[key]
            print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

    if "te_fit_r2" in data:
        min_r2 = float(np.asarray(data.get("te_min_r2", 0.90)))
        good = np.isfinite(data["te_fit_r2"])
        poor = good & (data["te_fit_r2"] < min_r2)
        print(f"finite Te fits: {np.count_nonzero(good)}")
        print(f"Te fits below R^2 {min_r2:.2f}: {np.count_nonzero(poor)}")


def plot_summary(data):
    """Make a quick four-panel summary plot from loaded NPZ data."""
    import matplotlib.pyplot as plt

    x = data["x_cm"]
    panels = [
        ("te_eV", "Electron Temperature", "T_e (eV)"),
        ("vp_V", "Plasma Potential", "V_p (V)"),
        ("vf_V", "Floating Potential", "V_f (V)"),
        ("ies_A", "Electron Saturation Current", "I_es (A)"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axs.flat, panels):
        ax.plot(x, data[key])
        ax.set_title(title)
        ax.set_xlabel("X (cm)")
        ax.set_ylabel(ylabel)
        ax.grid(True)

    if "te_fit_r2" in data and "te_min_r2" in data:
        min_r2 = float(np.asarray(data["te_min_r2"]))
        poor = np.isfinite(data["te_fit_r2"]) & np.isfinite(data["te_eV"]) & (data["te_fit_r2"] < min_r2)
        if np.any(poor):
            axs[0, 0].plot(x[poor], data["te_eV"][poor], "rx", ms=6, mew=1.5, label=f"Te fit R^2 < {min_r2:.2f}")
            axs[0, 0].legend()

    fig.suptitle("Langmuir X-Line NPZ Results")
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
    parser.add_argument("--plot", action="store_true", help="Show a quick summary plot.")
    args = parser.parse_args()

    data = load_langmuir_xline_npz(args.npz_path)
    print_summary(data)

    if args.plot:
        import matplotlib.pyplot as plt

        plot_summary(data)
        plt.show()


if __name__ == "__main__":
    main()
