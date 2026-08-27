"""Plotting helpers for results produced by :mod:`langmuir_analysis_core`."""

import numpy as np


def build_iv_diagnostic_summary(result):
    """Return concise scalar summary lines for one analyzed trace."""
    values = {
        "Te": (result.get("te_eV", np.nan), "eV", ".4g"),
        "Vp": (result.get("vp_V", np.nan), "V", ".4g"),
        "Vf": (result.get("vf_V", np.nan), "V", ".4g"),
        "I_es": (result.get("ies_A", np.nan), "A", ".3g"),
        "I_is": (result.get("iis_A", np.nan), "A", ".3g"),
        "n_e": (result.get("n_e_m3", np.nan), "m^-3", ".3g"),
    }
    summary = []
    for label, (value, unit, precision) in values.items():
        if np.isfinite(value):
            summary.append(f"{label} = {value:{precision}} {unit}")
        else:
            summary.append(f"{label} = NaN")

    r2 = result.get("te_fit_r2", np.nan)
    rel_uncertainty = result.get("te_fit_relative_uncertainty", np.nan)
    summary.extend(
        [
            f"fit valid = {'yes' if result.get('te_fit_valid', False) else 'no'}",
            f"R^2 = {r2:.4f}" if np.isfinite(r2) else "R^2 = NaN",
            (
                f"slope rel. uncertainty = {rel_uncertainty:.3g}"
                if np.isfinite(rel_uncertainty)
                else "slope rel. uncertainty = NaN"
            ),
            f"independent N = {int(result.get('te_fit_npts', 0))}",
        ]
    )
    vstart = result.get("te_fit_vstart", np.nan)
    vstop = result.get("te_fit_vstop", np.nan)
    if np.isfinite(vstart) and np.isfinite(vstop):
        summary.append(f"V fit = {vstart:.3g} to {vstop:.3g} V")
    ion_current = result.get("ion_current_A", np.nan)
    ion_noise = result.get("ion_noise_A", np.nan)
    ion_uncertainty = result.get("ion_current_uncertainty_A", np.nan)
    ion_snr = result.get("ion_current_snr", np.nan)
    vp_derivative = result.get("vp_derivative_V", np.nan)
    if np.isfinite(vp_derivative):
        summary.append(f"Vp derivative seed = {vp_derivative:.4g} V")
    vp_fit = result.get("vp_fit_V", np.nan)
    if np.isfinite(vp_fit):
        summary.append(f"Vp fit extrapolation = {vp_fit:.4g} V")
    if np.isfinite(ion_current):
        summary.append(f"I_ion = {ion_current:.3g} A")
    if np.isfinite(ion_noise):
        summary.append(f"ion residual MAD sigma = {ion_noise:.3g} A")
    if np.isfinite(ion_uncertainty):
        summary.append(f"I_ion uncertainty = {ion_uncertainty:.3g} A")
    if np.isfinite(ion_snr):
        summary.append(f"I_ion estimate SNR = {ion_snr:.3g}")
    model_notes = result.get("model_notes", [])
    if model_notes:
        summary.append(f"ideal-model deviations = {len(model_notes)}")
    return summary


def plot_iv_diagnostic_panels(result, axs):
    """Plot measured I-V data, its derivative, and the Maxwellian fit."""
    diagnostic_data = result.get("diagnostic_data")
    if diagnostic_data is None:
        raise ValueError("result does not include diagnostic_data")

    Vg = result["iv_voltage_grid"]
    Ig = result["iv_current_grid"]
    dIdV = result["iv_didv_grid"]
    fit_mask = np.asarray(result["iv_fit_mask"], dtype=bool)
    vf = result.get("vf_V", np.nan)
    vp = result.get("vp_V", np.nan)
    vp_derivative = result.get("vp_derivative_V", np.nan)

    axs[0].plot(
        diagnostic_data["V_raw"],
        diagnostic_data["I_raw"],
        alpha=0.30,
        label="raw time-ordered",
    )
    axs[0].plot(
        diagnostic_data["V_binned"],
        diagnostic_data["I_binned"],
        ".",
        ms=3,
        label="measured-bin means",
    )
    axs[0].plot(Vg, Ig, lw=1.5, label="display interpolation")
    if np.any(fit_mask):
        axs[0].plot(Vg[fit_mask], Ig[fit_mask], lw=3, label="Te fit range")
    _plot_potential_lines(axs[0], vf, vp, vp_derivative)
    axs[0].set_xlabel("Bias (V)")
    axs[0].set_ylabel("Current (A)")
    axs[0].grid(True)
    _show_legend(axs[0])

    if dIdV is not None:
        axs[1].plot(Vg, dIdV, lw=1.5)
    _plot_potential_lines(axs[1], vf, vp, vp_derivative)
    axs[1].set_xlabel("Bias (V)")
    axs[1].set_ylabel("dI/dV (A/V)")
    axs[1].grid(True)
    _show_legend(axs[1])

    ion_current = result.get("ion_current_A", np.nan)
    if np.isfinite(ion_current):
        electron_current = Ig - ion_current
        log_good = np.isfinite(electron_current) & (electron_current > 0)
        axs[2].plot(
            Vg[log_good],
            np.log(electron_current[log_good]),
            lw=1.5,
            label="log(Ie)",
        )
        fit_good = log_good & fit_mask
        if np.any(fit_good):
            axs[2].plot(
                Vg[fit_good],
                np.log(electron_current[fit_good]),
                ".",
                ms=5,
                label="fit range",
            )
        slope = result.get("te_fit_slope", np.nan)
        intercept = result.get("te_fit_intercept", np.nan)
        if np.any(fit_mask) and np.isfinite(slope) and np.isfinite(intercept):
            axs[2].plot(
                Vg[fit_mask],
                slope * Vg[fit_mask] + intercept,
                lw=2.5,
                label="single Maxwellian fit",
            )
    else:
        axs[2].text(
            0.5,
            0.5,
            "No valid ion-current estimate",
            transform=axs[2].transAxes,
            ha="center",
            va="center",
        )

    _plot_potential_lines(axs[2], vf, vp, vp_derivative)
    axs[2].text(
        0.02,
        0.98,
        "\n".join(build_iv_diagnostic_summary(result)),
        transform=axs[2].transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    axs[2].set_xlabel("Bias (V)")
    axs[2].set_ylabel("log electron current")
    axs[2].grid(True)
    _show_legend(axs[2])


def _plot_potential_lines(ax, vf, vp, vp_derivative):
    if np.isfinite(vf):
        ax.axvline(vf, ls="--", label="Vf")
    if np.isfinite(vp):
        ax.axvline(vp, ls="--", label="Vp")
    if np.isfinite(vp_derivative) and (
        not np.isfinite(vp) or not np.isclose(vp_derivative, vp)
    ):
        ax.axvline(vp_derivative, color="0.4", ls=":", label="Vp dI/dV seed")


def _show_legend(ax):
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def render_iv_diagnostic_plot(result, title):
    """Render the standard three-panel diagnostic figure."""
    if result.get("diagnostic_data") is None:
        raise ValueError(
            "result does not include diagnostic_data; rerun analyze_iv_trace "
            "with include_diagnostic_data=True"
        )

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(8, 11), constrained_layout=True)
    plot_iv_diagnostic_panels(result, axs)
    fig.suptitle(title)
    return fig
