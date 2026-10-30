"""Compare the ordinary solver with an opt-in background-separated bulk fit.

Run .venv/bin/python synthetic_langmuir_beam_validation.py. This is a research
diagnostic; no GUI defaults or existing fit/current/density definitions change.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.constants import e, m_e
from scipy.special import erfc

from langmuir_analysis_core import (
    analyze_iv_trace,
    scan_bulk_temperature_windows,
)
from synthetic_langmuir_validation import SyntheticPlasma, make_trace


def drifting_electron_current(
    voltage, *, density_m3, drift_energy_eV, spread_eV, vp_V, area_m2
):
    """Integrate a shifted normal velocity distribution over incident flux.

    The drift is directed toward the face, normal to it and parallel to B.
    Spread is the one-dimensional thermal temperature, not the drift energy.
    At zero drift this reduces exactly to planar Maxwellian collection. This
    ideal collisionless flux model omits beam-induced sheath instabilities.
    """
    if density_m3 < 0 or drift_energy_eV < 0 or spread_eV <= 0 or area_m2 <= 0:
        raise ValueError(
            "Beam density/energy must be nonnegative; spread/area positive."
        )
    voltage = np.asarray(voltage, dtype=float)
    threshold_speed = np.sqrt(2 * e * np.maximum(vp_V - voltage, 0) / m_e)
    drift = np.sqrt(2 * e * drift_energy_eV / m_e)
    sigma = np.sqrt(e * spread_eV / m_e)
    z = (threshold_speed - drift) / sigma
    flux_speed = sigma / np.sqrt(2 * np.pi) * np.exp(-0.5 * z**2)
    flux_speed += 0.5 * drift * erfc(z / np.sqrt(2))
    return e * area_m2 * density_m3 * flux_speed


def make_beam_trace(
    *, fraction=0.01, drift_eV=35.0, spread_eV=2.0, noise_fraction=0.0, seed=20261029
):
    """Add a beam whose density is a stated fraction of a fixed 3 eV bulk."""
    p = SyntheticPlasma()
    voltage = np.linspace(-250, 31, 14051)
    _, _, reference = make_trace(p)
    bulk = reference["ies_A"] * np.exp(np.minimum((voltage - p.vp_V) / p.te_eV, 0))
    beam = drifting_electron_current(
        voltage,
        density_m3=fraction * p.ne_cm3 * 1e6,
        drift_energy_eV=drift_eV,
        spread_eV=spread_eV,
        vp_V=p.vp_V,
        area_m2=p.area_m2,
    )
    # Bulk-dominated Bohm closure, with extra ions to balance beam charge.
    # This is a controlled ion background, not a kinetic beam/presheath model.
    ion = (
        -(1 + fraction)
        * reference["iis_at_vp_A"]
        * np.exp(-np.maximum(voltage - p.vp_V, 0) / p.ti_eV)
    )
    noise = np.random.default_rng(seed).normal(
        0, noise_fraction * reference["ies_A"], voltage.size
    )
    return (
        voltage,
        bulk + beam + ion + noise,
        {"bulk_A": bulk, "beam_A": beam, "ion_A": ion},
    )


def scan_bulk_windows(result, *, temperature_bounds):
    """Use the shared background-window diagnostic; no production fit changes."""
    return scan_bulk_temperature_windows(result, temperature_bounds=temperature_bounds)


def run_study(output_dir=Path("synthetic_validation_output/beam")):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [("bulk_only", 0, 35, 2, 0)]
    cases += [
        (f"beam_{fraction:g}_{energy:g}", fraction, energy, 2, 0)
        for fraction in (0.001, 0.003, 0.01)
        for energy in (20, 35, 60)
    ]
    cases += [("warm_beam", 0.01, 35, 10, 0), ("hot_halo", 0.01, 0, 20, 0)]
    cases += [(f"noisy_beam_{noise:g}", 0.01, 35, 2, noise) for noise in (0.001, 0.01)]
    cases = [(*case, 20261029) for case in cases]
    cases += [
        (
            f"noise_{'beam' if fraction else 'control'}_seed_{seed}",
            fraction,
            35,
            2,
            0.01,
            seed,
        )
        for fraction in (0, 0.01)
        for seed in range(8)
    ]
    rows, windows, archive, examples = [], [], {}, {}
    for case_id, (label, fraction, energy, spread, noise, seed) in enumerate(cases):
        voltage, current, truth = make_beam_trace(
            fraction=fraction,
            drift_eV=energy,
            spread_eV=spread,
            noise_fraction=noise,
            seed=seed,
        )
        config = {"probe_area": 4e-6, "current_zero_calibrated": True}
        ordinary = analyze_iv_trace(
            voltage, current, config, include_diagnostic_data=True
        )
        bounded = analyze_iv_trace(
            voltage, current, {**config, "te_region_method": "bounded"}
        )
        scans, bulk = scan_bulk_windows(ordinary, temperature_bounds=(0.1, 300))
        excess = (
            np.nan
            if bulk is None
            else bulk["background_at_vmax_A"] - ordinary["ion_current_A"]
        )
        uncertainty = (
            np.nan
            if bulk is None
            else np.hypot(
                bulk["background_at_vmax_uncertainty_A"],
                ordinary["ion_current_uncertainty_A"],
            )
        )
        rows.append(
            {
                "case_id": case_id,
                "case": label,
                "beam_over_bulk_density": fraction,
                "beam_drift_eV": energy,
                "beam_spread_eV": spread,
                "noise_over_bulk_ies": noise,
                "seed": seed,
                "true_bulk_te_eV": 3.0,
                "ordinary_te_eV": ordinary["te_eV"],
                "ordinary_ok": ordinary["ok"],
                "ordinary_vstart_V": ordinary["te_fit_vstart"],
                "ordinary_vstop_V": ordinary["te_fit_vstop"],
                "ordinary_warnings": " | ".join(ordinary["warnings"]),
                "bounded_te_eV": bounded["te_eV"],
                "bounded_ok": bounded["ok"],
                "bulk_candidate_found": bulk is not None,
                "bulk_candidate_te_eV": np.nan if bulk is None else bulk["te_eV"],
                "bulk_candidate_width_V": np.nan if bulk is None else bulk["width_V"],
                "bulk_candidate_vstart_V": np.nan if bulk is None else bulk["vstart_V"],
                "bulk_candidate_vstop_V": np.nan if bulk is None else bulk["vstop_V"],
                "bulk_signal_vstart_V": np.nan
                if bulk is None
                else bulk["bulk_signal_vstart_V"],
                "bulk_signal_vstop_V": np.nan
                if bulk is None
                else bulk["bulk_signal_vstop_V"],
                "excess_background_A": excess,
                "excess_background_5sigma": bool(excess > 5 * uncertainty),
            }
        )
        for fit in scans:
            windows.append(
                {
                    "case_id": case_id,
                    "case": label,
                    **{key: value for key, value in fit.items() if np.isscalar(value)},
                }
            )
        for key, array in {"voltage_V": voltage, "current_A": current, **truth}.items():
            archive[f"case_{case_id:03d}_{key}"] = array
        examples[label] = (voltage, current, truth, ordinary, scans, bulk)
        print(
            label,
            "ordinary:",
            round(ordinary["te_eV"], 3),
            ordinary["ok"],
            "bulk:",
            None if bulk is None else round(bulk["te_eV"], 3),
            flush=True,
        )
    for name, data in (("recovery.csv", rows), ("windows.csv", windows)):
        with (output_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    np.savez_compressed(output_dir / "traces.npz", **archive)
    _plot(examples, output_dir)
    _report(rows, output_dir)
    return rows


def _plot(examples, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    voltage, current, truth, ordinary, _scans, bulk = examples["beam_0.01_35"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    axs[0, 0].plot(voltage, current, "k", label="Measured net current")
    axs[0, 0].plot(voltage, truth["bulk_A"], label="True 3 eV bulk")
    axs[0, 0].plot(voltage, truth["beam_A"], label="1% beam, drift 35 eV, spread 2 eV")
    axs[0, 0].set(
        xlim=(-80, 15),
        xlabel="Bias (V)",
        ylabel="Current (A)",
        title="Bulk + directed warm beam",
    )
    electron = current - ordinary["ion_current_A"]
    keep = electron > 1e-5
    axs[0, 1].semilogy(
        voltage[keep], electron[keep], "k", label="Total electron current"
    )
    axs[0, 1].semilogy(voltage, truth["bulk_A"], "--", label="True bulk")
    axs[0, 1].semilogy(voltage, truth["beam_A"], "--", label="True beam")
    axs[0, 1].axvspan(
        ordinary["te_fit_vstart"],
        ordinary["te_fit_vstop"],
        alpha=0.12,
        color="red",
        label="Ordinary fit window (rejected)",
    )
    if bulk is not None:
        axs[0, 1].axvspan(
            bulk["vstart_V"],
            bulk["vstop_V"],
            alpha=0.12,
            color="green",
            label="Background-separated candidate",
        )
    axs[0, 1].set(
        xlim=(-80, 12),
        ylim=(1e-5, 2),
        xlabel="Bias (V)",
        ylabel="Electron current (A)",
        title="Selecting a window alone leaves beam current",
    )
    for label, color in (
        ("bulk_only", "black"),
        ("beam_0.01_35", "#b7352b"),
        ("warm_beam", "#2d61a3"),
        ("hot_halo", "#22834a"),
    ):
        fits = examples[label][4]
        axs[1, 0].plot(
            [fit["width_V"] for fit in fits],
            [fit["te_eV"] for fit in fits],
            ".-",
            color=color,
            label=label,
        )
        rejected = [
            fit
            for fit in fits
            if not (fit["valid"] and fit["residual_pass"] and fit["bulk_log_pass"])
        ]
        axs[1, 0].scatter(
            [fit["width_V"] for fit in rejected],
            [fit["te_eV"] for fit in rejected],
            color=color,
            marker="x",
            s=65,
        )
    axs[1, 0].axhline(3, color="gray", ls=":", label="Injected bulk Te")
    axs[1, 0].set(
        xlabel="Window width below knee (V)",
        ylabel="Exponential Te (eV)",
        title="Window stability; crosses fail research checks",
        ylim=(0, 12),
    )
    data = ordinary["diagnostic_data"]
    if bulk is not None:
        vfit = data["V_binned"][bulk["mask"]]
        corrected = data["I_binned"][bulk["mask"]] - bulk["background_current_A"]
        positive = (
            (corrected > 0)
            & (vfit >= bulk["bulk_signal_vstart_V"])
            & (vfit <= bulk["bulk_signal_vstop_V"])
        )
        axs[1, 1].semilogy(
            vfit[positive],
            corrected[positive],
            ".",
            ms=3,
            label="Measured minus fitted background",
        )
        axs[1, 1].semilogy(
            vfit,
            bulk["exponential_current_A"],
            "k--",
            label=f"Recovered Te = {bulk['te_eV']:.3f} eV",
        )
        axs[1, 1].semilogy(voltage, truth["bulk_A"], ":", label="True bulk")
        axs[1, 1].set(xlim=(vfit[positive][0], vfit[positive][-1]), ylim=(0.01, 1))
    axs[1, 1].set(
        xlabel="Bias (V)",
        ylabel="Bulk electron current (A)",
        title="Separating the background recovers bulk slope",
    )
    for ax in axs.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "A low-density electron beam can bias a single-temperature Langmuir fit"
    )
    fig.savefig(output_dir / "beam_recovery.png", dpi=160)
    plt.close(fig)

    # For a planar retarding collector, dI/dV samples the incident normal
    # velocity distribution. A displaced maximum is useful beam evidence;
    # an isotropic Druyvesteyn EEDF inversion is not valid for this anisotropy.
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    vg = ordinary["iv_voltage_grid"]
    energy = ordinary["vp_derivative_V"] - vg
    derivative = ordinary["iv_didv_grid"]
    keep = (energy > 0) & (derivative > 0)
    ax.semilogy(energy[keep], derivative[keep], "k", label="Smoothed measured dI/dV")
    energy_true = 10 - voltage
    retarding = energy_true >= 0
    ax.semilogy(
        energy_true[retarding],
        truth["bulk_A"][retarding] / 3,
        "--",
        label="Injected bulk derivative",
    )
    beam_derivative = np.gradient(truth["beam_A"], voltage)
    ax.semilogy(
        energy_true[retarding],
        beam_derivative[retarding],
        "--",
        label="Injected beam derivative",
    )
    ax.axvline(35, color="gray", ls=":", label="Beam drift energy 35 eV")
    ax.set(
        xlim=(0, 120),
        ylim=(1e-6, 1),
        xlabel="Retarding barrier energy (eV)",
        ylabel="dI/dV (A/V)",
        title="A separate peak in dI/dV at finite barrier energy is a beam clue",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output_dir / "beam_signature.png", dpi=160)
    plt.close(fig)


def _report(rows, output_dir):
    lines = [
        "# Bulk + beam injection/recovery",
        "",
        "Reproduce: `.venv/bin/python synthetic_langmuir_beam_validation.py`.",
        "",
        (
            "Bulk: Te = 3 eV, ne = 5e12 cm^-3. Same He+, B = 0.1 T, 2 × 2 mm "
            "face normal to B as the baseline study. Beam density is relative to "
            "bulk density; beam drift is toward the face, along B. The beam is a "
            "shifted Maxwellian in normal velocity, integrated above the sheath "
            "barrier. Drift energy and thermal spread are separate parameters. "
            "The zero-drift hot-halo control is not a directed beam."
        ),
        "",
        (
            "The synthetic model prescribes a bulk-dominated ion current and "
            "collisionless planar electron flux. It does not solve a self-consistent "
            "beam sheath, return current, instability or magnetic transport problem."
        ),
        "",
        "| Case | Beam / bulk density | Drift (eV) | Spread (eV) | Ordinary Te (eV) | Accepted | Bulk candidate (eV) | Excess background >5 sigma |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        candidate = (
            f"{row['bulk_candidate_te_eV']:.3f}"
            if row["bulk_candidate_found"]
            else "unresolved"
        )
        lines.append(
            f"| {row['case']} | {row['beam_over_bulk_density']:g} | "
            f"{row['beam_drift_eV']:g} | {row['beam_spread_eV']:g} | "
            f"{row['ordinary_te_eV']:.3f} | {row['ordinary_ok']} | {candidate} | "
            f"{row['excess_background_5sigma']} |"
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        (
            "A beam can dominate the negative-bias electron current even at low "
            "density because bulk electrons are exponentially suppressed by the "
            "barrier. The upper retarding branch has a larger bulk fraction, but "
            "a transmitted beam remains as an additive current there. A straight "
            "semilog fit to the sum can therefore remain too hot. Bias farther "
            "negative excludes bulk electrons first, not the energetic beam."
        ),
        "",
        (
            "The opt-in diagnostic fits current directly to "
            "A exp((V-Vmax)/T) + C + D(V-Vmax). C and D describe a locally smooth "
            "ion-plus-beam background. The scan anchors its upper bound 0.2 V "
            "below the measured derivative knee and varies the width. It reports "
            "a candidate only if three neighboring widths agree within 5%, "
            "the exponential spans at least one e-fold, slope uncertainty is "
            "resolved and residuals pass a noise/model-error check. The inferred "
            "bulk current must also have small block-averaged log residuals over "
            "resolved samples (3% amplitude floor or five times the ion-region "
            "noise) and at least one e-fold of signal. Temperature "
            "bounds are 0.1–300 eV; 3 eV is never supplied as a fit target. "
            "These research thresholds are not validated production settings."
        ),
        "",
        (
            "Excess background is the fitted C minus the independent low-bias "
            "ion estimate. A >5-sigma flag is conditional on independent current "
            "noise and the local model. It indicates extra background current, "
            "not uniquely a beam. Ion-current errors, current offsets and other "
            "electron populations can also cause it. A hot halo and a directed "
            "beam can both produce recoverable bulk candidates. Confirmation "
            "needs the beam-retardation shoulder, adequate negative-bias range, "
            "orientation dependence or an independent distribution measurement."
        ),
        "",
        (
            "For the ideal planar retarding model, dI/dV is proportional to the "
            "incident probe-normal velocity distribution evaluated at the barrier "
            "speed. A directed beam can therefore produce an additional peak "
            "in dI/dV near its drift energy. `beam_signature.png` shows this feature; "
            "the measured energy axis uses derivative Vp, so its offset includes "
            "the smoothing bias. The reference curves use the injected Vp. This "
            "is a distribution in normal velocity sampled through barrier energy, "
            "not an isotropic EEDF. Ion slopes and measurement artifacts still "
            "need exclusion before interpreting a measured peak as a beam."
        ),
        "",
        (
            "A finite ordinary temperature is not necessarily accepted: the "
            "solver retains rejected estimates for plotting. Inspect ordinary_ok "
            "and ordinary_warnings. A numerical value near 20 eV is consequently "
            "a possible beam-contamination clue, not proof of a 20 eV bulk. "
            "This study leaves production defaults and output schemas intact; "
            "the new routine is an explicitly invoked diagnostic. Its amplitude "
            "must not be combined with the ordinary total Ies to infer bulk density."
        ),
        "",
        (
            "`recovery.csv` records all scalar estimates and quality flags; "
            "`windows.csv` includes every scanned interval and its uncertainty; "
            "`traces.npz` stores numeric arrays for net, bulk, beam and ion current "
            "plus voltage. Load with `allow_pickle=False`."
        ),
        "",
        (
            "Flux model: [Polzin et al., NASA manuscript (2019), equations 2, 6 and 7]"
            "(https://ntrs.nasa.gov/citations/20190030439). "
            "Experimental interpretation: [UCLA plasma laboratory]"
            "(https://www.physics.ucla.edu/plasma-exp/180E-W97/RetRegAnalysis.html)."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run_study()
