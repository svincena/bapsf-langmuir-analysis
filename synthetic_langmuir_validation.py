"""Reproducible planar-probe injection/recovery study; no experiment files read.

Run with .venv/bin/python synthetic_langmuir_validation.py. Collection changes
are explicit sensitivity parameters, not a kinetic magnetized-sheath solution.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.constants import e, epsilon_0, m_e, physical_constants

from langmuir_analysis_core import analyze_iv_trace


@dataclass(frozen=True)
class SyntheticPlasma:
    """Physical truth and independently specified collection perturbations.

    The area is a single exposed face normal to B. Ti sets thermal gyroradius
    and ion repulsion above Vp; the Bohm amplitude uses the cold-ion closure.
    Electron collection efficiency represents an unknown effective-area/flux
    reduction, not a prediction from B. Ion expansion is fractional per Te of
    negative bias; positive-bias electron growth is fractional per volt.
    """

    te_eV: float = 3.0
    ne_cm3: float = 5e12
    ti_eV: float = 0.5
    vp_V: float = 10.0
    area_m2: float = 4e-6
    magnetic_field_T: float = 0.1
    ion_mass_kg: float = physical_constants["alpha particle mass"][0] + m_e
    electron_collection: float = 1.0
    ion_expansion: float = 0.0
    electron_growth_per_V: float = 0.0
    noise_fraction: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in (
            "te_eV",
            "ne_cm3",
            "ti_eV",
            "area_m2",
            "magnetic_field_T",
            "ion_mass_kg",
            "electron_collection",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("ion_expansion", "electron_growth_per_V", "noise_fraction"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")


def make_trace(plasma=None, *, scaled_span=True, seed=20261029):
    """Return voltage, measured current, and independently computed truth.

    Maxwellian flux integrated over velocities directed toward a planar face
    gives Ie = e n A sqrt(e Te / (2 pi me)) exp(min((V-Vp)/Te, 0)).
    The negative branch is a Bohm ion flux with optional area expansion.
    This approximation deliberately separates algorithm tests from uncertain
    finite-gyroradius transport and electron-sheath physics.
    """
    p = SyntheticPlasma() if plasma is None else plasma
    span_temperature = p.te_eV if scaled_span else 3.0
    voltage = np.linspace(
        p.vp_V - 15 * span_temperature, p.vp_V + 7 * span_temperature, 6601
    )
    bias_from_plasma = voltage - p.vp_V
    ne_m3 = p.ne_cm3 * 1e6
    random_speed = np.sqrt(p.te_eV * e / (2 * np.pi * m_e))
    ies = e * ne_m3 * p.area_m2 * random_speed * p.electron_collection
    iis = np.exp(-0.5) * e * ne_m3 * p.area_m2 * np.sqrt(p.te_eV * e / p.ion_mass_kg)
    electron = ies * np.exp(np.minimum(bias_from_plasma / p.te_eV, 0))
    electron *= 1 + p.electron_growth_per_V * np.maximum(bias_from_plasma, 0)
    ion = -iis * np.where(
        bias_from_plasma <= 0,
        1 + p.ion_expansion * np.maximum(-bias_from_plasma, 0) / p.te_eV,
        np.exp(-np.maximum(bias_from_plasma, 0) / p.ti_eV),
    )
    noise = np.random.default_rng(seed).normal(0, p.noise_fraction * ies, voltage.size)
    truth = {
        **asdict(p),
        "ne_m3": ne_m3,
        "ies_A": ies,
        "iis_at_vp_A": iis,
        "debye_m": np.sqrt(epsilon_0 * p.te_eV / (ne_m3 * e)),
        # Thermal gyroradii use v_perp = sqrt(2 e T / m), with singly charged He+.
        "ion_gyroradius_m": np.sqrt(2 * p.ion_mass_kg * p.ti_eV * e)
        / (e * p.magnetic_field_T),
        "electron_gyroradius_m": np.sqrt(2 * m_e * p.te_eV * e)
        / (e * p.magnetic_field_T),
        "electron_A": electron,
        "ion_A": ion,
        "noiseless_A": electron + ion,
    }
    return voltage, electron + ion + noise, truth


def run_study(output_dir):
    """Save all injected traces, fit estimates, quality flags, and diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = SyntheticPlasma()
    cases = [("baseline", base, True, 20261029)]
    cases += [
        ("density", replace(base, ne_cm3=n), True, 20261029)
        for n in (1e11, 5e11, 1e12, 5e12, 1e13)
    ]
    for te in (0.5, 1, 2, 3, 5, 10, 20):
        for scaled in (True, False):
            cases.append(
                (
                    "temperature_scaled" if scaled else "temperature_fixed",
                    replace(base, te_eV=te, ti_eV=te / 6),
                    scaled,
                    20261029,
                )
            )
    cases += [
        ("ion_expansion", replace(base, ion_expansion=a), True, 20261029)
        for a in (0, 0.01, 0.03, 0.1, 0.3)
    ]
    cases += [
        ("electron_growth", replace(base, electron_growth_per_V=a), True, 20261029)
        for a in (0, 0.01, 0.03, 0.06)
    ]
    cases += [
        ("electron_collection", replace(base, electron_collection=a), True, 20261029)
        for a in (1, 0.7, 0.5, 0.3)
    ]
    cases += [
        ("noise", replace(base, noise_fraction=a), True, seed)
        for a in (0.001, 0.01, 0.03)
        for seed in range(8)
    ]
    rows, archive, selected = [], {}, {}
    for case_id, (family, plasma, scaled, seed) in enumerate(cases):
        voltage, current, truth = make_trace(plasma, scaled_span=scaled, seed=seed)
        prefix = f"case_{case_id:03d}"
        archive[f"{prefix}_voltage_V"] = voltage
        archive[f"{prefix}_current_A"] = current
        archive[f"{prefix}_electron_A"] = truth["electron_A"]
        archive[f"{prefix}_ion_A"] = truth["ion_A"]
        for method, width in (
            ("high_bias_median", 2.5),
            ("at_vp", 2.5),
            ("at_vp", 0.25),
        ):
            result = analyze_iv_trace(
                voltage,
                current,
                {
                    "probe_area": plasma.area_m2,
                    "current_zero_calibrated": True,
                    "ies_method": method,
                    "vp_smoothing_width_V": width,
                },
                include_diagnostic_data=family == "baseline",
            )
            row = {
                "case_id": case_id,
                "family": family,
                "seed": seed,
                "scaled_span": scaled,
                **asdict(plasma),
                "ies_method": method,
                "smoothing_width_V": width,
                "ok": result["ok"],
                "fit_te_eV": result["te_eV"],
                "fit_ne_cm3": result["n_e_m3"] / 1e6,
                "te_error_percent": 100 * (result["te_eV"] / plasma.te_eV - 1),
                "ne_error_percent": 100 * (result["n_e_m3"] / truth["ne_m3"] - 1),
                "vp_derivative_V": result["vp_derivative_V"],
                "vp_fit_V": result["vp_fit_V"],
                "fit_vp_V": result["vp_V"],
                "fit_ies_A": result["ies_A"],
                "truth_ies_A": truth["ies_A"],
                "fit_iis_A": result["iis_A"],
                "truth_iis_at_vp_A": truth["iis_at_vp_A"],
                "te_fit_r2": result["te_fit_r2"],
                "te_fit_vstart": result["te_fit_vstart"],
                "te_fit_vstop": result["te_fit_vstop"],
                "warnings": " | ".join(result["warnings"]),
                "model_notes": " | ".join(result["model_notes"]),
            }
            rows.append(row)
            if family == "baseline":
                selected[(method, width)] = result
        print(f"{prefix}: {family}", flush=True)
    with (output_dir / "recovery.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output_dir / "traces.npz", **archive)
    _plot_study(rows, selected, output_dir)
    _write_report(rows, output_dir)
    return rows


def _plot_study(rows, selected, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 2, figsize=(13, 13), layout="constrained")
    voltage, current, truth = make_trace()
    axs[0, 0].plot(voltage, current, color="black", label="Injected net current")
    axs[0, 0].axvline(10, color="black", ls=":", label="True Vp")
    colors = ["#237b4b", "#c34632", "#286fc2"]
    for ((method, width), result), color in zip(selected.items(), colors):
        label = f"{method}, {width:g} V"
        axs[0, 0].axvline(result["vp_derivative_V"], color=color, ls="--", label=label)
        data = result["diagnostic_data"]
        mask = data["te_fit_mask"]
        axs[0, 1].semilogy(
            data["V_binned"][mask],
            data["I_binned"][mask] - result["ion_current_A"],
            color=color,
            label=f"{label}: Te={result['te_eV']:.3f}",
        )
    axs[0, 0].set(
        xlabel="Bias (V)",
        ylabel="Net current (A)",
        title="Te = 3 eV, ne = 5 × 10¹² cm⁻³",
    )
    axs[0, 1].semilogy(
        voltage[voltage <= 10],
        truth["electron_A"][voltage <= 10],
        "k:",
        label="True electron current",
    )
    axs[0, 1].set(
        xlabel="Bias (V)",
        ylabel="Electron current (A)",
        ylim=(0.01, 1.5),
        xlim=(-5, 11),
        title="Selected semilog fits",
    )
    families = ("temperature_scaled", "density", "ion_expansion", "electron_collection")
    keys = ("te_eV", "ne_cm3", "ion_expansion", "electron_collection")
    for ax, family, key in zip(axs.flat[2:], families, keys):
        for (method, width), color in zip(selected, colors):
            data = [
                r
                for r in rows
                if r["family"] == family
                and r["ies_method"] == method
                and r["smoothing_width_V"] == width
            ]
            ykey = (
                "te_error_percent" if family == "ion_expansion" else "ne_error_percent"
            )
            ax.plot(
                [r[key] for r in data],
                [r[ykey] for r in data],
                ".-",
                color=color,
                label=f"{method}, {width:g} V",
            )
            rejected = [r for r in data if not r["ok"]]
            ax.scatter(
                [r[key] for r in rejected],
                [r[ykey] for r in rejected],
                marker="x",
                s=90,
                color=color,
            )
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(
            xlabel={
                "te_eV": "True Te (eV)",
                "ne_cm3": "True ne (cm⁻³)",
                "ion_expansion": "Ion expansion per Te",
                "electron_collection": "Electron collection efficiency",
            }[key],
            ylabel="Te error (%)" if family == "ion_expansion" else "Density error (%)",
            title=family.replace("_", " ").capitalize(),
        )
        if family == "density":
            ax.set_xscale("log")
    for ax in axs.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Synthetic planar Langmuir injection / recovery — crosses mark rejected fits",
        fontsize=14,
    )
    fig.savefig(output_dir / "recovery.png", dpi=160)
    plt.close(fig)


def _write_report(rows, output_dir):
    _, _, truth = make_trace()
    lines = [
        "# Synthetic Langmuir injection/recovery study",
        "",
        "Reproduce: `.venv/bin/python synthetic_langmuir_validation.py`.",
        "",
        "## Physical assumptions",
        "",
        (
            "Injected Te = 3 eV, ne = 5e12 cm^-3 = 5e18 m^-3, Vp = 10 V. "
            "Single exposed 2 × 2 mm face normal to B = 0.1 T; singly charged He+, "
            "Ti = 0.5 eV. Thermal gyroradii use sqrt(2 e T / m)."
        ),
        (
            f"Debye length = {truth['debye_m'] * 1e6:.3f} µm; "
            f"electron gyroradius = {truth['electron_gyroradius_m'] * 1e3:.4f} mm; "
            f"ion gyroradius = {truth['ion_gyroradius_m'] * 1e3:.3f} mm."
        ),
        "",
        (
            "The baseline is a collisionless Maxwellian planar-flux benchmark, with "
            "Bohm ion current (cold-ion closure, presheath coefficient exp(-1/2)). "
            "I_e = I_es exp(min((V-Vp)/Te, 0)); "
            "I_i = -I_is below Vp, decaying as exp(-(V-Vp)/Ti) above Vp. "
            "The finite ion gyroradius sets the regime; it does not determine a "
            "unique magnetized collection law. Flow, orientation, collisions, "
            "transport and sheath geometry would be needed for that. This is not a "
            "PIC simulation. The deliberately independent ion/electron perturbations "
            "are sensitivity tests, not asserted predictions from B."
        ),
        "",
        (
            f"Baseline Ies = {truth['ies_A']:.6f} A, Iis = {truth['iis_at_vp_A']:.6f} A. "
            "Positive current means electron collection. Current zero and bias are "
            "assumed independently calibrated; no electronics transfer function is simulated."
        ),
        "",
        "## Baseline recovery",
        "",
        "| Ies method | Smoothing (V) | Te (eV) | ne (cm^-3) | ne error | derivative Vp (V) | Accepted |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["family"] == "baseline":
            lines.append(
                f"| {row['ies_method']} | {row['smoothing_width_V']} | "
                f"{row['fit_te_eV']:.5f} | {row['fit_ne_cm3']:.5g} | "
                f"{row['ne_error_percent']:+.2f}% | {row['vp_derivative_V']:.4f} | {row['ok']} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        (
            "The finite smoothing window shifts the maximum derivative below a sharp "
            "exponential-to-saturation knee. The at_vp method samples the electron "
            "current at that shifted voltage, so its density is systematically low "
            "even with a nearly exact temperature and R² near one. At_vp's existing "
            "fit-extrapolation consistency check uses this same sampled current and "
            "can therefore miss the bias. The solver now independently flags a "
            "greater-than-20% disagreement with a nearly flat high-bias branch; "
            "ideal-model mode rejects that inconsistency. It does not replace either "
            "current definition or silently correct the density."
        ),
        "",
        (
            "A 0.25 V window substantially reduces the baseline noiseless bias but "
            "is not a universal default: noise, voltage-bin spacing, ion repulsion "
            "and the physical knee width still matter. Check a range of smoothing "
            "widths on measured data. The high-bias method succeeds on a true "
            "plateau but overestimates density if electron collection keeps growing."
        ),
        "",
        (
            "Density-only scans test current-scale invariance. Temperature scans use "
            "Ti = Te/6 and either a span from Vp-15Te to Vp+7Te or the fixed baseline "
            "span (-35 to +31 V). Fixed-span failures at high Te indicate inadequate "
            "ion coverage, not a density conversion error. The default Te acceptance "
            "range remains in effect; rejected finite estimates are retained in the CSV."
        ),
        "",
        (
            "Ion expansion is I_i = -I_is [1 + alpha (Vp-V)/Te] below Vp. Subtracting "
            "one low-bias median then leaves a positive electron-current offset and "
            "can overestimate Te despite a high R². Correcting that requires a "
            "validated ion-collection model; the study does not change the physics."
        ),
        "",
        (
            "A bias-independent electron collection factor f is indistinguishable "
            "from f times the density when the solver is given the geometric area. "
            "Te can remain accurate while ne is too low. No fit-only correction can "
            "resolve that degeneracy without effective-area/transport information "
            "or an independent density measurement."
        ),
        "",
        "## Selected parameter-scan results",
        "",
        (
            "These rows use the default high-bias method and 2.5 V smoother. "
            "The strongly expanded ion case is rejected despite its finite estimate."
        ),
        "",
        "| Perturbation | Te recovered (eV) | Density error | R² | Accepted |",
        "|---|---:|---:|---:|---|",
    ]
    examples = (
        ("ion_expansion", "ion_expansion", 0.1, "Ion expansion alpha = 0.1"),
        ("ion_expansion", "ion_expansion", 0.3, "Ion expansion alpha = 0.3"),
        ("electron_growth", "electron_growth_per_V", 0.03, "Electron growth 0.03 / V"),
        (
            "electron_collection",
            "electron_collection",
            0.5,
            "Electron collection f = 0.5",
        ),
        ("temperature_fixed", "te_eV", 10, "Te = 10 eV, fixed voltage span"),
    )
    for family, key, value, label in examples:
        row = next(
            r
            for r in rows
            if r["family"] == family
            and r[key] == value
            and r["ies_method"] == "high_bias_median"
        )
        lines.append(
            f"| {label} | {row['fit_te_eV']:.4f} | "
            f"{row['ne_error_percent']:+.2f}% | {row['te_fit_r2']:.6f} | {row['ok']} |"
        )
    lines += [
        "",
        "## Noise ensemble",
        "",
        (
            "Gaussian raw-current noise is specified relative to true Ies, with "
            "eight independent seeds per level. Quantiles below use accepted fits "
            "only; the acceptance count exposes rejected cases."
        ),
        "",
        "| Noise / Ies | Method | Width (V) | Accepted / 8 | Te error median [5%,95%] | ne error median [5%,95%] |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for noise in (0.001, 0.01, 0.03):
        for method, width in (
            ("high_bias_median", 2.5),
            ("at_vp", 2.5),
            ("at_vp", 0.25),
        ):
            accepted = [
                r
                for r in rows
                if r["family"] == "noise"
                and r["noise_fraction"] == noise
                and r["ies_method"] == method
                and r["smoothing_width_V"] == width
                and r["ok"]
            ]
            stats = []
            for key in ("te_error_percent", "ne_error_percent"):
                if accepted:
                    lo, median, hi = np.percentile(
                        [r[key] for r in accepted], [5, 50, 95]
                    )
                    stats.append(f"{median:+.2f}% [{lo:+.2f}, {hi:+.2f}]")
                else:
                    stats.append("unavailable")
            lines.append(
                f"| {noise:g} | {method} | {width:g} | {len(accepted)} | "
                + " | ".join(stats)
                + " |"
            )
    lines += [
        "",
        "## Files and sources",
        "",
        (
            f"{len(rows)} fits; recovery.csv contains injected parameters, seed, "
            "fit interval, recovered estimates, acceptance, warnings and notes. "
            "traces.npz contains each case's bias, noisy net current and exact "
            "electron/ion components in SI units. Keys match case_NNN in the CSV. "
            "Load with np.load(..., allow_pickle=False). recovery.png summarizes "
            "the baseline and systematic scans. Solver options are core defaults "
            "except area, calibrated zero, and the displayed method/window."
        ),
        "",
        (
            "Planar Maxwellian/Bohm benchmark: "
            "[Qayyum et al. (2021), equations 1–3](https://doi.org/10.1007/s42452-020-04019-9). "
            "Magnetized collection is model dependent: "
            "[Brussaard et al. (1996)](https://doi.org/10.1103/PhysRevE.54.1906). "
            "Ion expansion and finite voltage-span sensitivity: "
            "[Komm et al. (2022)](https://doi.org/10.1088/1741-4326/ac8011)."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("synthetic_validation_output")
    )
    args = parser.parse_args()
    rows = run_study(args.output_dir)
    print(f"Saved {len(rows)} fits to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
