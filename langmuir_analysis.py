"""Reusable Langmuir probe analysis helpers."""

import os
import astropy.units as u
from astropy.constants import e, m_e, k_B
import numpy as np
from scipy.signal import savgol_filter



def density_from_electron_saturation_current(
    I_esat,
    Te,
    probe_area,
    use_abs=True,
):
    """
    Compute plasma density from electron saturation current.

    Assumes a Maxwellian electron distribution and the planar-probe
    electron saturation current relation

        I_esat = e * n_e * A * sqrt(k_B T_e / (2*pi*m_e))

    or, when T_e is supplied in eV,

        I_esat = e * n_e * A * sqrt(T_e[eV] * e / (2*pi*m_e))

    Parameters
    ----------
    I_esat : float or astropy.units.Quantity
        Electron saturation current. If unitless, assumed to be in amperes.

    Te : float or astropy.units.Quantity
        Electron temperature. If unitless, assumed to be in eV.
        Can also be supplied with units such as `u.eV`, `u.J`, or `u.K`.

    probe_area : float or astropy.units.Quantity
        Effective electron collection area. If unitless, assumed to be in m^2.

    use_abs : bool, optional
        If True, use the absolute value of I_esat. This is useful if the
        electron saturation current is recorded with a negative sign.

    Returns
    -------
    n_e : astropy.units.Quantity
        Electron density with units of m^-3.
    """

    # Attach default units if inputs are unitless
    if not isinstance(I_esat, u.Quantity):
        I_esat = np.asarray(I_esat) * u.A

    if not isinstance(Te, u.Quantity):
        Te = np.asarray(Te) * u.eV

    if not isinstance(probe_area, u.Quantity):
        probe_area = probe_area * u.m**2

    # Convert current and area to SI units
    I_esat = I_esat.to(u.A)
    probe_area = probe_area.to(u.m**2)

    if use_abs:
        I_esat = np.abs(I_esat)

    if np.any(probe_area <= 0 * u.m**2):
        raise ValueError("probe_probe_area must be positive.")

    if np.any(Te <= 0 * Te.unit):
        raise ValueError("Te must be positive.")

    # Convert electron temperature to energy
    if Te.unit.is_equivalent(u.K):
        Te_energy = (k_B * Te).to(u.J)
    else:
        Te_energy = Te.to(u.J)

    electron_flux_speed = np.sqrt(Te_energy / (2 * np.pi * m_e))

    n_e = I_esat / (e.si * probe_area * electron_flux_speed)

    return n_e.to(u.m**-3)

def choose_analysis_process_count(n_traces, requested_processes=None):
    """Choose a conservative worker count for per-trace Langmuir analysis."""
    n_traces = int(n_traces)
    if requested_processes is not None:
        return max(1, min(int(requested_processes), n_traces))

    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count - 1, n_traces))


def bin_average_by_voltage(V, I, bin_width):
    """
    Sort by V, bin nearby voltage samples, and average I within each bin.
    Returns monotonic Vb, Ib.
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)

    good = np.isfinite(V) & np.isfinite(I)
    V = V[good]
    I = I[good]

    if V.size < 2:
        return None, None

    order = np.argsort(V)
    V = V[order]
    I = I[order]

    vmin = V.min()
    bin_ids = np.floor((V - vmin) / bin_width).astype(int)

    uniq = np.unique(bin_ids)
    Vb = np.empty(len(uniq), dtype=float)
    Ib = np.empty(len(uniq), dtype=float)

    for j, bid in enumerate(uniq):
        m = (bin_ids == bid)
        Vb[j] = np.mean(V[m])
        Ib[j] = np.mean(I[m])

    keep = np.isfinite(Vb) & np.isfinite(Ib)
    Vb = Vb[keep]
    Ib = Ib[keep]

    dV = np.diff(Vb)
    keep2 = np.concatenate(([True], dV > 0))
    Vb = Vb[keep2]
    Ib = Ib[keep2]

    if Vb.size < 2:
        return None, None

    return Vb, Ib


def make_monotonic_iv_curve(bias, current, npts=600, bin_width=0.05):
    """
    Convert time-ordered sweep into a clean monotonic I(V) curve by:
      1) sorting by V
      2) bin-averaging repeated/near-repeated V values
      3) interpolating onto a uniform increasing V grid
    """
    V = np.asarray(bias.to_value(u.V), dtype=float)
    I = np.asarray(current.to_value(u.A), dtype=float)

    Vb, Ib = bin_average_by_voltage(V, I, bin_width=bin_width)
    if Vb is None or Vb.size < 5:
        return None

    Vg = np.linspace(Vb.min(), Vb.max(), npts)
    Ig = np.interp(Vg, Vb, Ib)

    return {
        "V_raw": V,
        "I_raw": I,
        "V_binned": Vb,
        "I_binned": Ib,
        "V_grid": Vg,
        "I_grid": Ig,
    }


def find_floating_potential_from_iv(V, I):
    """Find Vf from zero crossing of I(V) on monotonic grid."""
    signs = np.sign(I)
    crossings = np.where(np.diff(signs) != 0)[0]

    if crossings.size == 0:
        idx = np.argmin(np.abs(I))
        return V[idx] * u.V

    i0 = crossings[0]
    v1, v2 = V[i0], V[i0 + 1]
    i1, i2 = I[i0], I[i0 + 1]

    if i2 == i1:
        vf = 0.5 * (v1 + v2)
    else:
        vf = v1 - i1 * (v2 - v1) / (i2 - i1)

    return vf * u.V


def find_plasma_potential_from_iv(
    V,
    I,
    vf=None,
    smoothing=None,
    moving_average_width=10,
    iv_savgol_window=31,
    iv_savgol_order=2,
):
    """
    Estimate plasma potential from max dI/dV on a monotonic interpolated IV curve.
    Optionally smooths current before taking the derivative.
    """
    if len(V) < 7:
        return np.nan * u.V, np.full_like(V, np.nan)

    if smoothing is None or str(smoothing).lower() == "none":
        I_smooth = I.copy()
    elif str(smoothing).lower() == "moving":
        win = int(moving_average_width)
        if win < 1:
            raise ValueError("moving_average_width must be at least 1.")
        win = min(win, len(I))
        kernel = np.ones(win, dtype=float) / win
        weights = np.convolve(np.ones_like(I, dtype=float), kernel, mode="same")
        I_smooth = np.convolve(I, kernel, mode="same") / weights
    elif str(smoothing).lower() == "savgol":
        win = min(iv_savgol_window, len(I) - (1 - len(I) % 2))
        if win < 5:
            win = 5
        if win % 2 == 0:
            win -= 1
        if win <= iv_savgol_order:
            win = iv_savgol_order + 3
            if win % 2 == 0:
                win += 1
        if win >= len(I):
            win = len(I) - 1 if len(I) % 2 == 0 else len(I)

        if win < 5:
            I_smooth = I.copy()
        else:
            I_smooth = savgol_filter(I, win, iv_savgol_order)
    else:
        raise ValueError(
            f"Unknown smoothing type {smoothing!r}. Use None, 'moving', or 'savgol'."
        )

    dIdV = np.gradient(I_smooth, V)

    mask = np.isfinite(dIdV)
    if vf is not None and np.isfinite(vf.to_value(u.V)):
        mask &= (V >= vf.to_value(u.V))

    if not np.any(mask):
        return np.nan * u.V, dIdV

    idx_local = np.argmax(dIdV[mask])
    idx = np.where(mask)[0][idx_local]
    return V[idx] * u.V, dIdV


def find_best_te_window(
    V,
    I,
    vf=None,
    vp=None,
    min_points=32,
    max_points=1024,
    margin_from_vp=1.0,
    current_floor_frac=0.03,
    te_min_eV=0.1,
    te_max_eV=30.0,
    min_r2=0.90,
    subtract_i0=True,
):
    """
    Scan candidate windows in the electron-retarding region and fit log current vs V.
    If subtract_i0 is True, fit ln(I - I0), where I0 is a small baseline shift
    based on the candidate-region current minimum. Otherwise fit ln(I).
    """
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)

    mask = np.isfinite(V) & np.isfinite(I)

    if vf is not None and np.isfinite(vf.to_value(u.V)):
        mask &= (V >= vf.to_value(u.V))

    if vp is not None and np.isfinite(vp.to_value(u.V)):
        mask &= (V <= (vp.to_value(u.V) - margin_from_vp))

    idx_all = np.where(mask)[0]
    if idx_all.size < min_points:
        return None

    Vc = V[idx_all]
    Ic = I[idx_all]

    I_span = np.nanmax(Ic) - np.nanmin(Ic)
    if I_span <= 0:
        return None

    I_min_region = np.nanmin(Ic)
    I_scale_region = max(1.0, np.nanmax(np.abs(Ic)))
    I0_region = I_min_region - 1e-12 * I_scale_region if subtract_i0 else 0.0

    keep = Ic > (I_min_region + current_floor_frac * I_span)
    if np.count_nonzero(keep) < min_points:
        return None

    Vc = Vc[keep]
    Ic = Ic[keep]
    idx_all = idx_all[keep]

    best = None
    candidate_count = 0
    nreg = len(Vc)
    max_points = min(max_points, nreg)

    for start in range(0, nreg - min_points + 1):
        for stop in range(start + min_points, min(nreg, start + max_points) + 1):
            Vw = Vc[start:stop]
            Iw = Ic[start:stop]

            Ye = Iw - I0_region

            if np.any(Ye <= 0):
                continue

            y = np.log(Ye)

            try:
                a, b = np.polyfit(Vw, y, 1)
            except Exception:
                continue

            if not np.isfinite(a) or a <= 0:
                continue

            te_eV = 1.0 / a
            if not (te_min_eV <= te_eV <= te_max_eV):
                continue

            yfit = a * Vw + b
            resid = y - yfit

            ss_res = np.sum(resid**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf
            rmse = np.sqrt(np.mean(resid**2))
            dv = Vw[-1] - Vw[0]

            score = r2 + 1e-5 * dv + 1e-6 * len(Vw)
            candidate_count += 1

            candidate = {
                "score": score,
                "r2": r2,
                "rmse_logI": rmse,
                "slope": a,
                "intercept": b,
                "te_eV": te_eV,
                "npts": len(Vw),
                "start_idx_global": idx_all[start],
                "stop_idx_global": idx_all[stop - 1],
                "vstart": Vw[0],
                "vstop": Vw[-1],
                "I0": I0_region,
                "subtract_i0": bool(subtract_i0),
            }

            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        return None

    fit_mask = np.zeros_like(V, dtype=bool)
    fit_mask[best["start_idx_global"]:best["stop_idx_global"] + 1] = True
    best["fit_mask"] = fit_mask
    best["te"] = best["te_eV"] * u.eV
    best["passed_r2"] = bool(np.isfinite(best["r2"]) and best["r2"] >= min_r2)
    best["candidate_count"] = candidate_count
    return best


def analyze_iv_trace(
    bias_values,
    current_values,
    config,
    include_diagnostic_data=False,
    trace_label=None,
):
    """Run the numeric Langmuir analysis for one I-V trace."""
    result = {
        "trace_label": trace_label,
        "ok": False,
        "warnings": [],
        "te_eV": np.nan,
        "vp_V": np.nan,
        "vf_V": np.nan,
        "ies_A": np.nan,
        "iis_A": np.nan,
        "n_e_m3": np.nan,
        "te_fit_r2": np.nan,
        "te_fit_rmse": np.nan,
        "te_fit_npts": 0,
        "te_fit_vstart": np.nan,
        "te_fit_vstop": np.nan,
        "te_fit_slope": np.nan,
        "te_fit_intercept": np.nan,
        "te_fit_i0": np.nan,
        "te_fit_subtract_i0": bool(config.get("te_subtract_i0", True)),
        "te_fit_passed_r2": False,
        "te_fit_candidate_count": 0,
        "iv_voltage_grid": None,
        "iv_current_grid": None,
        "iv_didv_grid": None,
        "iv_fit_mask": None,
        "diagnostic_data": None,
    }

    bias = np.asarray(bias_values, dtype=float) * u.V
    current = np.asarray(current_values, dtype=float) * u.A

    iv = make_monotonic_iv_curve(
        bias,
        current,
        npts=config["iv_npts"],
        bin_width=config["voltage_bin_width"],
    )

    if iv is None:
        result["warnings"].append("Warning: could not make monotonic IV curve")
        return result

    Vg = iv["V_grid"]
    Ig = iv["I_grid"]

    V_F = find_floating_potential_from_iv(Vg, Ig)
    V_P, dIdV = find_plasma_potential_from_iv(
        Vg,
        Ig,
        vf=V_F,
        smoothing=config.get("vp_smoothing"),
        moving_average_width=config.get("vp_moving_average_width", 10),
        iv_savgol_window=config.get("iv_savgol_window", 31),
        iv_savgol_order=config.get("iv_savgol_order", 2),
    )

    if np.isfinite(V_P.to_value(u.V)):
        mask_es = (Vg >= V_P.to_value(u.V)) & (Vg <= V_P.to_value(u.V) + 1.0)
        if np.any(mask_es):
            I_es = np.mean(Ig[mask_es]) * u.A
        else:
            idx_vp = np.argmin(np.abs(Vg - V_P.to_value(u.V)))
            I_es = Ig[idx_vp] * u.A
    else:
        I_es = np.nan * u.A

    if np.isfinite(V_F.to_value(u.V)):
        V_min = np.min(Vg)
        V_end = V_min + 0.2 * (V_F.to_value(u.V) - V_min)
        if V_end <= V_min:
            I_is = np.nan * u.A
        else:
            mask_is = (Vg >= V_min) & (Vg <= V_end)
            if np.any(mask_is):
                I_is = np.abs(np.nanmean(Ig[mask_is])) * u.A
            else:
                I_is = np.nan * u.A
    else:
        I_is = np.nan * u.A

    te_result = find_best_te_window(
        Vg,
        Ig,
        vf=V_F,
        vp=V_P,
        min_points=config["te_min_points"],
        max_points=config["te_max_points"],
        margin_from_vp=config["te_margin_from_vp"],
        current_floor_frac=config["te_current_floor_frac"],
        te_min_eV=config["te_min_eV"],
        te_max_eV=config["te_max_eV"],
        min_r2=config.get("te_min_r2", 0.90),
        subtract_i0=config.get("te_subtract_i0", True),
    )

    fit_mask = np.zeros_like(Vg, dtype=np.uint8)
    if te_result is None:
        result["warnings"].append("Warning: could not find valid Te window")
        T_e = np.nan * u.eV
    else:
        T_e = te_result["te"]
        result["te_fit_r2"] = te_result["r2"]
        result["te_fit_rmse"] = te_result["rmse_logI"]
        result["te_fit_npts"] = te_result["npts"]
        result["te_fit_vstart"] = te_result["vstart"]
        result["te_fit_vstop"] = te_result["vstop"]
        result["te_fit_slope"] = te_result["slope"]
        result["te_fit_intercept"] = te_result["intercept"]
        result["te_fit_i0"] = te_result["I0"]
        result["te_fit_subtract_i0"] = te_result["subtract_i0"]
        result["te_fit_passed_r2"] = te_result["passed_r2"]
        result["te_fit_candidate_count"] = te_result["candidate_count"]
        fit_mask = te_result["fit_mask"].astype(np.uint8)

    result["te_eV"] = T_e.to_value(u.eV)
    result["vp_V"] = V_P.to_value(u.V)
    result["vf_V"] = V_F.to_value(u.V)
    result["ies_A"] = I_es.to_value(u.A)
    result["iis_A"] = I_is.to_value(u.A)
    result["iv_voltage_grid"] = Vg
    result["iv_current_grid"] = Ig
    result["iv_didv_grid"] = dIdV
    result["iv_fit_mask"] = fit_mask
    result["ok"] = True

    # Calculate electron density from electron saturation current and temperature
    if np.isfinite(I_es.to_value(u.A)) and np.isfinite(T_e.to_value(u.eV)):
        probe_area = config.get("probe_area")
        if probe_area is None:
            result["warnings"].append(
                "Missing required 'probe_area' in config; cannot compute electron density 'n_e_m3'."
            )
        else:
            try:
                n_e = density_from_electron_saturation_current(I_es, T_e, probe_area)
                result["n_e_m3"] = n_e.to_value(u.m**-3)
            except Exception as e:
                result["warnings"].append(f"Failed to compute n_e: {e}")

    if include_diagnostic_data:
        result["diagnostic_data"] = {
            "V_raw": iv["V_raw"],
            "I_raw": iv["I_raw"],
            "V_binned": iv["V_binned"],
            "I_binned": iv["I_binned"],
        }

    return result


def build_iv_diagnostic_summary(result):
    """Return standard scalar summary lines for an analyzed Langmuir I-V trace."""
    te_eV = result.get("te_eV", np.nan)
    vp_V = result.get("vp_V", np.nan)
    vf_V = result.get("vf_V", np.nan)
    ies_A = result.get("ies_A", np.nan)
    n_e_m3 = result.get("n_e_m3", np.nan)
    fit_r2 = result.get("te_fit_r2", np.nan)
    fit_rmse = result.get("te_fit_rmse", np.nan)
    fit_npts = result.get("te_fit_npts", 0)
    fit_vstart = result.get("te_fit_vstart", np.nan)
    fit_vstop = result.get("te_fit_vstop", np.nan)
    fit_i0 = result.get("te_fit_i0", np.nan)
    fit_subtract_i0 = result.get("te_fit_subtract_i0", True)
    fit_passed_r2 = result.get("te_fit_passed_r2", False)
    fit_npts_text = str(int(fit_npts)) if np.isfinite(fit_npts) else "NaN"

    summary = [
        f"Te = {te_eV:.4g} eV" if np.isfinite(te_eV) else "Te = NaN",
        f"Vp = {vp_V:.4g} V" if np.isfinite(vp_V) else "Vp = NaN",
        f"Vf = {vf_V:.4g} V" if np.isfinite(vf_V) else "Vf = NaN",
        f"I_es = {ies_A:.3g} A" if np.isfinite(ies_A) else "I_es = NaN",
        f"n_e = {n_e_m3:.3g} m^-3" if np.isfinite(n_e_m3) else "n_e = NaN",
        f"R^2 = {fit_r2:.4f}" if np.isfinite(fit_r2) else "R^2 = NaN",
        f"R^2 pass = {'yes' if fit_passed_r2 else 'no'}",
        f"RMSE = {fit_rmse:.3g}" if np.isfinite(fit_rmse) else "RMSE = NaN",
        f"N = {fit_npts_text}",
    ]
    if np.isfinite(fit_vstart) and np.isfinite(fit_vstop):
        summary.append(f"V fit = {fit_vstart:.3g} to {fit_vstop:.3g} V")
    if np.isfinite(fit_i0) and fit_subtract_i0:
        summary.append(f"I0 = {fit_i0:.3g} A")
    elif np.isfinite(fit_i0):
        summary.append("I0 subtraction = off")

    return summary


def plot_iv_diagnostic_panels(result, axs):
    """Plot standard I-V, dI/dV, and Te-fit diagnostic panels on three axes."""
    diagnostic_data = result["diagnostic_data"]
    Vg = result["iv_voltage_grid"]
    Ig = result["iv_current_grid"]
    dIdV = result["iv_didv_grid"]
    fit_mask = result["iv_fit_mask"].astype(bool)
    vf_V = result["vf_V"]
    vp_V = result["vp_V"]
    fit_vstart = result["te_fit_vstart"]
    fit_vstop = result["te_fit_vstop"]
    fit_slope = result["te_fit_slope"]
    fit_intercept = result["te_fit_intercept"]
    fit_i0 = result["te_fit_i0"]
    fit_subtract_i0 = result["te_fit_subtract_i0"]

    axs[0].plot(diagnostic_data["V_raw"], diagnostic_data["I_raw"], alpha=0.35, label="raw time-ordered")
    axs[0].plot(diagnostic_data["V_binned"], diagnostic_data["I_binned"], ".", ms=3, label="binned")
    axs[0].plot(Vg, Ig, lw=2, label="interpolated monotonic")
    if np.any(fit_mask):
        axs[0].plot(Vg[fit_mask], Ig[fit_mask], lw=3, label="Te fit window")
    axs[0].axvline(vf_V, ls="--", label="Vf")
    if np.isfinite(vp_V):
        axs[0].axvline(vp_V, ls="--", label="Vp")
    axs[0].set_xlabel("Bias (V)")
    axs[0].set_ylabel("Current (A)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(Vg, dIdV, lw=2)
    axs[1].axvline(vf_V, ls="--", label="Vf")
    if np.isfinite(vp_V):
        axs[1].axvline(vp_V, ls="--", label="Vp")
    axs[1].set_xlabel("Bias (V)")
    axs[1].set_ylabel("dI/dV (A/V)")
    axs[1].grid(True)
    axs[1].legend()

    te_fit_ylim_values = []

    if np.isfinite(fit_i0):
        electron_current = Ig - fit_i0
        log_good = np.isfinite(Vg) & np.isfinite(electron_current) & (electron_current > 0)
        log_label = "log(I - I0)" if fit_subtract_i0 else "log(I)"
        axs[2].plot(Vg[log_good], np.log(electron_current[log_good]), lw=1.5, label=log_label)

        fit_good = log_good & fit_mask
        if np.any(fit_good):
            fit_log_values = np.log(electron_current[fit_good])
            axs[2].plot(Vg[fit_good], fit_log_values, ".", ms=5, label="fit samples")
            te_fit_ylim_values.append(fit_log_values)

        if np.isfinite(fit_slope) and np.isfinite(fit_intercept):
            if np.isfinite(fit_vstart) and np.isfinite(fit_vstop):
                line_mask = np.isfinite(Vg) & (Vg >= fit_vstart) & (Vg <= fit_vstop)
            else:
                line_mask = fit_mask
            if np.count_nonzero(line_mask) >= 2:
                fit_line_values = fit_slope * Vg[line_mask] + fit_intercept
                axs[2].plot(
                    Vg[line_mask],
                    fit_line_values,
                    lw=2.5,
                    label="Te linear fit",
                )
                te_fit_ylim_values.append(fit_line_values)
    else:
        axs[2].text(0.5, 0.5, "No valid Te fit", transform=axs[2].transAxes, ha="center", va="center")

    axs[2].axvline(vf_V, ls="--", label="Vf")
    if np.isfinite(vp_V):
        axs[2].axvline(vp_V, ls="--", label="Vp")
    if np.isfinite(fit_vstart):
        axs[2].axvline(fit_vstart, color="0.4", ls=":", label="fit bounds")
    if np.isfinite(fit_vstop):
        axs[2].axvline(fit_vstop, color="0.4", ls=":")

    if te_fit_ylim_values:
        try:
            y_fit = np.concatenate([y[np.isfinite(y)] for y in te_fit_ylim_values])
            if y_fit.size > 0:
                ymin = np.nanmin(y_fit)
                ymax = np.nanmax(y_fit)
                yrange = ymax - ymin
                pad = 0.20 * yrange if yrange > 0 else max(0.1, 0.05 * max(abs(ymin), 1.0))
                ycenter = 0.5 * (ymin + ymax)
                yhalf_range = ymax - ycenter + pad
                axs[2].set_ylim(
                    ycenter - 2.0 * yhalf_range,
                    ycenter + 2.0 * yhalf_range,
                )
        except (TypeError, ValueError):
            pass

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
    axs[2].set_ylabel("log(I - I0)" if fit_subtract_i0 else "log(I)")
    axs[2].grid(True)
    axs[2].legend()
    try:
        if np.isfinite(vf_V) and np.isfinite(vp_V) and vf_V != vp_V:
            xmin = min(vf_V, vp_V)
            xmax = max(vf_V, vp_V)
            xcenter = 0.5 * (xmin + xmax)
            xhalf_range = xmax - xcenter
            axs[2].set_xlim(
                xcenter - 2.0 * xhalf_range,
                xcenter + 2.0 * xhalf_range,
            )
    except (TypeError, ValueError):
        pass


def render_iv_diagnostic_plot(result, title):
    """Render detailed diagnostics for one analyzed I-V trace."""
    if result.get("diagnostic_data") is None:
        raise ValueError("result does not include diagnostic_data; rerun analyze_iv_trace with include_diagnostic_data=True.")

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(8, 11), constrained_layout=True)
    plot_iv_diagnostic_panels(result, axs)
    fig.suptitle(title)
    return fig
