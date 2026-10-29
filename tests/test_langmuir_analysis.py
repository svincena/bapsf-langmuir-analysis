import astropy.units as u
import numpy as np
import pytest
from astropy.constants import e, k_B, m_e

from langmuir_analysis_core import (
    _fit_electron_temperature,
    _validated_config,
    analyze_iv_trace,
    density_from_electron_saturation_current,
)


def _config(**overrides):
    config = {
        "iv_npts": 512,
        "voltage_bin_width": 0.05,
        "vp_smoothing": "savgol",
        "vp_smoothing_width_V": 1.0,
        "vp_savgol_order": 2,
        "te_min_points": 8,
        "te_margin_from_vp": 0.2,
        "te_current_floor_frac": 0.03,
        "te_min_eV": 0.1,
        "te_max_eV": 30.0,
        "te_min_r2": 0.95,
        "probe_area": 4e-6,
        "current_zero_calibrated": True,
    }
    config.update(overrides)
    return config


def _ideal_trace(ion_current_A=-2e-4, n_samples=1901):
    te_eV = 3.0
    vp_V = 10.0
    density_m3 = 1e16
    area_m2 = 4e-6
    thermal_speed = np.sqrt((te_eV * u.eV).to_value(u.J) / (2 * np.pi * m_e.value))
    ies_A = e.si.value * density_m3 * area_m2 * thermal_speed

    voltage = np.linspace(-20.0, 18.0, n_samples)
    electron_current = ies_A * np.where(
        voltage <= vp_V,
        np.exp((voltage - vp_V) / te_eV),
        1.0,
    )
    ion_current = ion_current_A * np.where(
        voltage <= vp_V,
        1.0,
        np.exp(-(voltage - vp_V) / 0.2),
    )
    current = electron_current + ion_current
    vf_V = vp_V + te_eV * np.log(-ion_current_A / ies_A)
    truth = {
        "te_eV": te_eV,
        "vp_V": vp_V,
        "vf_V": vf_V,
        "ies_A": ies_A,
        "iis_A": -ion_current_A,
        "n_e_m3": density_m3,
    }
    return voltage, current, truth


def test_ideal_trace_recovers_physical_parameters():
    voltage, current, truth = _ideal_trace()
    result = analyze_iv_trace(voltage, current, _config())

    assert result["ok"] is True
    assert result["te_region_method"] == "linear_region"
    assert result["warnings"] == []
    assert result["te_fit_valid"] is True
    assert result["te_fit_passed_r2"] is True
    assert result["te_eV"] == pytest.approx(truth["te_eV"], rel=0.03)
    assert result["vp_V"] == pytest.approx(truth["vp_V"], abs=0.2)
    assert result["vf_V"] == pytest.approx(truth["vf_V"], abs=0.1)
    assert result["ies_A"] == pytest.approx(truth["ies_A"], rel=0.03)
    assert result["iis_A"] == pytest.approx(truth["iis_A"], rel=0.03)
    assert result["n_e_m3"] == pytest.approx(truth["n_e_m3"], rel=0.05)


def test_ies_method_selects_high_bias_median_or_electron_current_at_vp():
    voltage, current, truth = _ideal_trace()
    ideal_electron_current = truth["ies_A"] * np.where(
        voltage <= truth["vp_V"],
        np.exp((voltage - truth["vp_V"]) / truth["te_eV"]),
        1.0,
    )
    ion_current = current - ideal_electron_current
    expanding_sheath_current = np.where(
        voltage <= truth["vp_V"],
        ideal_electron_current,
        truth["ies_A"] * (1.0 + 0.06 * (voltage - truth["vp_V"])),
    )
    measured_current = ion_current + expanding_sheath_current

    high_bias = analyze_iv_trace(
        voltage,
        measured_current,
        _config(ies_method="high_bias_median"),
        include_diagnostic_data=True,
    )
    at_vp = analyze_iv_trace(
        voltage,
        measured_current,
        _config(ies_method="at_vp"),
        include_diagnostic_data=True,
    )

    assert high_bias["ok"] and at_vp["ok"]
    assert high_bias["ies_method"] == "high_bias_median"
    assert at_vp["ies_method"] == "at_vp"

    high_bias_data = high_bias["diagnostic_data"]
    cutoff = high_bias["vp_derivative_V"] + 0.5 * (
        high_bias_data["V_binned"][-1] - high_bias["vp_derivative_V"]
    )
    expected_high_bias = np.median(
        high_bias_data["I_binned"][high_bias_data["V_binned"] >= cutoff]
    )
    assert high_bias["ies_A"] == pytest.approx(expected_high_bias)

    at_vp_data = at_vp["diagnostic_data"]
    expected_at_vp = np.interp(
        at_vp["vp_derivative_V"],
        at_vp_data["V_binned"],
        at_vp_data["I_binned"] - at_vp["ion_current_A"],
    )
    assert at_vp["ies_A"] == pytest.approx(expected_at_vp)
    assert high_bias["ies_A"] > 1.3 * at_vp["ies_A"]


def test_ion_background_does_not_change_electron_solution():
    voltage_a, current_a, _ = _ideal_trace(ion_current_A=-1e-4)
    voltage_b, current_b, _ = _ideal_trace(ion_current_A=-5e-4)
    result_a = analyze_iv_trace(voltage_a, current_a, _config())
    result_b = analyze_iv_trace(voltage_b, current_b, _config())

    assert result_a["ok"] and result_b["ok"]
    assert result_a["te_eV"] == pytest.approx(result_b["te_eV"], rel=0.02)
    assert result_a["ies_A"] == pytest.approx(result_b["ies_A"], rel=0.02)
    assert result_a["n_e_m3"] == pytest.approx(result_b["n_e_m3"], rel=0.03)
    assert result_a["vp_V"] == pytest.approx(result_b["vp_V"], abs=0.2)
    assert result_a["vf_V"] != pytest.approx(result_b["vf_V"], abs=0.2)


def test_diagnostic_grid_resolution_does_not_change_physics():
    voltage, current, _ = _ideal_trace()
    low_resolution = analyze_iv_trace(voltage, current, _config(iv_npts=256))
    high_resolution = analyze_iv_trace(voltage, current, _config(iv_npts=2048))

    assert low_resolution["ok"] and high_resolution["ok"]
    for key in ("te_eV", "ies_A", "n_e_m3"):
        assert low_resolution[key] == pytest.approx(high_resolution[key], rel=0.01)
    assert low_resolution["vp_V"] == pytest.approx(high_resolution["vp_V"], abs=0.2)
    assert len(low_resolution["iv_voltage_grid"]) == 256
    assert len(high_resolution["iv_voltage_grid"]) == 2048


def test_trace_without_zero_crossing_is_rejected():
    voltage, current, _ = _ideal_trace()
    current = current - np.min(current) + 1e-4
    result = analyze_iv_trace(voltage, current, _config())

    assert result["ok"] is False
    assert np.isnan(result["vf_V"])
    assert np.isnan(result["te_eV"])
    assert np.isnan(result["n_e_m3"])
    assert any("no bracketed" in warning for warning in result["warnings"])


def test_endpoint_derivative_peak_is_not_a_plasma_potential():
    voltage = np.linspace(-20.0, 18.0, 1901)
    current = 2e-4 * np.expm1((voltage - 2.0) / 3.0)
    result = analyze_iv_trace(voltage, current, _config())

    assert result["ok"] is False
    assert np.isnan(result["vp_V"])
    assert np.isnan(result["te_eV"])
    assert any(
        "no interior plasma-potential" in warning for warning in result["warnings"]
    )


def test_failed_r2_gate_reports_fit_but_marks_it_rejected():
    voltage, current, truth = _ideal_trace()
    electron_current = truth["ies_A"] * np.where(
        voltage <= truth["vp_V"],
        np.exp((voltage - truth["vp_V"]) / truth["te_eV"]),
        1.0,
    )
    ion_current = current - electron_current
    modulation = 0.25 * np.sin(2 * np.pi * (voltage - truth["vp_V"]) / 2.5)
    distorted_electron_current = np.where(
        voltage <= truth["vp_V"],
        electron_current * np.exp(modulation),
        truth["ies_A"],
    )
    result = analyze_iv_trace(
        voltage,
        distorted_electron_current + ion_current,
        _config(te_min_r2=0.999),
    )

    assert result["ok"] is False
    assert result["te_fit_passed_r2"] is False
    assert np.isfinite(result["te_eV"])
    assert np.isfinite(result["vp_V"])
    assert np.isfinite(result["n_e_m3"])


def test_density_value_and_equivalent_units():
    expected = 5.384535228598626e15
    from_floats = density_from_electron_saturation_current(1e-3, 3.0, 4e-6)
    from_quantities = density_from_electron_saturation_current(
        1 * u.mA,
        3 * u.eV,
        4 * u.mm**2,
    )
    kelvin = (3 * u.eV / k_B).to(u.K)
    from_kelvin = density_from_electron_saturation_current(
        1 * u.mA,
        kelvin,
        4 * u.mm**2,
    )

    assert from_floats.unit.is_equivalent(u.m**-3)
    assert from_floats.value == pytest.approx(expected, rel=1e-12)
    assert from_quantities.value == pytest.approx(expected, rel=1e-12)
    assert from_kelvin.value == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("voltage", "current", "config"),
    [
        (np.arange(10.0), np.arange(9.0), _config()),
        (np.ones((2, 5)), np.ones((2, 5)), _config()),
        (np.arange(10.0), np.arange(10.0), _config(voltage_bin_width=0)),
        (np.arange(10.0), np.arange(10.0), _config(unknown_option=True)),
        (np.arange(10.0), np.arange(10.0), _config(ies_method="unknown")),
        (np.arange(10.0), np.arange(10.0), _config(probe_area=-1.0)),
        (np.arange(10.0), np.arange(10.0), _config(probe_area=4 * u.mm)),
    ],
)
def test_invalid_inputs_raise_value_error(voltage, current, config):
    with pytest.raises(ValueError):
        analyze_iv_trace(voltage, current, config)


def test_nonfinite_samples_are_dropped_when_enough_data_remain():
    voltage, current, _ = _ideal_trace()
    voltage[100] = np.nan
    current[200] = np.nan
    result = analyze_iv_trace(voltage, current, _config())
    assert result["ok"] is True


def test_nonpositive_saturation_current_is_not_silently_flipped():
    with pytest.raises(ValueError, match="I_esat"):
        density_from_electron_saturation_current(-1e-3, 3.0, 4e-6)


def test_reversed_current_polarity_is_rejected():
    voltage, current, _ = _ideal_trace()
    result = analyze_iv_trace(voltage, -current, _config())
    assert result["ok"] is False
    assert np.isnan(result["te_eV"])


def test_up_and_down_sweeps_are_not_mixed_by_sorting():
    voltage, current, _ = _ideal_trace(n_samples=951)
    full_voltage = np.concatenate((voltage, voltage[::-1]))
    full_current = np.concatenate((current, current[::-1]))
    result = analyze_iv_trace(full_voltage, full_current, _config())

    assert result["ok"] is False
    assert any("dominant sweep direction" in warning for warning in result["warnings"])


def test_partial_return_sweep_is_rejected():
    voltage, current, _ = _ideal_trace(n_samples=951)
    full_voltage = np.concatenate((voltage, voltage[-2:-302:-1]))
    full_current = np.concatenate((current, current[-2:-302:-1] - 2e-4))
    result = analyze_iv_trace(full_voltage, full_current, _config())

    assert result["ok"] is False
    assert any("dominant sweep direction" in warning for warning in result["warnings"])


def test_descending_quantity_sweep_matches_forward_float_sweep():
    voltage, current, _ = _ideal_trace()
    forward = analyze_iv_trace(voltage, current, _config())
    descending = analyze_iv_trace(
        (1000 * voltage[::-1]) * u.mV,
        (1000 * current[::-1]) * u.mA,
        _config(),
    )

    assert forward["ok"] and descending["ok"]
    for key in ("te_eV", "vp_V", "vf_V", "ies_A", "iis_A", "n_e_m3"):
        assert descending[key] == pytest.approx(forward[key], rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("method", ["bounded", "linear_region"])
def test_sloped_ion_branch_respects_region_and_ideal_model_checks(method):
    voltage, current, _ = _ideal_trace()
    drift = 15e-6 * np.clip(voltage - voltage[0], 0, 8)
    relaxed = analyze_iv_trace(
        voltage, current + drift, _config(te_region_method=method),
    )
    strict = analyze_iv_trace(
        voltage,
        current + drift,
        _config(te_region_method=method, enforce_ideal_model_checks=True),
    )

    assert relaxed["ok"] == (method == "bounded")
    if method == "linear_region":
        assert any("no sufficiently linear" in warning for warning in relaxed["warnings"])
    assert any("sloped" in note for note in relaxed["model_notes"])
    assert strict["ok"] is False
    assert any("ion-saturation plateau" in warning for warning in strict["warnings"])


def test_inadequate_negative_bias_coverage_is_rejected():
    voltage, current, _ = _ideal_trace()
    keep = voltage >= -2.0
    result = analyze_iv_trace(
        voltage[keep],
        current[keep],
        _config(enforce_ideal_model_checks=True),
    )

    assert result["ok"] is False
    assert any("negative-bias coverage" in warning for warning in result["warnings"])


def test_missing_voltage_support_near_knee_is_rejected():
    voltage, current, _ = _ideal_trace()
    keep = (voltage < 8.0) | (voltage > 12.0)
    result = analyze_iv_trace(voltage[keep], current[keep], _config())

    assert result["ok"] is False
    assert any("continuously support" in warning for warning in result["warnings"])


def test_absolute_current_zero_must_be_explicitly_calibrated():
    voltage, current, _ = _ideal_trace()
    result = analyze_iv_trace(
        voltage,
        current,
        _config(current_zero_calibrated=False),
        include_diagnostic_data=True,
    )

    assert result["ok"] is False
    assert result["diagnostic_data"] is not None
    assert any(
        "not independently calibrated" in warning for warning in result["warnings"]
    )


def test_moderate_noise_recovers_solution_but_excessive_noise_rejects_fit():
    voltage, current, truth = _ideal_trace()
    rng = np.random.default_rng(20260809)
    moderate_noise = rng.normal(0.0, 50e-6, voltage.size)
    excessive_noise = rng.normal(0.0, 200e-6, voltage.size)

    recovered = analyze_iv_trace(voltage, current + moderate_noise, _config())
    rejected = analyze_iv_trace(voltage, current + excessive_noise, _config())

    assert recovered["ok"] is True
    assert recovered["te_eV"] == pytest.approx(truth["te_eV"], rel=0.05)
    assert recovered["vp_V"] == pytest.approx(truth["vp_V"], abs=0.2)
    assert recovered["ies_A"] == pytest.approx(truth["ies_A"], rel=0.04)
    assert recovered["n_e_m3"] == pytest.approx(truth["n_e_m3"], rel=0.06)
    assert rejected["ok"] is False
    assert np.isfinite(rejected["te_eV"])
    assert any("no sufficiently linear" in warning for warning in rejected["warnings"])


def test_unresolved_regional_ion_estimate_is_rejected():
    voltage, current, _ = _ideal_trace(ion_current_A=-2e-5)
    noise = np.random.default_rng(77).normal(0.0, 200e-6, voltage.size)
    result = analyze_iv_trace(voltage, current + noise, _config())

    assert result["ok"] is False
    assert result["ion_current_snr"] < 3.0
    assert any("ion-saturation current estimate" in warning for warning in result["warnings"])


@pytest.mark.parametrize("method", ["bounded", "linear_region"])
def test_high_r2_bimaxwellian_branch_respects_region_and_curvature_checks(method):
    voltage, current, truth = _ideal_trace()
    ideal_electron_current = truth["ies_A"] * np.where(
        voltage <= truth["vp_V"],
        np.exp((voltage - truth["vp_V"]) / truth["te_eV"]),
        1.0,
    )
    ion_current = current - ideal_electron_current
    cold_fraction = 0.7
    bimaxwellian_current = truth["ies_A"] * (
        cold_fraction
        * np.where(
            voltage <= truth["vp_V"],
            np.exp((voltage - truth["vp_V"]) / 1.5),
            1.0,
        )
        + (1.0 - cold_fraction)
        * np.where(
            voltage <= truth["vp_V"],
            np.exp((voltage - truth["vp_V"]) / 6.0),
            1.0,
        )
    )

    relaxed = analyze_iv_trace(
        voltage, ion_current + bimaxwellian_current, _config(te_region_method=method),
    )
    result = analyze_iv_trace(
        voltage,
        ion_current + bimaxwellian_current,
        _config(te_region_method=method, enforce_ideal_model_checks=True),
    )

    assert relaxed["ok"] == (method == "bounded")
    assert any("curvature" in note for note in relaxed["model_notes"])
    assert result["te_fit_r2"] > 0.98
    assert result["ok"] is False
    assert np.isfinite(result["te_eV"])
    assert np.isfinite(result["vp_V"])
    assert np.isfinite(result["n_e_m3"])
    reason = "curvature" if method == "bounded" else "no sufficiently linear"
    assert any(reason in warning for warning in result["warnings"])


@pytest.mark.parametrize("method", ["bounded", "linear_region"])
@pytest.mark.parametrize("outlier_current", [1e-6, 1.2, np.nan])
def test_te_fit_region_does_not_bridge_current_threshold_excursions(outlier_current, method):
    voltage = np.linspace(-10.0, -1.0, 181)
    electron_current = np.exp(voltage / 3.0)
    electron_current[60] = outlier_current
    result = _fit_electron_temperature(
        voltage, electron_current, 1.0, -10.0, 0.0, 0.0,
        _validated_config(_config(te_region_method=method)),
    )

    mask = result["fit_mask"]
    assert not mask[60]
    selected = np.flatnonzero(mask)
    assert np.all(np.diff(selected) == 1)
    assert result["valid"]
    assert result["te_eV_candidate"] == pytest.approx(3.0, rel=1e-10)


def test_disconnected_short_intervals_cannot_satisfy_minimum_fit_points():
    voltage = np.linspace(-10.0, -1.0, 181)
    electron_current = np.exp(voltage / 3.0)
    electron_current[::6] = 1.2
    result = _fit_electron_temperature(
        voltage, electron_current, 1.0, -10.0, 0.0, 0.0,
        _validated_config(_config(te_min_points=12)),
    )

    assert result["npts"] < 12
    assert not result["valid"]
    assert np.isnan(result["te_eV_candidate"])
    assert "no sufficiently linear" in result["reason"]


@pytest.mark.parametrize("method", ["bounded", "linear_region"])
@pytest.mark.parametrize(
    ("floor_frac", "ion_noise", "vf", "margin"),
    [(0.03, 0.0, -20.0, 0.2), (0.2, 0.0, -20.0, 0.2),
     (0.03, 0.03, -20.0, 0.2), (0.03, 0.0, -6.0, 2.0)],
)
def test_te_fit_obeys_both_voltage_and_current_bounds(floor_frac, ion_noise, vf, margin, method):
    voltage = np.linspace(-20.0, 5.0, 501)
    electron_current = np.exp(np.minimum(voltage, 0.0) / 3.0)
    result = _fit_electron_temperature(
        voltage, electron_current, 1.0, vf, 0.0, ion_noise,
        _validated_config(_config(te_region_method=method,
                                  te_current_floor_frac=floor_frac,
                                  te_margin_from_vp=margin)),
    )
    mask = result["fit_mask"]
    assert result["valid"]
    assert np.all(voltage[mask] >= vf)
    assert np.all(voltage[mask] <= -margin)
    assert np.all(electron_current[mask] >= max(floor_frac, 5 * ion_noise))
    assert np.all(electron_current[mask] <= 0.8)
    assert result["vstart"] == voltage[mask][0]
    assert result["vstop"] == voltage[mask][-1]
    assert result["te_eV_candidate"] == pytest.approx(3.0, rel=1e-10)


@pytest.mark.parametrize("temperature", [0.5, 3.0, 20.0])
def test_known_temperature_is_recovered_with_selected_retarding_bounds(temperature):
    vp = 10.0
    voltage = np.linspace(vp - 12 * temperature, vp + 4 * temperature, 6401)
    electron_current = 0.01 * np.exp(np.minimum(voltage - vp, 0) / temperature)
    result = analyze_iv_trace(
        voltage, electron_current - 0.0002,
        _config(voltage_bin_width=temperature / 60,
                vp_smoothing_width_V=temperature / 3,
                te_margin_from_vp=temperature / 10,
                ies_method="at_vp"),
        include_diagnostic_data=True,
    )
    assert result["ok"]
    assert result["te_eV"] == pytest.approx(temperature, rel=0.01)
    assert result["te_fit_vstart"] >= result["vf_V"]
    assert result["te_fit_vstop"] <= result["vp_derivative_V"] - temperature / 10


def test_diagnostic_marks_measured_fit_bins_and_both_limits():
    import matplotlib.pyplot as plt

    from langmuir_diagnostics import plot_iv_diagnostic_panels

    voltage, current, _ = _ideal_trace()
    result = analyze_iv_trace(
        voltage, current, _config(iv_npts=7), include_diagnostic_data=True,
    )
    data = result["diagnostic_data"]
    mask = data["te_fit_mask"]
    assert mask.sum() == result["te_fit_npts"]
    assert data["V_binned"][mask][0] == result["te_fit_vstart"]
    assert data["V_binned"][mask][-1] == result["te_fit_vstop"]

    fig, axs = plt.subplots(3)
    try:
        plot_iv_diagnostic_panels(result, axs)
        lines = {line.get_label(): line for line in axs[2].lines}
        measured = lines["measured Te fit bins"]
        np.testing.assert_array_equal(measured.get_xdata(), data["V_binned"][mask])
        np.testing.assert_allclose(
            measured.get_ydata(), np.log(data["I_binned"][mask] - result["ion_current_A"]),
        )
        np.testing.assert_array_equal(
            lines["Te lower limit"].get_xdata(), [result["te_fit_vstart"]] * 2,
        )
        np.testing.assert_array_equal(
            lines["Te upper limit"].get_xdata(), [result["te_fit_vstop"]] * 2,
        )
    finally:
        plt.close(fig)


def test_linear_region_moves_both_limits_and_rejects_cancelling_curvature():
    voltage = np.linspace(-10.0, 4.0, 281)
    centered_voltage = voltage + 3
    log_current = voltage / 3.0 + (
        0.005 * np.sign(centered_voltage)
        * np.maximum(np.abs(centered_voltage) - 3, 0) ** 3
    )
    cfg = _validated_config(_config(te_region_method="bounded",
                                    te_current_floor_frac=0, te_min_r2=0.99))
    bounded = _fit_electron_temperature(
        voltage, np.exp(log_current), 10, -10, 5, 0, cfg,
    )
    selected = _fit_electron_temperature(
        voltage, np.exp(log_current), 10, -10, 5, 0,
        {**cfg, "te_region_method": "linear_region"},
    )

    # Both tails curve, with opposite signs: R² and a quadratic alone miss it.
    assert bounded["r2"] > 0.99
    assert bounded["curvature_log"] < 0.05
    assert selected["valid"]
    assert selected["vstart"] > bounded["vstart"]
    assert selected["vstop"] < bounded["vstop"]
    assert selected["te_eV_candidate"] == pytest.approx(3.0, rel=0.03)
    assert selected["block_deviation_log"] <= 0.02
    assert selected["log_span"] >= 1.0
    assert selected["candidate_count"] > 1


@pytest.mark.parametrize("temperature", [0.5, 3.0, 20.0])
def test_linear_region_preserves_true_temperature_without_targeting_acceptance(temperature):
    voltage = np.linspace(-3 * temperature, -0.3 * temperature, 181)
    electron_current = np.exp(voltage / temperature)
    result = _fit_electron_temperature(
        voltage, electron_current, 1, -4 * temperature, 0, 0,
        _validated_config(_config(te_region_method="linear_region", te_max_eV=10,
                                  te_margin_from_vp=0)),
    )
    assert result["te_eV_candidate"] == pytest.approx(temperature, rel=1e-10)
    assert result["vstart"] == voltage[0]
    assert result["vstop"] == voltage[-1]
    assert result["valid"] == (temperature <= 10)
    if temperature > 10:
        assert "physical range" in result["reason"]


def test_linear_region_rejects_short_straight_segment_without_an_efold():
    voltage = np.linspace(0.0, 1.0, 31)
    result = _fit_electron_temperature(
        voltage, np.exp(voltage / 2), 3, 0, 2, 0,
        _validated_config(_config(te_region_method="linear_region")),
    )
    assert result["r2"] == pytest.approx(1.0)
    assert not result["valid"]
    assert "no sufficiently linear" in result["reason"]
