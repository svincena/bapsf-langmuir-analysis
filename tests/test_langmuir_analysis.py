import astropy.units as u
from astropy.constants import e, k_B, m_e
import numpy as np
import pytest

from langmuir_analysis import (
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
    assert result["warnings"] == []
    assert result["te_fit_valid"] is True
    assert result["te_fit_passed_r2"] is True
    assert result["te_eV"] == pytest.approx(truth["te_eV"], rel=0.03)
    assert result["vp_V"] == pytest.approx(truth["vp_V"], abs=0.2)
    assert result["vf_V"] == pytest.approx(truth["vf_V"], abs=0.1)
    assert result["ies_A"] == pytest.approx(truth["ies_A"], rel=0.03)
    assert result["iis_A"] == pytest.approx(truth["iis_A"], rel=0.03)
    assert result["n_e_m3"] == pytest.approx(truth["n_e_m3"], rel=0.05)


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


def test_failed_r2_gate_produces_no_temperature_or_density():
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
    assert np.isnan(result["te_eV"])
    assert np.isnan(result["n_e_m3"])


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


def test_sloped_ion_branch_is_rejected():
    voltage, current, _ = _ideal_trace()
    drift = 15e-6 * np.clip(voltage - voltage[0], 0, 8)
    result = analyze_iv_trace(voltage, current + drift, _config())

    assert result["ok"] is False
    assert any("ion-saturation plateau" in warning for warning in result["warnings"])


def test_inadequate_negative_bias_coverage_is_rejected():
    voltage, current, _ = _ideal_trace()
    keep = voltage >= -2.0
    result = analyze_iv_trace(voltage[keep], current[keep], _config())

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


def test_moderate_noise_recovers_solution_but_unresolved_ion_signal_rejects():
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
    assert np.isnan(rejected["te_eV"])


def test_high_r2_bimaxwellian_branch_fails_curvature_gate():
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

    result = analyze_iv_trace(voltage, ion_current + bimaxwellian_current, _config())

    assert result["te_fit_r2"] > 0.98
    assert result["ok"] is False
    assert np.isnan(result["te_eV"])
    assert any("curvature" in warning for warning in result["warnings"])
