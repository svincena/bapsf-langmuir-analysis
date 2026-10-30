"""Beam flux and explicitly invoked exponential/background decomposition."""

import numpy as np
import pytest
from scipy.constants import e, m_e
from scipy.integrate import quad

from langmuir_analysis_core import (
    analyze_iv_trace,
    fit_exponential_with_linear_background,
    screen_for_hot_tail,
)
from synthetic_langmuir_beam_validation import (
    drifting_electron_current,
    make_beam_trace,
    scan_bulk_windows,
)


@pytest.mark.parametrize("temperature", [1.0, 3.0, 12.0, 20.0])
def test_background_decomposition_recovers_temperature_without_targeting_bulk(
    temperature,
):
    voltage = np.linspace(10 - 5 * temperature, 10, 301)
    x = voltage - voltage[-1]
    current = 0.8 * np.exp(x / temperature) + 0.1 + 0.001 * x
    result = fit_exponential_with_linear_background(
        voltage, current, temperature_bounds=(0.1, 300)
    )
    assert result["valid"]
    assert result["te_eV"] == pytest.approx(temperature, rel=1e-6)
    assert result["background_at_vmax_A"] == pytest.approx(0.1, abs=1e-7)
    assert result["background_slope_A_per_V"] == pytest.approx(0.001, abs=1e-8)
    np.testing.assert_allclose(result["fitted_current_A"], current, atol=1e-7)


def test_short_nearly_linear_window_does_not_identify_exponential_temperature():
    voltage = np.linspace(0, 0.1, 41)
    result = fit_exponential_with_linear_background(
        voltage,
        0.8 * np.exp((voltage - 0.1) / 20) + 0.1,
        temperature_bounds=(0.1, 300),
    )
    assert not result["valid"]


def test_background_fit_does_not_force_hot_component_below_temperature_bound():
    voltage = np.linspace(-100, 0, 401)
    result = fit_exponential_with_linear_background(
        voltage, np.exp(voltage / 40) + 0.1, temperature_bounds=(0.1, 10)
    )
    assert not result["valid"]


@pytest.mark.parametrize("bounds", [(0, 10), (10, 1), (0.1, np.inf)])
def test_background_fit_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="temperature_bounds"):
        fit_exponential_with_linear_background(
            np.arange(10.0), np.arange(10.0), temperature_bounds=bounds
        )


def test_zero_drift_beam_flux_reduces_to_planar_maxwellian():
    voltage = np.linspace(-50, 20, 201)
    temperature, density, area, vp = 20, 5e16, 4e-6, 10
    current = drifting_electron_current(
        voltage,
        density_m3=density,
        drift_energy_eV=0,
        spread_eV=temperature,
        vp_V=vp,
        area_m2=area,
    )
    expected = e * area * density * np.sqrt(e * temperature / (2 * np.pi * m_e))
    expected *= np.exp(np.minimum((voltage - vp) / temperature, 0))
    np.testing.assert_allclose(current, expected, rtol=1e-13)


def test_directed_beam_flux_matches_independent_velocity_quadrature():
    voltage = np.array([-70, -25, 5, 15])
    density, area, vp, drift_energy, temperature = 5e16, 4e-6, 10, 35, 2
    drift = np.sqrt(2 * e * drift_energy / m_e)
    sigma = np.sqrt(e * temperature / m_e)
    expected = []
    for bias in voltage:
        threshold = np.sqrt(2 * e * max(vp - bias, 0) / m_e)
        lower = (threshold - drift) / sigma
        # Integrate in standardized velocity to resolve the narrow warm beam.
        flux, _ = quad(
            lambda z: (drift + sigma * z) * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi),
            lower,
            np.inf,
        )
        expected.append(e * area * density * flux)
    actual = drifting_electron_current(
        voltage,
        density_m3=density,
        drift_energy_eV=drift_energy,
        spread_eV=temperature,
        vp_V=vp,
        area_m2=area,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("noise", [0, 0.01])
def test_low_density_beam_can_mimic_twenty_ev_while_bulk_is_recoverable(noise):
    voltage, current, _ = make_beam_trace(noise_fraction=noise)
    ordinary = analyze_iv_trace(
        voltage,
        current,
        {"probe_area": 4e-6, "current_zero_calibrated": True},
        include_diagnostic_data=True,
    )
    assert 19 < ordinary["te_eV"] < 22
    assert not ordinary["ok"]
    scans, bulk = scan_bulk_windows(ordinary, temperature_bounds=(0.1, 300))
    assert bulk is not None
    assert bulk["te_eV"] == pytest.approx(3, rel=0.05)
    # A smaller supported interval isolates a more nearly linear beam background.
    if noise == 0:
        local = next(fit for fit in scans if fit["width_V"] == 12)
        assert local["te_eV"] == pytest.approx(3, rel=0.005)
    assert bulk["background_at_vmax_A"] > ordinary["ion_current_A"] + 0.05


def test_beam_free_control_does_not_report_extra_electron_background():
    voltage, current, _ = make_beam_trace(fraction=0)
    ordinary = analyze_iv_trace(
        voltage,
        current,
        {"probe_area": 4e-6, "current_zero_calibrated": True},
        include_diagnostic_data=True,
    )
    _, bulk = scan_bulk_windows(ordinary, temperature_bounds=(0.1, 300))
    assert bulk is not None
    assert bulk["te_eV"] == pytest.approx(3, rel=0.001)
    assert bulk["background_at_vmax_A"] == pytest.approx(
        ordinary["ion_current_A"], abs=1e-6
    )


def test_explicit_hot_tail_diagnostic_recovers_bulk_without_changing_fit():
    voltage, current, _ = make_beam_trace(fraction=0.05, drift_eV=0, spread_eV=20)
    result = analyze_iv_trace(
        voltage,
        current,
        {"probe_area": 4e-6, "current_zero_calibrated": True},
        include_diagnostic_data=True,
    )
    ordinary_temperature = result["te_eV"]
    check = screen_for_hot_tail(result, minimum_current_fraction=0.05)
    assert check["status"] == "possible"
    assert check["bulk_candidate_te_eV"] == pytest.approx(3, rel=0.1)
    assert result["te_eV"] == ordinary_temperature
    assert (
        screen_for_hot_tail(result, minimum_current_fraction=0.5)["status"]
        == "not_resolved"
    )


def test_explicit_hot_tail_check_is_inconclusive_without_supported_diagnostics():
    voltage, current, _ = make_beam_trace()
    result = analyze_iv_trace(
        voltage, current, {"probe_area": 4e-6, "current_zero_calibrated": True}
    )
    assert (
        screen_for_hot_tail(result, minimum_current_fraction=0.05)["status"]
        == "inconclusive"
    )
