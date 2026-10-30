"""Independent injection/recovery checks at BaPSF-like density and scale."""

from dataclasses import replace

import numpy as np
import pytest

from langmuir_analysis_core import analyze_iv_trace
from synthetic_langmuir_validation import SyntheticPlasma, make_trace


def _analyze(plasma, **overrides):
    voltage, current, _ = make_trace(plasma)
    return analyze_iv_trace(
        voltage,
        current,
        {"probe_area": plasma.area_m2, "current_zero_calibrated": True, **overrides},
    )


def test_baseline_matches_requested_magnetized_planar_scale():
    plasma = SyntheticPlasma()
    voltage, current, truth = make_trace(plasma)
    assert truth["ne_m3"] == 5e18
    assert truth["debye_m"] == pytest.approx(5.7583e-6, rel=1e-4)
    assert truth["ion_gyroradius_m"] == pytest.approx(2.04e-3, rel=0.01)
    assert np.sqrt(plasma.area_m2) / truth["debye_m"] > 300
    np.testing.assert_allclose(current, truth["electron_A"] + truth["ion_A"])
    assert np.all(np.diff(voltage) > 0)
    result = _analyze(plasma)
    assert result["ok"]
    assert result["te_eV"] == pytest.approx(3, rel=0.001)
    assert result["n_e_m3"] == pytest.approx(5e18, rel=0.001)
    assert result["vp_V"] == pytest.approx(10, abs=0.03)


@pytest.mark.parametrize("ne_cm3", [1e11, 5e12, 1e13])
def test_density_scaling_preserves_temperature(ne_cm3):
    result = _analyze(replace(SyntheticPlasma(), ne_cm3=ne_cm3))
    assert result["ok"]
    assert result["te_eV"] == pytest.approx(3, rel=0.001)
    assert result["n_e_m3"] == pytest.approx(ne_cm3 * 1e6, rel=0.001)


def test_at_vp_detects_self_consistent_but_smoothing_biased_density():
    plasma = SyntheticPlasma()
    broad = _analyze(plasma, ies_method="at_vp")
    narrow = _analyze(plasma, ies_method="at_vp", vp_smoothing_width_V=0.25)
    strict = _analyze(plasma, ies_method="at_vp", enforce_ideal_model_checks=True)
    assert broad["ok"] and narrow["ok"]
    assert broad["te_eV"] == pytest.approx(3, rel=0.001)
    assert broad["n_e_m3"] / 5e18 < 0.8
    # The original extrapolation check misses the error because it uses I(Vp).
    assert abs(broad["vp_fit_V"] - broad["vp_derivative_V"]) < 0.02
    assert any("nearly flat high-bias" in note for note in broad["model_notes"])
    assert not any("nearly flat high-bias" in note for note in narrow["model_notes"])
    assert narrow["n_e_m3"] == pytest.approx(5e18, rel=0.05)
    assert not strict["ok"]
    assert any("nearly flat high-bias" in warning for warning in strict["warnings"])
    assert np.isfinite(strict["te_eV"]) and np.isfinite(strict["n_e_m3"])
    # The diagnostic must not silently substitute a different Ies definition.
    assert strict["ies_A"] == pytest.approx(broad["ies_A"])


def test_independent_plateau_check_does_not_assume_a_sloped_branch_is_flat():
    result = _analyze(
        replace(SyntheticPlasma(), electron_growth_per_V=0.06), ies_method="at_vp"
    )
    assert result["ok"]
    assert not any("nearly flat high-bias" in note for note in result["model_notes"])


def test_unknown_electron_collection_reduces_density_without_changing_temperature():
    result = _analyze(replace(SyntheticPlasma(), electron_collection=0.5))
    assert result["ok"]
    assert result["te_eV"] == pytest.approx(3, rel=0.001)
    assert result["n_e_m3"] == pytest.approx(0.5 * 5e18, rel=0.001)


def test_seeded_noise_is_reproducible_and_accepted():
    plasma = replace(SyntheticPlasma(), noise_fraction=0.01)
    _, first, _ = make_trace(plasma, seed=7)
    _, repeat, _ = make_trace(plasma, seed=7)
    _, different, _ = make_trace(plasma, seed=8)
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, different)
    result = _analyze(plasma)
    assert result["ok"]
    assert result["te_eV"] == pytest.approx(3, rel=0.05)
    assert result["n_e_m3"] == pytest.approx(5e18, rel=0.05)
