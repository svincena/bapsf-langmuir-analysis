import numpy as np
import pytest

from read_langmuir_xy_npz import (
    EXPECTED_LANGMUIR_XY_NPZ_KEYS,
    validate_langmuir_xy_npz,
)


def _minimal_schema():
    return {key: np.array(0) for key in EXPECTED_LANGMUIR_XY_NPZ_KEYS}


def test_xy_npz_schema_allows_forward_compatible_diagnostics():
    data = _minimal_schema()
    data["future_quality_metric"] = np.array(1.0)

    assert validate_langmuir_xy_npz(data) is data


def test_xy_npz_schema_still_rejects_missing_required_key():
    data = _minimal_schema()
    del data["te_eV"]

    with pytest.raises(ValueError, match="missing keys.*te_eV"):
        validate_langmuir_xy_npz(data)
