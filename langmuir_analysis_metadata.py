"""HDF5 metadata helpers shared by the GUI and analysis worker."""

from __future__ import annotations

import math
from pathlib import Path


def _validated_trace_length(specs, role):
    """Return a positive integer trace length from digitizer specifications."""
    nt_value = specs.get("nt")
    try:
        nt_full = int(nt_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"The {role} channel has no valid digitizer trace length."
        ) from error
    if isinstance(nt_value, bool) or nt_full < 2 or nt_full != nt_value:
        raise ValueError(
            f"The {role} channel has invalid digitizer trace length {nt_value!r}."
        )
    return nt_full


def _sample_interval_from_specs(file_obj, specs, role):
    """Return a scalar sample interval in seconds for one digitizer channel."""
    import astropy.units as u
    import numpy as np

    clock_rate = specs.get("clock rate")
    if hasattr(clock_rate, "to_value"):
        try:
            sample_average = specs.get("sample average")
            sample_average = 1.0 if sample_average is None else float(sample_average)
            dt_s = sample_average / clock_rate.to_value(u.Hz)
        except (TypeError, ValueError, u.UnitConversionError) as error:
            raise ValueError(
                f"The {role} channel has invalid clock-rate or sample-average metadata."
            ) from error
    else:
        try:
            raw_time = np.asanyarray(file_obj.get_time_array(specs))
        except Exception as error:
            raise ValueError(
                f"The {role} channel has no usable temporal sampling metadata."
            ) from error
        if raw_time.ndim != 1 or raw_time.size < 2:
            raise ValueError(
                f"The {role} channel time array must be one-dimensional with at least two samples."
            )
        time_s = np.asarray(raw_time, dtype=np.float64)
        if not np.all(np.isfinite(time_s)):
            raise ValueError(f"The {role} channel time array contains nonfinite values.")
        dt_s = float((time_s[-1] - time_s[0]) / (time_s.size - 1))
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError(
                f"The {role} channel time array must be strictly increasing."
            )

        # Stored float32 time axes show quantization at large sample indices.
        # Compare the complete axis with a linear model using a dtype-aware
        # absolute tolerance rather than requiring identical local differences.
        expected_time = time_s[0] + np.arange(time_s.size) * dt_s
        precision = (
            np.finfo(raw_time.dtype).eps
            if np.issubdtype(raw_time.dtype, np.floating)
            else np.finfo(np.float64).eps
        )
        scale = max(float(np.max(np.abs(time_s))), abs(dt_s))
        if not np.allclose(
            time_s,
            expected_time,
            rtol=0.0,
            atol=8.0 * precision * scale,
        ):
            raise ValueError(
                f"The {role} channel time array is not evenly spaced; "
                "the analysis requires a scalar sample interval."
            )

    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError(
            f"The {role} channel has invalid sample interval {dt_s!r} s."
        )
    return float(dt_s)


def read_digitizer_temporal_metadata(
    filename,
    *,
    digitizer,
    adc,
    config_name,
    board,
    voltage_channel,
    current_channel,
):
    """Return common ``nt_full`` and ``dt_s`` for both probe channels."""
    filename = Path(filename).expanduser()
    if not filename.is_file():
        raise ValueError(f"Experiment HDF5 file does not exist: {filename}")

    file_obj = None
    try:
        from bapsflib import lapd

        file_obj = lapd.File(filename, silent=True)
        channel_metadata = {}
        for role, channel in (
            ("voltage", voltage_channel),
            ("current", current_channel),
        ):
            specs = file_obj.get_digitizer_specs(
                board,
                channel,
                digitizer=str(digitizer).strip(),
                adc=str(adc).strip(),
                config_name=str(config_name).strip(),
                silent=True,
            )
            channel_metadata[role] = (
                _validated_trace_length(specs, role),
                _sample_interval_from_specs(file_obj, specs, role),
            )

        voltage_nt, voltage_dt_s = channel_metadata["voltage"]
        current_nt, current_dt_s = channel_metadata["current"]
        if voltage_nt != current_nt:
            raise ValueError(
                "Voltage and current channels have different trace lengths: "
                f"{voltage_nt} and {current_nt}."
            )
        if not math.isclose(
            voltage_dt_s,
            current_dt_s,
            rel_tol=1e-9,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "Voltage and current channels have different sample intervals: "
                f"{voltage_dt_s:.15g} s and {current_dt_s:.15g} s."
            )
        return {"nt_full": voltage_nt, "dt_s": voltage_dt_s}
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            f"Could not read digitizer temporal metadata from {filename}: {error}"
        ) from error
    finally:
        if file_obj is not None:
            file_obj.close()
