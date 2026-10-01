"""HDF5 metadata helpers shared by the GUI and analysis worker."""

from __future__ import annotations

import math
from pathlib import Path


def _decode_hdf5_text(value):
    """Return one HDF5 scalar string as ordinary display text."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip("\x00 ")


def _sis_data_type_attribute(adc, channel):
    """Return the SIS configuration attribute for one physical channel."""
    adc = str(adc).strip()
    channel = int(channel)
    if adc == "SIS 3302":
        return f"Data type {channel}"
    if adc == "SIS 3305":
        fpga = 1 if channel <= 4 else 2
        fpga_channel = channel if fpga == 1 else channel - 4
        return f"FPGA {fpga} Data type {fpga_channel}"
    raise ValueError(
        f'SIS crate data-type labels are unavailable for ADC "{adc}".'
    )


def read_sis_channel_data_types(
    filename,
    *,
    digitizer,
    adc,
    config_name,
    board,
    channels,
):
    """Return the configured ``Data type`` text for SIS crate channels."""
    filename = Path(filename).expanduser()
    if not filename.is_file():
        raise ValueError(f"Experiment HDF5 file does not exist: {filename}")

    digitizer = str(digitizer).strip()
    adc = str(adc).strip()
    config_name = str(config_name).strip()
    board = int(board)
    channels = tuple(int(channel) for channel in channels)

    file_obj = None
    try:
        # bapsflib supplies the authoritative relationship between an SIS
        # board number and its physical crate slot. The per-channel Data type
        # strings themselves are retained only in the raw configuration group.
        from bapsflib import lapd

        file_obj = lapd.File(filename, silent=True)
        if digitizer not in file_obj.digitizers:
            raise ValueError(
                f'Digitizer "{digitizer}" is not mapped in {filename}.'
            )
        digitizer_map = file_obj.digitizers[digitizer]
        if config_name not in digitizer_map.configs:
            raise ValueError(
                f'SIS configuration "{config_name}" is not mapped in {filename}.'
            )

        slot = digitizer_map.get_slot(board, adc)
        if slot is None:
            raise ValueError(
                f'Board {board} is not available for ADC "{adc}".'
            )

        config = digitizer_map.configs[config_name]
        config_group = file_obj[config["config group path"]]
        slots = tuple(config_group.attrs["SIS crate slot numbers"])
        indices = tuple(config_group.attrs["SIS crate config indices"])
        if len(slots) != len(indices):
            raise ValueError(
                "SIS crate slot and configuration-index metadata have "
                "different lengths."
            )
        matching_indices = [
            int(index)
            for candidate_slot, index in zip(slots, indices, strict=True)
            if int(candidate_slot) == int(slot)
        ]
        if len(matching_indices) != 1:
            raise ValueError(
                f"SIS crate slot {slot} does not identify exactly one configuration."
            )

        adc_number = adc.removeprefix("SIS ")
        adc_group_name = (
            f"SIS crate {adc_number} configurations[{matching_indices[0]}]"
        )
        if adc_group_name not in config_group:
            raise ValueError(
                f'SIS ADC configuration group "{adc_group_name}" was not found.'
            )
        adc_group = config_group[adc_group_name]

        data_types = {}
        for channel in channels:
            attribute = _sis_data_type_attribute(adc, channel)
            if attribute not in adc_group.attrs:
                raise ValueError(
                    f'SIS channel {channel} has no "Data type" metadata.'
                )
            data_types[channel] = _decode_hdf5_text(adc_group.attrs[attribute])
        return data_types
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            f"Could not read SIS channel data types from {filename}: {error}"
        ) from error
    finally:
        if file_obj is not None:
            file_obj.close()


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
