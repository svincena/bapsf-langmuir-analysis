import astropy.units as u
import numpy as np
import pytest

from bapsflib import lapd
from langmuir_analysis_metadata import (
    read_digitizer_temporal_metadata,
    read_sis_channel_data_types,
)


def _read_metadata(path):
    return read_digitizer_temporal_metadata(
        path,
        digitizer="SIS crate",
        adc="SIS 3302",
        config_name="Langmuir",
        board=2,
        voltage_channel=3,
        current_channel=2,
    )


def test_reads_common_trace_length_and_sample_interval(monkeypatch, tmp_path):
    hdf5_path = tmp_path / "experiment.hdf5"
    hdf5_path.touch()
    requested_channels = []

    class FakeFile:
        def get_digitizer_specs(self, board, channel, **kwargs):
            requested_channels.append((board, channel, kwargs))
            return {
                "nt": 139_264,
                "clock rate": 100 * u.MHz,
                "sample average": 2,
            }

        def close(self):
            return None

    monkeypatch.setattr(lapd, "File", lambda *_args, **_kwargs: FakeFile())

    assert _read_metadata(hdf5_path) == {
        "nt_full": 139_264,
        "dt_s": 2e-8,
    }
    assert [(board, channel) for board, channel, _kwargs in requested_channels] == [
        (2, 3),
        (2, 2),
    ]
    assert all(
        request[2]["config_name"] == "Langmuir"
        for request in requested_channels
    )


@pytest.mark.parametrize(
    ("adc", "board", "slot", "attributes", "channels", "expected"),
    (
        (
            "SIS 3302",
            2,
            7,
            {"Data type 2": np.bytes_("Probe current"), "Data type 3": "Bias"},
            (3, 2),
            {3: "Bias", 2: "Probe current"},
        ),
        (
            "SIS 3305",
            1,
            13,
            {
                "FPGA 1 Data type 4": np.bytes_("Fast voltage"),
                "FPGA 2 Data type 1": np.bytes_("Fast current"),
            },
            (4, 5),
            {4: "Fast voltage", 5: "Fast current"},
        ),
    ),
)
def test_reads_sis_channel_data_type_labels(
    monkeypatch,
    tmp_path,
    adc,
    board,
    slot,
    attributes,
    channels,
    expected,
):
    import h5py

    hdf5_path = tmp_path / "experiment.hdf5"
    config_path = "/Raw data + config/SIS crate/Langmuir"
    with h5py.File(hdf5_path, "w") as h5_file:
        config_group = h5_file.create_group(config_path)
        config_group.attrs["SIS crate slot numbers"] = np.array([slot])
        config_group.attrs["SIS crate config indices"] = np.array([6])
        adc_group = config_group.create_group(
            f"SIS crate {adc.removeprefix('SIS ')} configurations[6]"
        )
        for name, value in attributes.items():
            adc_group.attrs[name] = value

    class FakeDigitizerMap:
        configs = {"Langmuir": {"config group path": config_path}}

        @staticmethod
        def get_slot(requested_board, requested_adc):
            assert (requested_board, requested_adc) == (board, adc)
            return slot

    class FakeFile:
        def __init__(self, filename):
            self.handle = h5py.File(filename, "r")
            self.digitizers = {"SIS crate": FakeDigitizerMap()}

        def __getitem__(self, key):
            return self.handle[key]

        def close(self):
            self.handle.close()

    monkeypatch.setattr(
        lapd,
        "File",
        lambda filename, **_kwargs: FakeFile(filename),
    )

    assert read_sis_channel_data_types(
        hdf5_path,
        digitizer="SIS crate",
        adc=adc,
        config_name="Langmuir",
        board=board,
        channels=channels,
    ) == expected


@pytest.mark.parametrize(
    ("current_specs", "message"),
    (
        (
            {"nt": 128, "clock rate": 100 * u.MHz, "sample average": 2},
            "different trace lengths",
        ),
        (
            {"nt": 256, "clock rate": 100 * u.MHz, "sample average": 4},
            "different sample intervals",
        ),
    ),
)
def test_rejects_channel_timing_mismatches(
    monkeypatch, tmp_path, current_specs, message
):
    hdf5_path = tmp_path / "experiment.hdf5"
    hdf5_path.touch()

    class FakeFile:
        def get_digitizer_specs(self, _board, channel, **_kwargs):
            if channel == 3:
                return {
                    "nt": 256,
                    "clock rate": 100 * u.MHz,
                    "sample average": 2,
                }
            return current_specs

        def close(self):
            return None

    monkeypatch.setattr(lapd, "File", lambda *_args, **_kwargs: FakeFile())

    with pytest.raises(ValueError, match=message):
        _read_metadata(hdf5_path)


def test_uses_evenly_spaced_time_dataset_when_clock_rate_is_absent(
    monkeypatch, tmp_path
):
    hdf5_path = tmp_path / "experiment.hdf5"
    hdf5_path.touch()
    time_s = np.arange(8, dtype=np.float32) * np.float32(2e-8)

    class FakeFile:
        def get_digitizer_specs(self, _board, _channel, **_kwargs):
            return {"nt": time_s.size, "clock rate": None}

        def get_time_array(self, _specs):
            return time_s

        def close(self):
            return None

    monkeypatch.setattr(lapd, "File", lambda *_args, **_kwargs: FakeFile())

    metadata = _read_metadata(hdf5_path)
    assert metadata["nt_full"] == time_s.size
    assert metadata["dt_s"] == pytest.approx(2e-8)


def test_rejects_nonuniform_time_dataset(monkeypatch, tmp_path):
    hdf5_path = tmp_path / "experiment.hdf5"
    hdf5_path.touch()
    time_s = np.array([0.0, 1e-6, 2e-6, 4e-6], dtype=np.float64)

    class FakeFile:
        def get_digitizer_specs(self, _board, _channel, **_kwargs):
            return {"nt": time_s.size, "clock rate": None}

        def get_time_array(self, _specs):
            return time_s

        def close(self):
            return None

    monkeypatch.setattr(lapd, "File", lambda *_args, **_kwargs: FakeFile())

    with pytest.raises(ValueError, match="not evenly spaced"):
        _read_metadata(hdf5_path)
