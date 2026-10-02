import astropy.units as u
import numpy as np
import pytest
from bapsflib import lapd

from langmuir_analysis_config import default_parameters
from langmuir_analysis_metadata import (
    read_digitizer_temporal_metadata,
    read_sis_channel_data_types,
    read_sweep_preview_trace,
)


@pytest.mark.parametrize("source", ["manual", "hdf5"])
def test_preview_reads_only_one_shot_read_only_with_analysis_scaling(
    monkeypatch, tmp_path, source
):
    path = tmp_path / "preview.hdf5"
    path.touch()
    values = default_parameters("x_line")
    values.update(
        filename=str(path),
        nt_full=8,
        dt_s=1e-6,
        temporal_metadata_source=source,
        vsweep_attenuation=100.0,
        isweep_resistance_ohm=2.0,
        negate_Isweep_current=True,
        subtract_dc=True,
        isweep_dc_offset_start_index=0,
        isweep_dc_offset_end_index=1,
    )
    reads, opens, closed = [], [], []

    class Raw(dict):
        dt = 2e-6 * u.s

    class FakeFile:
        def get_digitizer_specs(self, *_args, **_kwargs):
            return {"nt": 8, "clock rate": 1 * u.MHz, "sample average": 2}

        def read_data(self, board, channel, **kwargs):
            reads.append((board, channel, kwargs))
            return Raw(
                signal=np.arange(8, dtype=np.float32)[None, :], shotnum=np.array([108])
            )

        def close(self):
            closed.append(True)

    def open_file(filename, **kwargs):
        opens.append(kwargs)
        return FakeFile()

    monkeypatch.setattr(lapd, "File", open_file)
    preview = read_sweep_preview_trace(values, 108)
    assert len(reads) == 2
    assert [item[1] for item in reads] == [
        values["vsweep_channel"],
        values["isweep_channel"],
    ]
    assert all(item[2]["shotnum"] == slice(108, 109) for item in reads)
    assert opens[-1]["mode"] == "r"
    assert len(closed) == len(opens)
    assert np.array_equal(preview["voltage_V"], np.arange(8) * 100)
    assert np.array_equal(preview["current_A"], -np.arange(8) / 2 + 0.25)
    assert preview["dt_s"] == (1e-6 if source == "manual" else 2e-6)
    assert np.allclose(preview["time_s"], np.arange(8) * preview["dt_s"])


def test_preview_rejects_channel_timing_mismatch_and_closes_file(monkeypatch, tmp_path):
    path = tmp_path / "preview.hdf5"
    path.touch()
    values = default_parameters("x_line")
    values.update(
        filename=str(path), temporal_metadata_source="manual", nt_full=8, dt_s=1e-6
    )
    closed = []

    class Raw(dict):
        dt = None

    class FakeFile:
        def read_data(self, board, channel, **kwargs):
            raw = Raw(signal=np.ones((1, 8)), shotnum=np.array([1]))
            raw.dt = (1e-6 if channel == values["vsweep_channel"] else 2e-6) * u.s
            return raw

        def close(self):
            closed.append(True)

    monkeypatch.setattr(lapd, "File", lambda *_args, **_kwargs: FakeFile())
    with pytest.raises(ValueError, match="different sample intervals"):
        read_sweep_preview_trace(values, 1)
    assert closed == [True]


def test_preview_rejects_missing_shot_instead_of_substituting_another(
    monkeypatch, tmp_path
):
    path = tmp_path / "preview.hdf5"
    path.touch()
    values = default_parameters("x_line")
    values.update(filename=str(path), temporal_metadata_source="manual", nt_full=8)

    class FakeFile:
        def read_data(self, *_args, **_kwargs):
            return {"signal": np.ones((1, 8)), "shotnum": np.array([2])}

        def close(self):
            pass

    monkeypatch.setattr(lapd, "File", lambda *_args, **_kwargs: FakeFile())
    with pytest.raises(ValueError, match="does not contain shot 1"):
        read_sweep_preview_trace(values, 1)


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
