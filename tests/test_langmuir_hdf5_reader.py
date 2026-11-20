import h5py
import numpy as np
import pytest

import read_langmuir_results_hdf5 as reader
from read_langmuir_results_hdf5 import (
    EXPECTED_HDF5_DATASETS_BY_GEOMETRY,
    HDF5_GROUP_BY_GEOMETRY,
    list_langmuir_result_geometries,
    load_langmuir_results_hdf5,
    print_hdf5_inventory,
)


def _add_result_group(path, geometry, omitted_dataset=None):
    with h5py.File(path, "a") as h5_file:
        group = h5_file.create_group(HDF5_GROUP_BY_GEOMETRY[geometry])
        group.attrs["geometry"] = geometry
        group.attrs["source_file"] = "synthetic.hdf5"
        group.attrs["analysis_date"] = "2026-11-20"
        group.attrs["te_min_r2"] = 0.95
        for name in EXPECTED_HDF5_DATASETS_BY_GEOMETRY[geometry]:
            if name != omitted_dataset:
                group.create_dataset(name, data=np.array(0.0))
        group.create_dataset("iv_voltage_grid_shot_V", data=np.zeros((2, 3)))
    return path


@pytest.mark.parametrize("geometry", ["x_line", "xy_plane"])
def test_hdf5_reader_detects_and_loads_each_geometry(tmp_path, geometry):
    path = _add_result_group(tmp_path / f"{geometry}.hdf5", geometry)

    assert list_langmuir_result_geometries(path) == (geometry,)
    data = load_langmuir_results_hdf5(path, summary_only=True)

    assert data["geometry"].item() == geometry
    assert data["hdf5_group"].item() == f"/{HDF5_GROUP_BY_GEOMETRY[geometry]}"
    assert data["analysis_date"] == "2026-11-20"
    assert "te_eV" in data
    assert "iv_voltage_grid_shot_V" not in data


def test_hdf5_reader_loads_all_or_requested_datasets(tmp_path):
    path = _add_result_group(tmp_path / "xline.hdf5", "x_line")

    all_data = load_langmuir_results_hdf5(path)
    selected_data = load_langmuir_results_hdf5(
        path,
        dataset_names={"te_eV", "n_e_m3"},
    )

    assert "iv_voltage_grid_V" in all_data
    assert "te_eV" in selected_data
    assert "n_e_m3" in selected_data
    assert "iv_voltage_grid_V" not in selected_data
    assert selected_data["source_file"] == "synthetic.hdf5"


def test_hdf5_reader_requires_geometry_when_both_groups_exist(tmp_path):
    path = _add_result_group(tmp_path / "both.hdf5", "x_line")
    _add_result_group(path, "xy_plane")

    with pytest.raises(ValueError, match="multiple.*--geometry"):
        load_langmuir_results_hdf5(path, summary_only=True)

    xy_data = load_langmuir_results_hdf5(
        path,
        geometry="xy_plane",
        summary_only=True,
    )
    assert xy_data["geometry"].item() == "xy_plane"


def test_hdf5_reader_reports_missing_stable_dataset(tmp_path):
    path = _add_result_group(
        tmp_path / "incomplete.hdf5",
        "xy_plane",
        omitted_dataset="te_eV",
    )

    with pytest.raises(ValueError, match="missing datasets.*te_eV"):
        load_langmuir_results_hdf5(path)


def test_hdf5_reader_rejects_unknown_requested_dataset(tmp_path):
    path = _add_result_group(tmp_path / "xline.hdf5", "x_line")

    with pytest.raises(KeyError, match="not present.*unknown_result"):
        load_langmuir_results_hdf5(path, dataset_names={"unknown_result"})


def test_hdf5_inventory_lists_groups_and_dataset_shapes(tmp_path, capsys):
    path = _add_result_group(tmp_path / "xy.hdf5", "xy_plane")

    print_hdf5_inventory(path)

    output = capsys.readouterr().out
    assert "xy_plane: /langmuir_xy" in output
    assert "te_eV: shape=(), dtype=float64" in output


@pytest.mark.parametrize("geometry", ["x_line", "xy_plane"])
def test_hdf5_reader_preserves_optional_te_region_method(tmp_path, geometry):
    path = _add_result_group(tmp_path / "linear-region.hdf5", geometry)
    with h5py.File(path, "a") as h5_file:
        h5_file[HDF5_GROUP_BY_GEOMETRY[geometry]].attrs["te_region_method"] = "linear_region"
    loaded = load_langmuir_results_hdf5(path, summary_only=True)
    assert loaded["te_region_method"] == "linear_region"


@pytest.mark.parametrize("use_picker", [False, True])
@pytest.mark.parametrize("list_fields", [False, True])
def test_hdf5_reader_cli_optional_filename(
    monkeypatch, tmp_path, use_picker, list_fields
):
    path = _add_result_group(tmp_path / "selected.hdf5", "x_line")
    picker_calls = []

    def select_file():
        picker_calls.append(True)
        return str(path)

    monkeypatch.setattr(reader, "_select_hdf5_file", select_file)
    monkeypatch.setattr(reader, "_choose_reader_action", lambda path: "summary")
    summaries = []
    monkeypatch.setattr(reader, "print_summary", summaries.append)
    argv = [] if use_picker else [str(path)]
    if list_fields:
        argv.append("--list-fields")

    data = reader.main(argv)

    assert picker_calls == ([True] if use_picker else [])
    if list_fields:
        assert data is None
        assert summaries == []
    else:
        assert data["hdf5_file"].item() == str(path)
        assert summaries == [data]


def test_hdf5_reader_cli_cancel_does_not_open_file(monkeypatch):
    monkeypatch.setattr(reader, "_select_hdf5_file", lambda: "")

    def unexpected_read(*args, **kwargs):
        pytest.fail("Cancelling the file picker must not read a file.")

    monkeypatch.setattr(reader, "load_langmuir_results_hdf5", unexpected_read)
    monkeypatch.setattr(reader, "print_hdf5_inventory", unexpected_read)
    assert reader.main([]) is None
    assert reader.main(["--list-fields"]) is None


def test_hdf5_file_picker(monkeypatch):
    # Keep the same import ordering as the desktop entry points.
    import dateutil.rrule
    import dateutil.tz  # noqa: F401
    from PySide6 import QtWidgets

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    calls = []

    def open_file(parent, title, directory, filters):
        calls.append((parent, title, directory, filters))
        return "/tmp/selected.hdf5", "HDF5 files"

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", open_file)

    assert reader._select_hdf5_file() == "/tmp/selected.hdf5"
    assert QtWidgets.QApplication.instance() is application
    assert len(calls) == 1
    assert "*.hdf5" in calls[0][3]


@pytest.mark.parametrize("action", ["analysis", None])
def test_hdf5_reader_interactive_action_skips_results(monkeypatch, action):
    monkeypatch.setattr(reader, "_select_hdf5_file", lambda: "experiment.hdf5")
    monkeypatch.setattr(reader, "_choose_reader_action", lambda path: action)
    launches = []
    monkeypatch.setattr(reader, "_launch_analysis_gui", launches.append)

    def unexpected_read(*args, **kwargs):
        pytest.fail("Only the summary choice should load saved results.")

    monkeypatch.setattr(reader, "load_langmuir_results_hdf5", unexpected_read)
    monkeypatch.setattr(reader, "print_hdf5_inventory", unexpected_read)

    assert reader.main([]) is None
    assert launches == (["experiment.hdf5"] if action == "analysis" else [])


def test_hdf5_reader_analysis_launch_runs_existing_application(monkeypatch):
    from types import SimpleNamespace

    import langmuir_analysis_gui as gui

    window = object()
    calls = []

    def launch_gui(*, source_filename):
        calls.append(("launch", source_filename))
        return window

    def run_application():
        calls.append(("exec",))
        return 0

    monkeypatch.setattr(gui, "launch_gui", launch_gui)
    monkeypatch.setattr(
        reader, "_reader_application", lambda: SimpleNamespace(exec=run_application)
    )

    assert reader._launch_analysis_gui("experiment.hdf5") == 0
    assert calls == [("launch", "experiment.hdf5"), ("exec",)]


@pytest.mark.parametrize(
    "button_text, expected",
    [("Show Summary", "summary"), ("Open Full Analysis", "analysis"), (None, None)],
)
def test_hdf5_reader_action_dialog(monkeypatch, button_text, expected):
    application = reader._reader_application()
    from PySide6 import QtWidgets

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "clickedButton",
        lambda self: next(
            (button for button in self.buttons() if button.text() == button_text),
            None,
        ),
    )
    assert reader._choose_reader_action("experiment.hdf5") == expected
    assert QtWidgets.QApplication.instance() is application
