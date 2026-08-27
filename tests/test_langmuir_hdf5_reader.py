import h5py
import numpy as np
import pytest

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
