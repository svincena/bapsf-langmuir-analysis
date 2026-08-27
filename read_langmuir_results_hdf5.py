"""Read and optionally plot Langmuir results stored in experiment HDF5 files.

The analyzer stores results in ``/langmuir_xline`` or ``/langmuir_xy``. This
reader discovers those groups, validates their stable datasets, and presents
the same flat mapping and summary plots as :mod:`read_langmuir_results_npz`.
Files are always opened read-only.

Examples
--------
Print a compact summary::

    python read_langmuir_results_hdf5.py experiment.hdf5

Display the geometry-appropriate result plot::

    python read_langmuir_results_hdf5.py experiment.hdf5 --plot

List available result groups and datasets without loading their contents::

    python read_langmuir_results_hdf5.py experiment.hdf5 --list-fields
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from read_langmuir_results_npz import plot_summary, print_summary


HDF5_GROUP_BY_GEOMETRY = {
    "x_line": "langmuir_xline",
    "xy_plane": "langmuir_xy",
}
SUPPORTED_LANGMUIR_GEOMETRIES = set(HDF5_GROUP_BY_GEOMETRY)

_COMMON_EXPECTED_HDF5_DATASETS = {
    "x_cm",
    "time_s",
    "te_eV",
    "vp_V",
    "vf_V",
    "ies_A",
    "iis_A",
    "isat_A",
    "n_e_m3",
    "te_raw_eV",
    "vp_raw_V",
    "vf_raw_V",
    "ies_raw_A",
    "iis_raw_A",
    "n_e_raw_m3",
    "te_processed_eV",
    "vp_processed_V",
    "vf_processed_V",
    "ies_processed_A",
    "iis_processed_A",
    "n_e_processed_m3",
    "vp_spike_mask",
    "analysis_ok",
    "analysis_valid_count",
    "analysis_ok_shot",
    "iv_voltage_grid_V",
    "iv_current_grid_A",
    "iv_didv_grid_A_per_V",
    "iv_te_fit_mask",
    "te_fit_r2",
    "te_fit_rmse_logI",
    "te_fit_npts",
    "te_fit_vstart_V",
    "te_fit_vstop_V",
    "te_fit_slope_logI_per_V",
    "te_fit_intercept_logI",
    "te_fit_i0_A",
    "te_fit_passed_r2",
    "te_fit_candidate_count",
    "trace_spatial_shape",
}

EXPECTED_LANGMUIR_XLINE_HDF5_DATASETS = _COMMON_EXPECTED_HDF5_DATASETS | {
    "xline_shape_info",
}
EXPECTED_LANGMUIR_XY_HDF5_DATASETS = _COMMON_EXPECTED_HDF5_DATASETS | {
    "y_cm",
    "X_cm",
    "Y_cm",
    "xy_shape_info",
}
EXPECTED_HDF5_DATASETS_BY_GEOMETRY = {
    "x_line": EXPECTED_LANGMUIR_XLINE_HDF5_DATASETS,
    "xy_plane": EXPECTED_LANGMUIR_XY_HDF5_DATASETS,
}

# These are sufficient for print_summary() and plot_summary(). The Python API
# loads every dataset by default, but the CLI uses this subset so it does not
# pull large per-shot I-V grids into memory just to display a summary.
SUMMARY_HDF5_DATASETS = {
    "x_cm",
    "y_cm",
    "X_cm",
    "Y_cm",
    "te_eV",
    "vp_V",
    "vf_V",
    "ies_A",
    "iis_A",
    "n_e_m3",
    "te_std_eV",
    "vp_std_V",
    "vf_std_V",
    "ies_std_A",
    "iis_std_A",
    "n_e_std_m3",
    "te_raw_eV",
    "vp_raw_V",
    "analysis_ok",
    "analysis_valid_count",
    "vp_spike_mask",
    "te_fit_r2",
}


def _decode_hdf5_value(value):
    """Decode scalar HDF5 byte strings while preserving arrays and numbers."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return value


def _available_geometry_groups(h5_file):
    available = {}
    for geometry, group_name in HDF5_GROUP_BY_GEOMETRY.items():
        if group_name in h5_file and isinstance(h5_file[group_name], h5py.Group):
            available[geometry] = h5_file[group_name]
    return available


def list_langmuir_result_geometries(path):
    """Return geometry names that have result groups in an HDF5 file."""
    with h5py.File(path, "r") as h5_file:
        return tuple(sorted(_available_geometry_groups(h5_file)))


def _select_result_group(h5_file, geometry=None):
    available = _available_geometry_groups(h5_file)
    if not available:
        group_names = ", ".join(f"/{name}" for name in HDF5_GROUP_BY_GEOMETRY.values())
        raise ValueError(
            "HDF5 file contains no Langmuir result group; expected "
            f"one of {group_names}."
        )

    if geometry is None:
        if len(available) > 1:
            choices = ", ".join(sorted(available))
            raise ValueError(
                "HDF5 file contains multiple Langmuir result geometries "
                f"({choices}); select one with geometry= or --geometry."
            )
        geometry = next(iter(available))
    elif geometry not in SUPPORTED_LANGMUIR_GEOMETRIES:
        choices = ", ".join(sorted(SUPPORTED_LANGMUIR_GEOMETRIES))
        raise ValueError(
            f"Unsupported geometry {geometry!r}; expected one of {{{choices}}}."
        )
    elif geometry not in available:
        expected_group = HDF5_GROUP_BY_GEOMETRY[geometry]
        raise ValueError(
            f"HDF5 file does not contain /{expected_group} for {geometry!r} results."
        )

    return geometry, available[geometry]


def validate_langmuir_hdf5_group(group, geometry):
    """Validate one selected result group while allowing extra datasets."""
    if geometry not in SUPPORTED_LANGMUIR_GEOMETRIES:
        raise ValueError(f"Unsupported geometry {geometry!r}.")

    stored_geometry = _decode_hdf5_value(group.attrs.get("geometry", geometry))
    if str(stored_geometry) != geometry:
        raise ValueError(
            f"HDF5 group {group.name} is selected as {geometry!r}, but its "
            f"geometry attribute is {stored_geometry!r}."
        )

    dataset_names = {
        name for name, item in group.items() if isinstance(item, h5py.Dataset)
    }
    missing = EXPECTED_HDF5_DATASETS_BY_GEOMETRY[geometry] - dataset_names
    if missing:
        raise ValueError(
            f"HDF5 group {group.name} does not conform to the expected "
            f"{geometry} Langmuir schema: missing datasets: {sorted(missing)}"
        )
    return group


def load_langmuir_results_hdf5(
    path,
    geometry=None,
    dataset_names=None,
    summary_only=False,
):
    """Load Langmuir result datasets and group attributes into a dictionary.

    Parameters
    ----------
    path : path-like
        Experiment HDF5 file containing a Langmuir result group.
    geometry : {"x_line", "xy_plane"}, optional
        Geometry to read. It is inferred when exactly one result group exists.
    dataset_names : iterable of str, optional
        Load only these datasets. By default all datasets are loaded.
    summary_only : bool, default False
        Load only fields needed by the shared summary and plot functions.
        This cannot be combined with ``dataset_names``.

    Notes
    -----
    Loading all fields may use substantial memory because per-shot I-V grids
    are stored in the HDF5 group. Use ``dataset_names`` or ``summary_only``
    when only selected products are needed.
    """
    if summary_only and dataset_names is not None:
        raise ValueError("summary_only and dataset_names cannot be used together.")

    path = Path(path)
    with h5py.File(path, "r") as h5_file:
        geometry, group = _select_result_group(h5_file, geometry=geometry)
        validate_langmuir_hdf5_group(group, geometry)

        available_datasets = {
            name for name, item in group.items() if isinstance(item, h5py.Dataset)
        }
        if summary_only:
            selected_names = available_datasets & SUMMARY_HDF5_DATASETS
        elif dataset_names is None:
            selected_names = available_datasets
        else:
            selected_names = set(dataset_names)
            missing_requested = selected_names - available_datasets
            if missing_requested:
                raise KeyError(
                    f"Requested datasets are not present in {group.name}: "
                    f"{sorted(missing_requested)}"
                )

        # Attributes are kept in the same flat mapping as datasets so the
        # existing NPZ summary and plotting functions can consume either source.
        data = {
            name: _decode_hdf5_value(value)
            for name, value in group.attrs.items()
        }
        for name in sorted(selected_names):
            data[name] = _decode_hdf5_value(group[name][()])

        data["geometry"] = np.array(geometry)
        data["hdf5_file"] = np.array(str(path))
        data["hdf5_group"] = np.array(group.name)
        return data


def print_hdf5_inventory(path):
    """Print available result groups, attributes, and dataset shapes."""
    with h5py.File(path, "r") as h5_file:
        available = _available_geometry_groups(h5_file)
        if not available:
            print("No Langmuir result groups found.")
            return

        for geometry in sorted(available):
            group = available[geometry]
            print(f"{geometry}: {group.name}")
            for name, item in sorted(group.items()):
                if isinstance(item, h5py.Dataset):
                    print(f"  {name}: shape={item.shape}, dtype={item.dtype}")


def main(argv=None):
    """Run the read-only HDF5 result-reader command line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Read and optionally plot Langmuir results stored in an experiment "
            "HDF5 file."
        ),
        epilog=(
            "Examples:\n"
            "  python read_langmuir_results_hdf5.py experiment.hdf5\n"
            "  python read_langmuir_results_hdf5.py experiment.hdf5 --plot\n"
            "  python read_langmuir_results_hdf5.py experiment.hdf5 --list-fields"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("hdf5_path", help="Path to an experiment HDF5 file.")
    parser.add_argument(
        "--geometry",
        choices=sorted(SUPPORTED_LANGMUIR_GEOMETRIES),
        help="Result geometry; required only if the file contains both groups.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show the geometry-appropriate six-panel summary plot.",
    )
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="List result groups and dataset shapes without loading arrays.",
    )
    args = parser.parse_args(argv)

    if args.list_fields:
        print_hdf5_inventory(args.hdf5_path)
        return None

    data = load_langmuir_results_hdf5(
        args.hdf5_path,
        geometry=args.geometry,
        summary_only=True,
    )
    print_summary(data)

    if args.plot:
        import matplotlib.pyplot as plt

        plot_summary(data)
        plt.show()
    return data


if __name__ == "__main__":
    main()
