#!/usr/bin/env python3
"""Read and inspect an NPZ archive produced by ``read_mono.py``.

The archive is intentionally self-describing and contains three categories:

* Raw inputs: ``time`` and ``wlengths``. The much larger oscilloscope signal
  named by ``signal_hdf5_dataset`` is not duplicated in the archive.
* Derived results: ``time_ms``, ``dtime_ms``, and ``allpeaks``. Rows of
  ``allpeaks`` correspond to ``wlengths`` and columns correspond to 1 ms bins.
* Reproducibility metadata: HDF5 source paths, excluded-signal shape and dtype,
  peak-finding settings, analysis dimensions, and ``analysis_script_source``,
  which is the exact text of ``read_mono.py`` captured when results were saved.

Run ``python read_mono_npz.py ARCHIVE.npz`` to print a summary. Add
``--show-source`` to print the archived analysis script. The public
``read_mono_npz`` function returns a dictionary of arrays and Python scalar
values for use from another program.
"""

import argparse
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = {
    'archive_format_version',
    'time',
    'wlengths',
    'time_ms',
    'dtime_ms',
    'allpeaks',
    'n_lambda',
    'n_shots',
    'n_ms',
    'hdf_filepath',
    'signal_hdf5_dataset',
    'excluded_raw_shape',
    'excluded_raw_dtype',
    'analysis_script_source',
}


def read_mono_npz(npz_filepath):
    """Load and validate a monochromator results archive.

    Parameters
    ----------
    npz_filepath : str or pathlib.Path
        Path to an NPZ file written by ``read_mono.py``.

    Returns
    -------
    dict
        Mapping of archive field names to values. Zero-dimensional arrays are
        converted to ordinary Python scalar values; numerical arrays remain
        NumPy arrays.

    Raises
    ------
    ValueError
        If required fields are absent or the dimensions of ``allpeaks`` do not
        agree with ``n_lambda`` and ``n_ms``.

    Notes
    -----
    Loading uses ``allow_pickle=False`` because the writer stores no Python
    objects. This avoids executing arbitrary pickled content from an archive.
    """
    npz_filepath = Path(npz_filepath).expanduser().resolve()
    with np.load(npz_filepath, allow_pickle=False) as archive:
        missing = REQUIRED_FIELDS.difference(archive.files)
        if missing:
            missing_fields = ', '.join(sorted(missing))
            raise ValueError(f'Archive is missing required fields: {missing_fields}')

        contents = {
            name: value.item() if value.ndim == 0 else value.copy()
            for name, value in ((name, archive[name]) for name in archive.files)
        }

    expected_shape = (contents['n_lambda'], contents['n_ms'])
    if contents['allpeaks'].shape != expected_shape:
        raise ValueError(
            f"allpeaks shape {contents['allpeaks'].shape} does not match "
            f'recorded shape {expected_shape}'
        )

    return contents


def print_archive_summary(contents):
    """Print the principal arrays and provenance from a loaded archive."""
    print(f"Archive format: {contents['archive_format_version']}")
    print(f"Source HDF5: {contents['hdf_filepath']}")
    print(f"Raw time shape: {contents['time'].shape}")
    print(f"Wavelength shape: {contents['wlengths'].shape}")
    print(f"Peak-count shape: {contents['allpeaks'].shape}")
    print(
        'Excluded raw signal: '
        f"{contents['signal_hdf5_dataset']} "
        f"shape={tuple(contents['excluded_raw_shape'])} "
        f"dtype={contents['excluded_raw_dtype']}"
    )
    print(f"Shots per wavelength: {contents['n_shots']}")


def main():
    """Parse command-line arguments, load one archive, and print its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz_filepath', help='NPZ archive produced by read_mono.py')
    parser.add_argument(
        '--show-source',
        action='store_true',
        help='print the archived read_mono.py source after the summary',
    )
    args = parser.parse_args()

    contents = read_mono_npz(args.npz_filepath)
    print_archive_summary(contents)
    if args.show_source:
        print('\n--- Archived read_mono.py source ---\n')
        print(contents['analysis_script_source'])


if __name__ == '__main__':
    main()
