# Langmuir Probe Analysis

This directory contains a unified swept-Langmuir-probe analysis program and a
geometry-aware reader for its compressed NPZ results.

## Analysis program

[`langmuir_analysis.py`](langmuir_analysis.py) processes either a one-dimensional
x-line scan or a two-dimensional xy-plane scan. Select the pipeline near the
top of the file:

```python
analysis_geometry = "x_line"   # or "xy_plane"
```

The selected geometry block contains its acquisition settings, sweep indices,
probe calibration, fitting controls, post-processing options, plotting flags,
and output controls. Run it with:

```bash
python langmuir_analysis.py
```

### Coding perspective

The program performs the full dataset workflow:

1. Reads voltage and current channels from a LAPD HDF5 file using `bapsflib`.
2. Reshapes the shots as `(x, shot, time)` or `(y, x, shot, time)`.
3. Applies the configured electrical scaling, polarity, baseline treatment,
   and time-domain smoothing.
4. Analyzes every shot independently through
   [`langmuir_analysis_core.py`](langmuir_analysis_core.py). Independent traces
   can be distributed across worker processes.
5. Calculates shot statistics and applies geometry-specific spatial cleanup:
   one-dimensional interpolation/smoothing for x-line data or two-dimensional
   neighborhood processing for xy data.
6. Applies the common interferometer calibration. For an xy plane, the row
   nearest `interferometer_profile_y_cm` is extracted as an x-line; its
   normalized spatial profile and shape factor calibrate the complete density
   map.
7. Produces diagnostic figures and optionally writes results into the source
   HDF5 file and a compressed NPZ file.

The geometry-dependent acquisition and visualization code lives in
`langmuir_analysis.py`; the reusable numerical I–V solver lives in
`langmuir_analysis_core.py`. This separation keeps the physical trace model
independent of scan geometry.

### Plasma-physics perspective

For each swept probe trace, the analysis constructs a monotonic current–voltage
curve and estimates:

- Floating potential, `Vf`, from the bracketed negative-to-positive current
  crossing.
- Plasma potential, `Vp`, primarily from the peak in `dI/dV`, with a fitted
  retarding-branch estimate retained as a consistency diagnostic.
- Ion saturation current, `Iis`, from the low-bias ion region.
- Electron saturation current, `Ies`, from the high-bias electron region.
- Electron temperature, `Te`, from a single-temperature semilog fit to the
  electron-retarding branch after subtracting the measured ion-region current.
- Electron density, `n_e`, from the planar Maxwellian electron-saturation
  current relation using the configured probe area.

The solver records fit quality, uncertainties, model-consistency notes, and
rejection reasons. Finite estimates may be retained for diagnosis even when a
trace fails the full acceptance policy; use `analysis_ok` and the related
per-shot masks to distinguish accepted fits.

Absolute current calibration matters physically. Set
`current_zero_calibrated = True` only when the current zero has been established
independently. Do not use low-bias sweep samples as an artificial zero: that
removes the ion current and biases `Vf` and `Iis`. Likewise,
`negate_Isweep_current` must match the acquisition electronics so the solver
receives negative ion current and positive electron current.

### Result files

The NPZ filename is derived from the input HDF5 filename:

- `*_langmuir_xline.npz` for x-line results
- `*_langmuir_xy.npz` for xy-plane results

Both formats store a scalar `geometry` field. Primary quantities include
`te_eV`, `vp_V`, `vf_V`, `ies_A`, `iis_A`, and `n_e_m3`. The files also contain
raw and processed spatial results, shot statistics, fit-quality fields,
interpolated I–V products, calibration settings, and optional embedded plot
images.

## Reading NPZ results

[`read_langmuir_results_npz.py`](read_langmuir_results_npz.py) reads both result
geometries. It detects the correct schema and plot style from the stored
`geometry` field; no reader-side geometry flag is needed.

Print a compact inventory and quality summary:

```bash
python read_langmuir_results_npz.py path/to/results.npz
```

Open the matching six-panel summary plot:

```bash
python read_langmuir_results_npz.py path/to/results.npz --plot
```

Show all command-line options:

```bash
python read_langmuir_results_npz.py --help
```

### Using the reader from Python

```python
from read_langmuir_results_npz import load_langmuir_results_npz

data = load_langmuir_results_npz("path/to/results.npz")

geometry = data["geometry"].item()
electron_temperature_eV = data["te_eV"]
plasma_potential_V = data["vp_V"]
electron_density_m3 = data["n_e_m3"]
analysis_ok = data["analysis_ok"].astype(bool)
```

To create the appropriate plot programmatically:

```python
import matplotlib.pyplot as plt
from read_langmuir_results_npz import (
    load_langmuir_results_npz,
    plot_summary,
)

data = load_langmuir_results_npz("path/to/results.npz")
figure = plot_summary(data)
plt.show()
```

Geometry-specific compatibility loaders remain available:

```python
from read_langmuir_results_npz import (
    load_langmuir_xline_npz,
    load_langmuir_xy_npz,
)
```

These loaders verify that the file contains the requested geometry.

### Reader troubleshooting

- A missing or unsupported `geometry` field produces an explicit error.
- Missing stable result fields are reported by name during schema validation.
- NPZ files are loaded with `allow_pickle=False`; results should contain NumPy
  arrays and scalar strings rather than arbitrary Python objects.
- `--plot` requires a working Matplotlib display backend. The non-plotting
  inventory command also works in headless environments.
- XY plots mark the x-line used for interferometer calibration when that
  metadata is present.

## Environment

The numerical package versions verified with the core tests are listed in
[`requirements-analysis.txt`](requirements-analysis.txt). The full HDF5
analysis program also requires `bapsflib` and `h5py` for LAPD data access and
result storage.
