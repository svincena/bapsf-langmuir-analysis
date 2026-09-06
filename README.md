# Langmuir Probe Analysis

This directory contains a unified swept-Langmuir-probe analysis program and a
geometry-aware reader for its compressed NPZ results.

## Analysis program

[`langmuir_analysis.py`](langmuir_analysis.py) processes either a one-dimensional
x-line scan or a two-dimensional xy-plane scan. Run it to open the graphical
parameter editor:

```bash
python langmuir_analysis.py
```

### Graphical interface

The interface has separate **X-line scan** and **XY-plane scan** tabs. The tab
that is open when **Start Analysis** is pressed defines the geometry; there is
no geometry flag to edit in the source code. Each tab groups its settings into
titled cards for:

- data source and spatial/acquisition geometry;
- probe conversion and sweep windows;
- I–V physics and time/spatial processing;
- interferometer calibration; and
- execution, plots, diagnostics, and result output.

The experiment path has a native file picker, numerical values use direct
numeric entry fields with units, boolean controls are explicit, and invalid
cross-field combinations are rejected before a run starts. Analysis runs in a
separate process, leaving the interface responsive while its output appears in
the live console. **Stop analysis** terminates that worker if necessary.

The **Geometry source** control supports manual spatial entries or authoritative
geometry from the selected file's `bmotion` control. In HDF5 mode, choose the
motion configuration when a file contains more than one. The GUI reads its
target positions through `bapsflib`, then replaces and locks the spatial point
counts and bounds, shots per position, and shot offset. File-backed scans must
have constant repetitions and the regular X-line or rectangular XY acquisition
ordering expected by the analysis. XY rows may run in either Y direction; the
direction is detected and normalized so result coordinates increase from
negative to positive Y. Incompatible motion lists are rejected before
digitizer data are read.

The **Temporal information source** control similarly selects manual timing or
authoritative digitizer metadata from the HDF5 file. In HDF5 mode, the GUI uses
`bapsflib` with the selected digitizer, ADC, configuration, board, voltage
channel, and current channel to replace and lock **Samples per trace**
(`nt_full`) and **Sample interval** (`dt`). The two channels must report the
same trace length and sample interval. A stored time array is accepted only
when it is evenly spaced, because the analysis uses one scalar `dt`. Manual
mode leaves both fields editable and uses the entered sample interval even when
the file reports a different value.

Both tabs are saved automatically in `last_parameters.json`, along with the
last active tab. The file is local run state and is intentionally ignored by
Git. If it is absent, all controls use the documented defaults in
[`langmuir_analysis_config.py`](langmuir_analysis_config.py); a malformed file
also restores defaults and reports why. Use **Save parameters** to save without
starting a run, or **Restore this tab's defaults** to reset one geometry.

### Coding perspective

The program performs the full dataset workflow:

1. Reads voltage and current channels from a LAPD HDF5 file using `bapsflib`.
2. Reshapes the shots as `(x, shot, time)` or `(y, x, shot, time)`, then can
   extract multiple equal-length, evenly spaced ramps as
   `(x, shot, ramp, ramp_time)` or `(y, x, shot, ramp, ramp_time)`.
3. Applies the configured electrical scaling, polarity, baseline treatment,
   and time-domain smoothing.
4. Analyzes every shot and ramp independently through
   [`langmuir_analysis_core.py`](langmuir_analysis_core.py). Independent traces
   can be distributed across worker processes.
5. Calculates shot statistics and applies geometry-specific spatial cleanup:
   one-dimensional interpolation/smoothing for x-line data or two-dimensional
   neighborhood processing for xy data.
6. Applies the common interferometer calibration. For an xy plane, the row
   nearest `interferometer_profile_y_cm` is extracted as an x-line; its
   normalized spatial profile and shape factor calibrate the complete density
   map. The physical coefficient `1/(c r_e) = 1.18e6 s m⁻² rad⁻¹` is an
   internal constant rather than a user-configurable parameter.
7. Produces diagnostic figures and optionally writes results into the source
   HDF5 file and a compressed NPZ file.

The typed defaults, labels, ranges, validation, and persistence schema live in
`langmuir_analysis_config.py`; the Qt interface lives in
[`langmuir_analysis_gui.py`](langmuir_analysis_gui.py). Geometry-dependent data
acquisition and visualization remain in `langmuir_analysis.py`, while the
reusable numerical I–V solver lives in `langmuir_analysis_core.py`. The GUI
starts `langmuir_analysis.py` in a saved-parameter worker mode, keeping the
physical solver independent of both scan geometry and interface code.

### Plasma-physics perspective

For each swept probe trace, the analysis constructs a monotonic current–voltage
curve and estimates:

- Floating potential, `Vf`, from the bracketed negative-to-positive current
  crossing.
- Plasma potential, `Vp`, primarily from the peak in `dI/dV`, with a fitted
  retarding-branch estimate retained as a consistency diagnostic.
- Ion saturation current, `Iis`, from the low-bias ion region.
- Electron saturation current, `Ies`, using the selected plasma-potential or
  high-bias method described below.
- Electron temperature, `Te`, from a single-temperature semilog fit to the
  electron-retarding branch after subtracting the measured ion-region current.
- Electron density, `n_e`, from the planar Maxwellian electron-saturation
  current relation using the configured probe area.

#### Electron-saturation current methods

The **Ies estimation method** control provides two definitions. The selected
`Ies` is also used by the electron-temperature fit thresholds, the fitted-`Vp`
consistency diagnostic, and the electron-density calculation.

- **At plasma potential (ion-subtracted)** (`at_vp`) subtracts the median
  low-bias ion-region current from the binned measured current, then linearly
  interpolates that electron current at the derivative-based `Vp` found from
  the `dI/dV` peak. This method does not assume that the positive-bias branch
  reaches a flat plateau. It is therefore less susceptible to an increasing
  collection area caused by sheath expansion, but it is more sensitive to the
  accuracy of the local `Vp`, ion-current, and current measurements.
- **High-bias regional median** (`high_bias_median`) is the original method and
  remains the default for backward-compatible results. It selects binned
  points satisfying
  `V >= Vp + 0.5 * (Vmax - Vp)` and uses their median measured current. The
  regional median is resistant to point noise, but interpreting it as `Ies`
  assumes that this part of the electron branch is effectively saturated. If
  sheath expansion makes current continue to rise above `Vp`, this method will
  generally return a larger `Ies` and consequently a larger inferred density
  than the `at_vp` method.

The two definitions coincide only when the ion-subtracted electron current at
`Vp` is equal to a flat high-bias saturation level. Saved HDF5 and NPZ results
record both `ies_method` and a human-readable `ies_definition` so the choice is
not ambiguous during later analysis.

The **Shot fitting mode** control determines when repeated shots are combined.
**Fit each shot separately** preserves the default workflow: every shot is fit
independently, then the fitted quantities are averaged and their sample
standard deviations are calculated. **Average shots before fitting** pools the
measured samples from all shots at each spatial location, averages them in the
configured voltage bins, and performs one Langmuir fit on that mean I–V curve.
This voltage-space average tolerates small timing and endpoint differences
between sweeps. In averaged mode, fit-derived shot standard deviations and
per-shot fit products are stored as unavailable (`NaN`); the result metadata
records `shot_analysis_mode` and `shot_statistics_available`.

For a discharge containing repeated sweeps, **Sweep start** and **Sweep end**
define the first ramp. **Number of ramps** enables repeated-ramp analysis, and
**Ramp start spacing** gives the start-to-start separation in samples. Ramps
must not overlap and every ramp must fit within the acquired trace. Each ramp
is fitted and spatially post-processed independently. With per-shot fitting,
x-line products have shape `(nx, nshots, nramps)` and XY products have shape
`(ny, nx, nshots, nramps)`. Averaging shots before fitting produces
`(nx, nramps)` or `(ny, nx, nramps)`, respectively. A single-ramp run retains
the legacy shapes without a singleton ramp axis.

The **Multiple-ramp display** control selects either distinct ramp profiles or
six x-versus-ramp-center-time maps. The display choice does not combine ramps
or change their fitted values. Multi-ramp XY runs produce a separate six-panel
spatial-map summary for each ramp. Interferometer density calibration is also
calculated independently for every ramp in either geometry.

The solver records fit quality, uncertainties, model-consistency notes, and
rejection reasons. Finite estimates may be retained for diagnosis even when a
trace fails the full acceptance policy; use `analysis_ok` and the related
per-shot masks to distinguish accepted fits.

Absolute current calibration matters physically. Enable **Absolute current
zero calibrated** only when the current zero has been established independently.
Do not use low-bias sweep samples as an artificial zero: that removes the ion
current and biases `Vf` and `Iis`. Likewise, **Reverse current polarity** must
match the acquisition electronics so the solver receives negative ion current
and positive electron current.

### Result files

Results are saved in two places when **Save HDF5 and NPZ results** is enabled:

- Inside the source experiment HDF5 file under `/langmuir_xline` or
  `/langmuir_xy`.
- In a separate compressed NPZ file derived from the HDF5 filename:
  - `*_langmuir_xline.npz` for x-line results
  - `*_langmuir_xy.npz` for xy-plane results

Both storage formats record the result geometry. Primary quantities include
`te_eV`, `vp_V`, `vf_V`, `ies_A`, `iis_A`, and `n_e_m3`. The files also contain
raw and processed spatial results, shot statistics, fit-quality fields,
interpolated I–V products, calibration settings, and optional embedded plot
images. Multi-ramp files additionally record ramp indices, start and center
times, start spacing, and explicit axis-order metadata. XY output uses
`y,x,shot,ramp` for per-shot products and `y,x,ramp` for aggregate maps.

## Reading HDF5 results

[`read_langmuir_results_hdf5.py`](read_langmuir_results_hdf5.py) reads results
directly from the experiment HDF5 file. It opens the file read-only and detects
`/langmuir_xline` or `/langmuir_xy` automatically when exactly one is present.

Print a compact result and fit-quality summary:

```bash
python read_langmuir_results_hdf5.py path/to/experiment.hdf5
```

Open the geometry-appropriate six-panel plot:

```bash
python read_langmuir_results_hdf5.py path/to/experiment.hdf5 --plot
```

List result groups, dataset names, shapes, and data types without loading the
array contents:

```bash
python read_langmuir_results_hdf5.py path/to/experiment.hdf5 --list-fields
```

If a file contains both result groups, select one explicitly:

```bash
python read_langmuir_results_hdf5.py path/to/experiment.hdf5 \
    --geometry xy_plane --plot
```

### Using the HDF5 reader from Python

By default, the Python loader returns every result dataset plus the group
attributes in one flat dictionary:

```python
from read_langmuir_results_hdf5 import load_langmuir_results_hdf5

data = load_langmuir_results_hdf5("path/to/experiment.hdf5")
electron_temperature_eV = data["te_eV"]
electron_density_m3 = data["n_e_m3"]
```

Per-shot I–V grids can be large. Load only the plotting and summary fields when
full numerical products are unnecessary:

```python
data = load_langmuir_results_hdf5(
    "path/to/experiment.hdf5",
    summary_only=True,
)
```

Or request specific datasets:

```python
data = load_langmuir_results_hdf5(
    "path/to/experiment.hdf5",
    geometry="x_line",
    dataset_names={"x_cm", "te_eV", "n_e_m3"},
)
```

Group attributes such as `geometry`, `source_file`, fit settings, acquisition
settings, and interferometer calibration metadata are included automatically.
The returned `hdf5_group` field records which result group was read.

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

For multi-ramp XY results, `plot_summary` creates every per-ramp figure and
returns them as a list; single-ramp and x-line calls return one figure.

Geometry-specific compatibility loaders remain available:

```python
from read_langmuir_results_npz import (
    load_langmuir_xline_npz,
    load_langmuir_xy_npz,
)
```

These loaders verify that the file contains the requested geometry.

### NPZ reader troubleshooting

- A missing or unsupported `geometry` field produces an explicit error.
- Missing stable result fields are reported by name during schema validation.
- NPZ files are loaded with `allow_pickle=False`; results should contain NumPy
  arrays and scalar strings rather than arbitrary Python objects.
- `--plot` requires a working Matplotlib display backend. The non-plotting
  inventory command also works in headless environments.
- XY plots mark the x-line used for interferometer calibration when that
  metadata is present.

### HDF5 reader troubleshooting

- A file with neither `/langmuir_xline` nor `/langmuir_xy` is not an analyzed
  Langmuir result file.
- If both groups are present, use `--geometry x_line` or
  `--geometry xy_plane`.
- Missing stable datasets are reported by name during group validation.
- Use `--list-fields` to inspect large files without loading result arrays.
- The reader always uses HDF5 read-only mode and does not modify the experiment
  file.

## Environment

The numerical package versions verified with the core tests are listed in
[`requirements-analysis.txt`](requirements-analysis.txt). The full HDF5
analysis program also requires `bapsflib` and `h5py` for LAPD data access and
result storage. The graphical interface uses PySide6 (Qt 6).
