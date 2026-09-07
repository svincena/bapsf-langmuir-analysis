# AGENTS.md

This file applies to the entire repository.

## Project purpose

The primary goal of this repository is to provide easy, reliable access to
fitting Langmuir-probe data produced at the Basic Plasma Science Facility
(BaPSF). Favor workflows and interfaces that let facility users move from an
experiment HDF5 file to trustworthy plasma parameters without needing to edit
analysis source code or understand internal storage details.

The analysis supports LAPD x-line and xy-plane scans, repeated shots, multiple
voltage ramps, optional pre-fit shot averaging, diagnostic plots, and HDF5/NPZ
result output. Scientific correctness, clear user feedback, and preservation
of BaPSF acquisition conventions take priority over broad refactors or added
complexity.

## Environment

- Use Python 3.12 through the repository virtual environment. Prefer
  `.venv/bin/python` explicitly; a shell prompt showing both `.venv` and Conda
  does not guarantee that `python` resolves to the virtual environment.
- Install the verified dependency set with:

  ```bash
  .venv/bin/python -m pip install -r requirements-analysis.txt
  ```

- NumPy must remain below version 2 unless the complete `bapsflib`, Astropy,
  `bapsf-motion`, and analysis stack has been verified with it.
- Do not modify the user's global or Conda environment to fix repository-local
  dependency problems.

## Architecture

- `langmuir_analysis.py`: worker/CLI entry point, HDF5 acquisition, geometry
  pipelines, post-processing, plotting, and result writing.
- `langmuir_analysis_core.py`: reusable, geometry-independent I-V fitting and
  plasma-parameter calculations.
- `langmuir_analysis_config.py`: sole source of truth for parameter keys,
  defaults, labels, types, ranges, choice values, persistence, and cross-field
  validation.
- `langmuir_analysis_gui.py`: PySide6 parameter editor and worker-process UI.
- `langmuir_analysis_metadata.py`: HDF5 digitizer timing metadata shared by the
  GUI and worker.
- `langmuir_diagnostics.py`: trace-level diagnostic plotting.
- `read_langmuir_results_hdf5.py` and `read_langmuir_results_npz.py`: stable
  result readers and schema checks.
- `tests/`: numerical, configuration, GUI, pipeline, metadata, and result-reader
  regression coverage.

Put reusable numerical fitting behavior in the core, parameter definitions in
the configuration module, and shared HDF5 metadata logic in the metadata
module. Do not duplicate parameter defaults or metadata calculations across
the GUI and worker.

## Scientific and data-shape invariants

- Do not alter the I-V fitting model, current polarity, saturation-current
  definitions, or density calculation unless the user explicitly requests a
  physics change.
- X-line acquired data are shaped `(nx, nshots, nt_full)`; xy-plane acquired
  data are shaped `(ny, nx, nshots, nt_full)` after acquisition-order
  normalization.
- Multiple ramps add a ramp axis. Per-shot fitted products are
  `(nx, nshots, nramps)` or `(ny, nx, nshots, nramps)`. Shot-averaged fitting
  produces `(nx, nramps)` or `(ny, nx, nramps)`. Preserve legacy shapes when
  `nramps == 1`.
- Shot averaging occurs in voltage space before fitting. Per-shot fit products
  and shot-to-shot fit statistics are unavailable in that mode and remain
  explicitly marked as such.
- XY acquisition may proceed from positive to negative Y or the reverse.
  Normalize results to ascending Y without reversing voltage/current versus
  time or voltage.
- `nt_full` is the complete digitizer trace length. The internal `nt` is the
  extracted sweep/ramp length; do not conflate them.
- The diagnostic directory belongs beside the source HDF5 file and is named
  `<source-stem>_Langmuir_diagnostic_plots`.

## HDF5 geometry and timing

- Manual and file-backed modes must remain distinct and user controlled.
- In bmotion geometry mode, HDF5 values replace and lock spatial bounds/counts,
  `nshots`, and `data_offset`. Re-read them before analysis; saved GUI values
  are only cached display state.
- In HDF5 temporal mode, use `bapsflib` digitizer metadata for the selected
  digitizer, ADC, configuration, board, voltage channel, and current channel.
  Replace and lock `nt_full` and `dt_s`.
- Require the voltage and current channels to agree on trace length and sample
  interval. If only a stored time array is available, accept it only when it is
  evenly spaced; account for float32 quantization when checking uniformity.
- In manual temporal mode, use the entered `dt_s`. Still reject disagreement
  between voltage and current channel timing when both channels report it.
- Prefer supported `bapsflib` APIs such as `get_digitizer_specs()` over parsing
  dataset names directly. Keep GUI-side scientific imports lazy and convert
  dependency failures into actionable parameter errors.
- Metadata inspection must be read-only. Do not run a real analysis merely to
  inspect a source file because result saving can modify that HDF5 file.

## GUI conventions

- Numeric entry fields use `DirectNumericInput`, not spin boxes, so scrolling
  cannot accidentally change values.
- `analysis_processes` is the intentional exception: its spin box supports the
  textual `Automatic` value.
- Finite textual choices remain combo boxes.
- Physical constants are implementation details, not editable GUI fields.
- When a source mode controls a value, disable the corresponding manual fields
  and reconcile again immediately before saving or starting analysis.
- Both geometry tabs are persisted in `last_parameters.json`. This file is
  local run state and must remain ignored by Git.
- Preserve startup import ordering. The no-argument path in
  `langmuir_analysis.py` intentionally launches the GUI before importing the
  heavy analysis stack. `python-dateutil` modules must initialize before
  PySide6 installs its feature-import hook, or later `bapsflib` imports can
  fail through `six.moves`.

## Output contracts

- Keep HDF5 and NPZ metadata and array meanings synchronized when adding or
  changing results.
- Record source/mode choices that affect interpretation, including spatial and
  temporal metadata sources, shot-analysis mode, ramp metadata, channel
  routing, `nt_full`, and `dt_s`.
- Preserve reader compatibility and update both result-reader tests when a
  stable output field changes.
- Never enable pickle loading for NPZ result files.

## Validation and tests

Run focused tests while developing, then run the full suite before handoff:

```bash
PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen \
  .venv/bin/python -m pytest -q
```

Run lint and whitespace checks on touched Python files:

```bash
ruff check <touched-python-files>
git diff --check
```

GUI tests must run with `QT_QPA_PLATFORM=offscreen`. Mock file dialogs,
message boxes, worker processes, and HDF5 metadata reads rather than requiring
interactive UI or real experiment files. Use temporary HDF5/NPZ files for
write-path tests.

When fixing a regression, add a focused test that fails for the reported
behavior. For scientific changes, include a synthetic trace with known
expected plasma parameters or an explicit shape/metadata assertion.

## Change and Git hygiene

- Preserve unrelated user changes and generated experiment data.
- Do not commit `.venv`, `last_parameters.json`, diagnostic plots, result files,
  bytecode, or operating-system metadata.
- Add comments for non-obvious physics, acquisition ordering, compatibility
  workarounds, and metadata assumptions. Avoid comments that merely restate the
  code.
- Do not stage, commit, or push unless the user explicitly requests it. Before
  committing, inspect the staged diff and report the test results. Use a concise
  imperative subject and a body explaining user-visible behavior and important
  invariants.
