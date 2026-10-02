import astropy.units as u
import h5py
import numpy as np
import pytest

import langmuir_analysis as analysis
from langmuir_analysis_config import default_parameters


def test_run_analysis_dispatches_selected_geometry(monkeypatch):
    called = []
    monkeypatch.setattr(
        analysis,
        "run_xline_analysis",
        lambda: called.append("x_line"),
    )
    monkeypatch.setattr(
        analysis,
        "run_xy_analysis",
        lambda: called.append("xy_plane"),
    )

    for geometry in ("x_line", "xy_plane"):
        values = default_parameters(geometry)
        values["temporal_metadata_source"] = "manual"
        analysis.run_analysis(geometry, values)

    assert called == ["x_line", "xy_plane"]


def test_run_analysis_rejects_unknown_geometry():
    with pytest.raises(ValueError, match="analysis_geometry"):
        analysis.run_analysis("radial", {})


def test_run_analysis_refreshes_hdf5_temporal_metadata(monkeypatch):
    values = default_parameters("x_line")
    values.update(nt_full=10, dt_s=1e-6)
    monkeypatch.setattr(
        analysis,
        "read_digitizer_temporal_metadata",
        lambda *_args, **_kwargs: {"nt_full": 20_000, "dt_s": 2e-8},
    )
    observed = []
    monkeypatch.setattr(
        analysis,
        "run_xline_analysis",
        lambda: observed.append((analysis.nt_full, analysis.dt_s)),
    )

    analysis.run_analysis("x_line", values)

    assert observed == [(20_000, 2e-8)]


def test_worker_resolves_time_entries_after_rereading_selected_channel_timing(
    monkeypatch,
):
    values = default_parameters("x_line")
    values.update(
        dt_s=20e-9,
        sweep_start_input_unit="microseconds",
        sweep_start_time_s=100e-6,
        sweep_end_input_unit="milliseconds",
        sweep_end_time_s=300e-6,
        ramp_start_spacing_input_unit="microseconds",
        ramp_start_spacing_time_s=400e-6,
        nramps=2,
    )
    monkeypatch.setattr(
        analysis,
        "read_digitizer_temporal_metadata",
        lambda *_args, **_kwargs: {"nt_full": 30_000, "dt_s": 40e-9},
    )
    observed = []
    monkeypatch.setattr(
        analysis,
        "run_xline_analysis",
        lambda: observed.append(
            (
                analysis.sweep_start_index,
                analysis.sweep_end_index,
                analysis.ramp_start_spacing_samples,
                analysis.nt,
            )
        ),
    )
    analysis.run_analysis("x_line", values)
    assert observed == [(2500, 7500, 10000, 5001)]


def test_sample_interval_selection_honors_manual_mode_and_checks_hdf5_mode():
    assert analysis.resolve_sample_interval(2e-6, "manual", 1e-6, 1e-6) == 2e-6
    assert analysis.resolve_sample_interval(1e-6, "hdf5", 1e-6, 1e-6) == 1e-6

    with pytest.raises(ValueError, match="different sample intervals"):
        analysis.resolve_sample_interval(1e-6, "manual", 1e-6, 2e-6)
    with pytest.raises(ValueError, match="changed after HDF5"):
        analysis.resolve_sample_interval(2e-6, "hdf5", 1e-6, 1e-6)


def test_configure_xy_analysis_derives_spatial_and_shot_shapes():
    values = default_parameters("xy_plane")
    configured = analysis.configure_analysis("xy_plane", values)

    assert configured == values
    assert analysis.x.shape == (values["nx"],)
    assert analysis.y.shape == (values["ny"],)
    assert analysis.X.shape == (values["ny"], values["nx"])
    assert analysis.Y.shape == (values["ny"], values["nx"])
    assert analysis.nt == (values["sweep_end_index"] - values["sweep_start_index"] + 1)
    assert analysis.n_expected_shots == (values["ny"] * values["nx"] * values["nshots"])
    assert analysis.shot_analysis_mode == "individual"
    expected_scaling = (
        analysis.INTERFEROMETER_PHYSICAL_COEFFICIENT
        * values["f_microwave_GHz"]
        * u.GHz
        * values["interferometer_phase_rad"]
        * u.rad
        / values["N_passes"]
    )
    assert u.allclose(analysis.interferometer_scaling, expected_scaling)
    source_path = analysis.Path(values["filename"])
    assert analysis.diagnostic_plot_output_dir == source_path.with_name(
        f"{source_path.stem}_Langmuir_diagnostic_plots"
    )


def test_diagnostic_plot_directory_is_named_for_and_beside_source_file(tmp_path):
    source_path = tmp_path / "experiment run.hdf5"

    assert analysis.diagnostic_plot_directory(source_path) == (
        tmp_path / "experiment run_Langmuir_diagnostic_plots"
    )


def test_xy_reader_flips_descending_acquisition_y_to_ascending_coordinates():
    class FakeReadResult(dict):
        def __init__(self, signal):
            super().__init__(signal=signal)
            self.dt = type("TimeStep", (), {"value": 1.0e-6})()

    class FakeFile:
        def read_data(self, *_args, **_kwargs):
            # Acquisition order is (+y, x0), (+y, x1), (-y, x0), (-y, x1).
            signal = np.repeat(np.arange(4, dtype=float)[:, np.newaxis], 2, axis=1)
            return FakeReadResult(signal)

    signal, _dt = analysis.read_channel_xy(
        FakeFile(),
        board=1,
        channel=1,
        shotnum_start=1,
        shotnum_end=5,
        ny=2,
        nx=2,
        nshots=1,
        nt_full=2,
        digitizer="SIS crate",
        adc="SIS 3302",
        config_name="config",
        flipup=True,
    )

    # The result's first row is now -y and its last row is +y.
    assert signal[:, :, 0, 0].tolist() == [[2.0, 3.0], [0.0, 1.0]]


def test_prepare_trace_for_analysis_selects_one_individual_shot():
    voltage = np.array([[[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]]])
    current = 2.0 * voltage - 1.0

    selected_voltage, selected_current = analysis.prepare_trace_for_analysis(
        voltage,
        current,
        (0, 1),
        "individual",
        0.25,
    )

    assert np.array_equal(selected_voltage, voltage[0, 1])
    assert np.array_equal(selected_current, current[0, 1])


@pytest.mark.parametrize("spatial_shape", [(3,), (2, 3)])
@pytest.mark.parametrize("ramp_shape", [(), (2,)])
def test_float32_individual_tasks_retain_only_the_selected_trace(
    spatial_shape, ramp_shape
):
    shape = spatial_shape + (2,) + ramp_shape + (16,)
    voltage = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    current = 2.0 * voltage - 1.0
    tasks = []
    for index in np.ndindex(shape[:-1]):
        selected = analysis.prepare_trace_for_analysis(
            voltage, current, index, "individual", 0.25
        )
        tasks.append(selected)
        for trace, source in zip(selected, (voltage, current)):
            assert np.array_equal(trace, source[index])
            assert trace.dtype == np.dtype(float)
            # A small view into a freshly converted full scan retains the whole
            # allocation for every queued task, exhausting memory on XY scans.
            assert trace.flags.owndata

    assert sum(trace.nbytes for task in tasks for trace in task) == (
        2 * voltage.size * np.dtype(float).itemsize
    )


def test_prepare_trace_for_analysis_averages_shots_in_voltage_bins():
    voltage = np.array([[[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]]])
    current = np.array([[[0.0, 2.0, 4.0], [2.0, 4.0, 6.0]]])

    averaged_voltage, averaged_current = analysis.prepare_trace_for_analysis(
        voltage,
        current,
        (0, 0),
        "average",
        0.5,
    )

    assert np.allclose(averaged_voltage, [0.05, 1.05, 2.05])
    assert np.allclose(averaged_current, [1.0, 3.0, 5.0])


def test_averaged_mode_marks_per_shot_fit_products_unavailable():
    assert analysis.analysis_trace_shape((2, 3), 4, "individual") == (2, 3, 4)
    assert analysis.analysis_trace_shape((2, 3), 4, "average") == (2, 3, 1)

    products = analysis.unavailable_per_shot_fit_products((2, 3), 4, 8)

    assert products["te_shot"].shape == (2, 3, 4)
    assert np.all(np.isnan(products["te_shot"]))
    assert products["iv_current_grid_shot"].shape == (2, 3, 4, 8)
    assert np.all(np.isnan(products["iv_current_grid_shot"]))
    assert not np.any(products["analysis_ok_shot"])

    available, stage = analysis.shot_statistics_metadata("average", 4)
    assert not available
    assert "unavailable" in stage

    available, stage = analysis.shot_statistics_metadata("individual", 4)
    assert available
    assert "individual fits" in stage


def test_repeated_ramps_preserve_shot_and_ramp_axes():
    signal = np.arange(2 * 3 * 20).reshape(2, 3, 20)
    extracted = analysis.extract_evenly_spaced_ramps(
        signal,
        sweep_start=2,
        sweep_end=5,
        ramp_count=3,
        ramp_spacing=6,
    )

    assert extracted.shape == (2, 3, 3, 4)
    assert np.array_equal(extracted[:, :, 0], signal[:, :, 2:6])
    assert np.array_equal(extracted[:, :, 2], signal[:, :, 14:18])
    assert analysis.analysis_trace_shape(
        (2,),
        3,
        "individual",
        (4,),
    ) == (2, 3, 4)
    assert analysis.analysis_trace_shape(
        (2,),
        3,
        "average",
        (4,),
    ) == (2, 1, 4)
    products = analysis.unavailable_per_shot_fit_products((2,), 3, 8, (4,))
    assert products["te_shot"].shape == (2, 3, 4)
    assert products["iv_current_grid_shot"].shape == (2, 3, 4, 8)


def test_shot_averaging_keeps_repeated_ramps_independent():
    voltage = np.array(
        [
            [
                [[0.0, 1.0], [10.0, 11.0]],
                [[0.1, 1.1], [10.1, 11.1]],
            ]
        ]
    )
    current = np.array(
        [
            [
                [[0.0, 2.0], [20.0, 22.0]],
                [[2.0, 4.0], [22.0, 24.0]],
            ]
        ]
    )

    averaged_voltage, averaged_current = analysis.prepare_trace_for_analysis(
        voltage,
        current,
        (0, 0, 1),
        "average",
        0.5,
        shot_axis=1,
    )

    assert np.allclose(averaged_voltage, [10.05, 11.05])
    assert np.allclose(averaged_current, [21.0, 23.0])


def test_xline_postprocessing_does_not_mix_ramps():
    x = np.arange(5, dtype=float)
    first = np.array([1.0, 1.0, 20.0, 1.0, 1.0])
    second = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    values = np.column_stack((first, second))

    processed = analysis.apply_optional_x_ramp_postprocessing(
        x,
        values,
        values,
        values,
        enable_vp_spike_rejection=True,
        vp_spike_half_window=1,
        vp_spike_threshold_V=5.0,
        vp_replace_flagged_with_local_interp=True,
        enable_neighbor_smoothing=False,
    )

    assert processed["vp_spike_mask"].shape == (5, 2)
    assert processed["vp_spike_mask"][2, 0]
    assert not np.any(processed["vp_spike_mask"][:, 1])
    assert np.all(processed["vp_processed"][:, 1] == 100.0)


def test_interferometer_calibrates_each_ramp_independently():
    x = np.array([-10.0, 0.0, 10.0])
    density = np.array(
        [
            [1.0, 1.0],
            [2.0, 4.0],
            [1.0, 1.0],
        ]
    ) * u.m**-3
    line_density = 4.0e17 * u.m**-2

    shape_factor, normalized, scaled, scale = (
        analysis.scale_xline_density_ramps_to_interferometer(
            x,
            density,
            line_density,
        )
    )

    assert shape_factor.shape == (2,)
    assert normalized.shape == (3, 2)
    assert scaled.shape == (3, 2)
    assert scale.shape == (2,)
    assert np.allclose(np.nanmax(normalized, axis=0), 1.0)
    assert scale[0] != pytest.approx(scale[1])


@pytest.mark.parametrize("display_mode", ["separate_profiles", "ramp_time_map"])
def test_multi_ramp_xline_summary_rendering(display_mode):
    import matplotlib.pyplot as plt

    x = np.array([-1.0, 0.0, 1.0])
    values = np.arange(6, dtype=float).reshape(3, 2) + 1.0
    figure = analysis.render_xline_multi_ramp_summary_plot(
        x,
        np.array([0.01, 0.02]),
        values * u.eV,
        values * u.V,
        values * u.V,
        values * u.A,
        values * u.A,
        values * u.m**-3,
        display_mode=display_mode,
        te_std=np.ones_like(values) * u.eV,
        analysis_ok=np.ones_like(values, dtype=np.uint8),
        shape_factor=np.array([0.1, 0.2]) * u.m,
    )

    figure.canvas.draw()
    assert len(figure.axes) >= 6
    plt.close(figure)


@pytest.mark.parametrize(
    ("shot_mode", "expected_analysis_traces"),
    (("individual", 12), ("average", 4)),
)
def test_xline_pipeline_exports_repeated_ramp_shapes(
    monkeypatch,
    tmp_path,
    shot_mode,
    expected_analysis_traces,
):
    source_path = tmp_path / f"multi_ramp_{shot_mode}.hdf5"
    with h5py.File(source_path, "w"):
        pass

    values = default_parameters("x_line")
    values.update(
        filename=str(source_path),
        temporal_metadata_source="manual",
        dt_s=2.0e-6,
        nx=2,
        nshots=3,
        nt_full=30,
        sweep_start_index=2,
        sweep_end_index=5,
        nramps=2,
        ramp_start_spacing_samples=10,
        isat_start_index=0,
        isat_end_index=1,
        sg_smooth_bins=3,
        sg_smooth_order=1,
        iv_npts=8,
        shot_analysis_mode=shot_mode,
        calculate_shape_factor=False,
        enable_vp_spike_rejection=False,
        enable_neighbor_smoothing=False,
        parallel_analysis=False,
        diagnostic_plot_every=0,
        make_all_iv_diagnostic_plot=False,
        plot_results=False,
        save_results=True,
    )

    class FakeLapdFile:
        def close(self):
            return None

    monkeypatch.setattr(analysis.lapd, "File", lambda *_args, **_kwargs: FakeLapdFile())

    def fake_read_channel(**kwargs):
        shape = (kwargs["nx"], kwargs["nshots"], kwargs["nt_full"])
        signal = np.arange(np.prod(shape), dtype=float).reshape(shape)
        return signal, 1.0e-6

    monkeypatch.setattr(analysis, "read_channel_xline", fake_read_channel)

    def fake_analyze(task):
        flat_index, idx, *_rest = task
        grid = np.linspace(-2.0, 2.0, values["iv_npts"])
        return {
            "flat_index": flat_index,
            "idx": idx,
            "warnings": [],
            "ok": True,
            "te_eV": 2.0 + idx[-1],
            "vp_V": 1.0,
            "vf_V": -1.0,
            "ies_A": 0.02,
            "iis_A": 0.01,
            "n_e_m3": 1.0e17,
            "te_fit_r2": 0.999,
            "te_fit_rmse": 0.01,
            "te_fit_npts": 8,
            "te_fit_vstart": -1.0,
            "te_fit_vstop": 1.0,
            "te_fit_slope": 0.5,
            "te_fit_intercept": -2.0,
            "te_fit_i0": 0.0,
            "te_fit_passed_r2": True,
            "te_fit_candidate_count": 1,
            "iv_voltage_grid": grid,
            "iv_current_grid": grid * 0.01,
            "iv_didv_grid": np.full(grid.shape, 0.01),
            "iv_fit_mask": np.ones(grid.shape, dtype=np.uint8),
            "diagnostic_data": None,
        }

    monkeypatch.setattr(analysis, "analyze_trace_worker", fake_analyze)
    analysis.run_analysis("x_line", values)

    with h5py.File(source_path, "r") as h5_file:
        group = h5_file["langmuir_xline"]
        assert group["te_eV"].shape == (2, 2)
        assert group["te_shot_eV"].shape == (2, 3, 2)
        assert group["iv_voltage_grid_V"].shape == (2, 2, 8)
        assert group["vsweep_mean_V"].shape == (2, 2, 4)
        assert group["ramp_center_time_s"].shape == (2,)
        assert group.attrs["per_shot_axis_order"] == "x,shot,ramp"
        assert group.attrs["profile_axis_order"] == "x,ramp"
        assert group.attrs["n_analysis_traces"] == expected_analysis_traces
        assert group.attrs["spatial_geometry_source"] == "manual"
        assert group.attrs["bmotion_config_name"] == ""
        assert group.attrs["temporal_metadata_source"] == "manual"
        assert group.attrs["dt_s"] == 2.0e-6


def test_xy_postprocessing_and_calibration_do_not_mix_ramps():
    first = np.ones((3, 3))
    first[1, 1] = 20.0
    second = np.full((3, 3), 100.0)
    values = np.stack((first, second), axis=-1)
    processed = analysis.apply_optional_xy_ramp_postprocessing(
        values,
        values,
        values,
        values,
        values,
        enable_vp_spike_rejection=True,
        vp_spike_half_window=(1, 1),
        vp_spike_threshold_V=5.0,
        vp_replace_flagged_with_local_median=True,
        enable_neighbor_smoothing=False,
    )
    assert processed["vp_spike_mask"].shape == (3, 3, 2)
    assert processed["vp_spike_mask"][1, 1, 0]
    assert not np.any(processed["vp_spike_mask"][..., 1])
    assert np.all(processed["vp_processed"][..., 1] == 100.0)

    x = np.array([-10.0, 0.0, 10.0])
    y = np.array([-1.0, 0.0, 1.0])
    density = np.stack(
        (
            np.tile(np.array([1.0, 2.0, 1.0]), (3, 1)),
            np.tile(np.array([1.0, 4.0, 1.0]), (3, 1)),
        ),
        axis=-1,
    ) * u.m**-3
    calibrated = analysis.scale_xy_density_ramps_to_interferometer(
        x,
        y,
        density,
        4.0e17 * u.m**-2,
    )
    assert calibrated["shape_factor"].shape == (2,)
    assert calibrated["density_scale"].shape == (2,)
    assert calibrated["normalized_xline"].shape == (3, 2)
    assert calibrated["scaled_density_map"].shape == (3, 3, 2)
    assert calibrated["density_scale"][0] != pytest.approx(
        calibrated["density_scale"][1]
    )


def test_multi_ramp_xy_summary_rendering():
    import matplotlib.pyplot as plt

    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([-1.0, 1.0])
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="xy")
    values = np.arange(12, dtype=float).reshape(2, 3, 2) + 1.0
    figures = analysis.render_xy_ramp_summary_plots(
        x_mesh,
        y_mesh,
        np.array([0.01, 0.02]),
        values * u.eV,
        values * u.V,
        values * u.V,
        values * u.A,
        values * u.A,
        values * u.m**-3,
        analysis_ok=np.ones_like(values, dtype=np.uint8),
        shape_factor=np.array([0.1, 0.2]) * u.m,
    )
    assert len(figures) == 2
    for figure in figures:
        figure.canvas.draw()
        assert len(figure.axes) >= 6
        plt.close(figure)


@pytest.mark.parametrize(
    ("shot_mode", "expected_analysis_traces"),
    (("individual", 24), ("average", 12)),
)
def test_xy_pipeline_exports_repeated_ramp_shapes(
    monkeypatch,
    tmp_path,
    shot_mode,
    expected_analysis_traces,
):
    source_path = tmp_path / f"xy_multi_ramp_{shot_mode}.hdf5"
    with h5py.File(source_path, "w"):
        pass

    values = default_parameters("xy_plane")
    values.update(
        filename=str(source_path),
        temporal_metadata_source="manual",
        dt_s=2.0e-6,
        ny=2,
        nx=3,
        nshots=2,
        nt_full=30,
        sweep_start_index=2,
        sweep_end_index=5,
        nramps=2,
        ramp_start_spacing_samples=10,
        isat_start_index=0,
        isat_end_index=1,
        sg_smooth_bins=3,
        sg_smooth_order=1,
        iv_npts=8,
        shot_analysis_mode=shot_mode,
        calculate_shape_factor=False,
        enable_vp_spike_rejection=False,
        enable_neighbor_smoothing=False,
        parallel_analysis=False,
        diagnostic_plot_every=0,
        make_all_iv_diagnostic_plot=False,
        plot_results=False,
        save_results=True,
    )

    class FakeLapdFile:
        def close(self):
            return None

    monkeypatch.setattr(analysis.lapd, "File", lambda *_args, **_kwargs: FakeLapdFile())

    def fake_read_channel(**kwargs):
        shape = (
            kwargs["ny"],
            kwargs["nx"],
            kwargs["nshots"],
            kwargs["nt_full"],
        )
        signal = np.arange(np.prod(shape), dtype=float).reshape(shape)
        return signal, 1.0e-6

    monkeypatch.setattr(analysis, "read_channel_xy", fake_read_channel)

    def fake_analyze(task):
        flat_index, idx, *_rest = task
        grid = np.linspace(-2.0, 2.0, values["iv_npts"])
        return {
            "flat_index": flat_index,
            "idx": idx,
            "warnings": [],
            "ok": True,
            "te_eV": 2.0 + idx[-1],
            "vp_V": 1.0,
            "vf_V": -1.0,
            "ies_A": 0.02,
            "iis_A": 0.01,
            "n_e_m3": 1.0e17,
            "te_fit_r2": 0.999,
            "te_fit_rmse": 0.01,
            "te_fit_npts": 8,
            "te_fit_vstart": -1.0,
            "te_fit_vstop": 1.0,
            "te_fit_slope": 0.5,
            "te_fit_intercept": -2.0,
            "te_fit_i0": 0.0,
            "te_fit_passed_r2": True,
            "te_fit_candidate_count": 1,
            "iv_voltage_grid": grid,
            "iv_current_grid": grid * 0.01,
            "iv_didv_grid": np.full(grid.shape, 0.01),
            "iv_fit_mask": np.ones(grid.shape, dtype=np.uint8),
            "diagnostic_data": None,
        }

    monkeypatch.setattr(analysis, "analyze_trace_worker", fake_analyze)
    analysis.run_analysis("xy_plane", values)

    with h5py.File(source_path, "r") as h5_file:
        group = h5_file["langmuir_xy"]
        assert group["te_eV"].shape == (2, 3, 2)
        assert group["te_shot_eV"].shape == (2, 3, 2, 2)
        assert group["iv_voltage_grid_V"].shape == (2, 3, 2, 8)
        assert group["vsweep_mean_V"].shape == (2, 3, 2, 4)
        assert group["interferometer_xline_n_e_raw_m3"].shape == (3, 2)
        assert group["ramp_center_time_s"].shape == (2,)
        assert group.attrs["per_shot_axis_order"] == "y,x,shot,ramp"
        assert group.attrs["profile_axis_order"] == "y,x,ramp"
        assert group.attrs["n_analysis_traces"] == expected_analysis_traces
        assert group.attrs["spatial_geometry_source"] == "manual"
        assert group.attrs["bmotion_config_name"] == ""
        assert group.attrs["temporal_metadata_source"] == "manual"
        assert group.attrs["dt_s"] == 2.0e-6
        assert group.attrs["y_acquisition_order"] == "descending"

    npz_path = source_path.with_name(f"{source_path.stem}_langmuir_xy.npz")
    with np.load(npz_path, allow_pickle=False) as result:
        assert result["te_eV"].shape == (2, 3, 2)
        assert result["te_shot_eV"].shape == (2, 3, 2, 2)
        assert result["xy_ramp_shape_info"].tolist() == [2, 3, 2, 2, 4, 8]
        assert result["profile_axis_order"].item() == "y,x,ramp"
        assert result["spatial_geometry_source"].item() == "manual"
        assert result["bmotion_config_name"].item() == ""
        assert result["temporal_metadata_source"].item() == "manual"
        assert result["y_acquisition_order"].item() == "descending"


def test_select_xline_from_xy_map_uses_nearest_y_coordinate():
    y = np.array([-4.0, -1.0, 2.0, 5.0])
    density = np.arange(12, dtype=float).reshape(4, 3) * u.m**-3

    index, selected_y, profile = analysis.select_xline_from_xy_map(
        y,
        density,
        requested_y_cm=0.2,
    )

    assert index == 1
    assert selected_y == pytest.approx(-1.0)
    assert u.allclose(profile, density[1])


def test_xy_interferometer_scaling_uses_selected_xline_shape_factor():
    x = np.array([-10.0, 0.0, 10.0])
    y = np.array([-2.0, 1.0, 4.0])
    density = (
        np.array(
            [
                [0.5, 1.0, 0.5],
                [1.0, 2.0, 1.0],
                [1.5, 3.0, 1.5],
            ]
        )
        * 1e16
        * u.m**-3
    )
    line_integrated_density = 6e15 * u.m**-2

    result = analysis.scale_xy_density_map_to_interferometer(
        x,
        y,
        density,
        line_integrated_density,
        requested_y_cm=0.0,
    )

    assert result["selected_y_index"] == 1
    assert result["selected_y_cm"] == pytest.approx(1.0)
    assert result["shape_factor"].to_value(u.m) == pytest.approx(0.15)
    assert result["density_scale"] == pytest.approx(2.0)
    assert np.allclose(result["normalized_xline"], [0.5, 1.0, 0.5])
    assert u.allclose(result["scaled_xline"], density[1] * 2.0)
    assert u.allclose(result["scaled_density_map"], density * 2.0)


@pytest.mark.parametrize(
    ("y, density, message"),
    [
        ([], np.empty((0, 2)), "non-empty one-dimensional"),
        ([0.0, 1.0], np.empty((3, 2)), "shape"),
    ],
)
def test_select_xline_from_xy_map_rejects_inconsistent_shapes(y, density, message):
    with pytest.raises(ValueError, match=message):
        analysis.select_xline_from_xy_map(y, density)


@pytest.mark.parametrize("empty", [False, True])
def test_xy_summary_annotations_stay_in_narrow_side_columns(empty):
    import matplotlib.pyplot as plt

    x_mesh, y_mesh = np.meshgrid(np.linspace(-10, 10, 5), np.linspace(-10, 10, 5))
    values = np.full((5, 5), np.nan) if empty else np.arange(25).reshape(5, 5) + 1.0
    figure = analysis.render_xy_summary_plot(
        x_mesh, y_mesh, values * u.eV, values * u.V, values * u.V,
        values * u.A, values * u.A, values * u.m**-3,
        analysis_ok=np.zeros((5, 5), dtype=bool),
        te_fit_r2=np.full((5, 5), 0.1), te_poor_fit_r2=0.9,
        vp_spike_mask=np.ones((5, 5), dtype=bool),
        shape_factor=0.12 * u.m, interferometer_profile_y_cm=0,
    )
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for index in range(6):
            map_ax, _colorbar_ax, notes_ax = figure.axes[3 * index : 3 * index + 3]
            assert map_ax.get_legend() is None
            assert not map_ax.texts
            assert notes_ax.get_position().x0 > map_ax.get_position().x1
            assert notes_ax.get_position().width < 0.1
            legend = notes_ax.get_legend()
            if legend is not None:
                bounds = legend.get_window_extent(renderer)
                assert bounds.x0 > map_ax.get_window_extent(renderer).x1
                assert bounds.x1 <= notes_ax.get_window_extent(renderer).x1 + 1
            if empty:
                assert any("No accepted" in text.get_text() for text in notes_ax.texts)
        if not empty:
            assert len(figure.axes[2].get_legend().get_texts()) == 2
    finally:
        plt.close(figure)


def test_xy_summary_title_fits_saved_figure_with_long_status():
    import matplotlib.pyplot as plt

    x_mesh, y_mesh = np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
    values = np.ones((3, 3))
    title = "Experiment 2026-10-01 run 1234 - Langmuir XY-plane Summary"
    status = (
        "Reported 120/140 finite per-shot fits; accepted 95/140 per-shot fits; "
        "top rejection (45): insufficient points in the selected electron "
        "temperature fit interval after excluding invalid current samples"
    )
    figure = analysis.render_xy_summary_plot(
        x_mesh, y_mesh, values * u.eV, values * u.V, values * u.V,
        values * u.A, values * u.A, analysis_status=status, title=title,
    )
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bounds = figure._suptitle.get_window_extent(renderer)
        width, height = figure.canvas.get_width_height()
        assert 0 <= bounds.x0 < bounds.x1 <= width
        assert 0 <= bounds.y0 < bounds.y1 <= height
        assert " ".join(figure._suptitle.get_text().split()) == f"{title} {status}"
    finally:
        plt.close(figure)
