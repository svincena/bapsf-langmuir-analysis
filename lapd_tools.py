#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:50:21 2026

@author: ChatGPT 5.5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.signal import savgol_filter, butter, sosfiltfilt


def smooth_time_series(
    A,
    method="moving",
    *,
    time_axis=-1,
    window_size=11,
    polyorder=3,
    sigma=2.0,
    cutoff=None,
    fs=None,
    dt=None,
    butter_order=4,
    mode="nearest",
    nan_policy="propagate",
):
    """
    Smooth a numpy array A(ny, nx, nt) along the time dimension only.

    Parameters
    ----------
    A : ndarray
        Input array, typically shape (ny, nx, nt).

    method : {"moving", "savgol", "gaussian", "butterworth"}
        Smoothing method.

        "moving"
            Centered moving-average window of size `window_size`.

        "savgol"
            Savitzky-Golay filter using `window_size` and `polyorder`.
            Good for preserving peak shapes and derivatives.

        "gaussian"
            Gaussian smoothing along time using standard deviation `sigma`,
            given in samples. Good general-purpose time-series smoothing.

        "butterworth"
            Zero-phase Butterworth low-pass filter using `cutoff`, `fs` or `dt`,
            and `butter_order`. Good when you know the physical frequency cutoff.

    time_axis : int
        Axis corresponding to time. Default is -1.

    window_size : int
        Window size in samples for "moving" and "savgol".
        For Savitzky-Golay, this must be odd and larger than `polyorder`.

    polyorder : int
        Polynomial order for Savitzky-Golay smoothing.

    sigma : float
        Gaussian sigma in samples for "gaussian".

    cutoff : float
        Low-pass cutoff frequency for "butterworth", in Hz.

    fs : float, optional
        Sampling frequency in Hz. Use either `fs` or `dt`.

    dt : float, optional
        Sampling interval in seconds. If supplied, fs = 1 / dt.

    butter_order : int
        Order of the Butterworth low-pass filter.

    mode : str
        Boundary handling mode for moving average and Gaussian filters.
        Common choices are "nearest", "reflect", "mirror", "constant", "wrap".

    nan_policy : {"propagate", "interp"}
        How to handle NaNs.

        "propagate"
            Leave NaNs as-is. Most filters will spread NaNs locally.

        "interp"
            Linearly interpolate NaNs along the time axis before smoothing.
            Pixels with all-NaN time traces remain NaN.

    Returns
    -------
    Asmooth : ndarray
        Smoothed array with the same shape as A.

    Notes
    -----
    The smoothing is applied independently to every spatial pixel A[y, x, :].
    The spatial dimensions are not mixed.
    """

    A = np.asarray(A)

    if A.ndim != 3:
        raise ValueError("Expected a 3D array with shape like (ny, nx, nt).")

    if not np.issubdtype(A.dtype, np.number):
        raise TypeError("A must contain numeric data.")

    # Work in floating point so filters behave predictably
    B = A.astype(float, copy=True)

    time_axis = np.core.multiarray.normalize_axis_index(time_axis, B.ndim)
    nt = B.shape[time_axis]

    if nt < 2:
        raise ValueError("Time axis must contain at least 2 samples.")

    if nan_policy not in {"propagate", "interp"}:
        raise ValueError("nan_policy must be either 'propagate' or 'interp'.")

    if nan_policy == "interp":
        B = _interp_nans_along_axis(B, axis=time_axis)

    method = method.lower()

    if method in {"moving", "boxcar", "mean"}:
        if window_size < 1:
            raise ValueError("window_size must be >= 1.")

        return uniform_filter1d(
            B,
            size=window_size,
            axis=time_axis,
            mode=mode,
            origin=0,
        )

    elif method in {"savgol", "savitzky-golay", "savitzky_golay"}:
        if window_size % 2 == 0:
            raise ValueError("For Savitzky-Golay, window_size must be odd.")

        if window_size <= polyorder:
            raise ValueError("For Savitzky-Golay, window_size must be > polyorder.")

        if window_size > nt:
            raise ValueError("For Savitzky-Golay, window_size cannot exceed nt.")

        return savgol_filter(
            B,
            window_length=window_size,
            polyorder=polyorder,
            axis=time_axis,
            mode="interp",
        )

    elif method in {"gaussian", "gauss"}:
        if sigma <= 0:
            raise ValueError("sigma must be positive.")

        return gaussian_filter1d(
            B,
            sigma=sigma,
            axis=time_axis,
            mode=mode,
        )

    elif method in {"butterworth", "butter", "lowpass"}:
        if cutoff is None:
            raise ValueError("For Butterworth smoothing, cutoff must be supplied.")

        if fs is None:
            if dt is None:
                raise ValueError("For Butterworth smoothing, supply either fs or dt.")
            fs = 1.0 / dt

        if fs <= 0:
            raise ValueError("fs must be positive.")

        nyquist = 0.5 * fs

        if not (0 < cutoff < nyquist):
            raise ValueError(
                f"cutoff must satisfy 0 < cutoff < Nyquist frequency = {nyquist:g}."
            )

        sos = butter(
            butter_order,
            cutoff,
            btype="lowpass",
            fs=fs,
            output="sos",
        )

        # sosfiltfilt needs enough points for padding.
        # This keeps it from failing for moderately short time traces.
        padlen = min(nt - 1, 3 * (2 * sos.shape[0] + 1))

        if padlen < 1:
            raise ValueError("Time axis is too short for zero-phase Butterworth filter.")

        return sosfiltfilt(
            sos,
            B,
            axis=time_axis,
            padlen=padlen,
        )

    else:
        raise ValueError(
            "Unknown method. Use one of: "
            "'moving', 'savgol', 'gaussian', or 'butterworth'."
        )


def _interp_nans_along_axis(A, axis=-1):
    """
    Linearly interpolate NaNs along one axis.
    Traces that are entirely NaN remain NaN.
    """
    A = np.asarray(A, dtype=float)
    axis = np.core.multiarray.normalize_axis_index(axis, A.ndim)

    # Move interpolation axis to the end
    B = np.moveaxis(A, axis, -1)
    original_shape = B.shape
    nt = original_shape[-1]

    B2 = B.reshape(-1, nt)
    t = np.arange(nt)

    for row in B2:
        good = np.isfinite(row)

        if good.all():
            continue

        if not good.any():
            continue

        row[~good] = np.interp(t[~good], t[good], row[good])

    B = B2.reshape(original_shape)

    return np.moveaxis(B, -1, axis)

def correct_ByBz_probe_rotation(by, bz, remove_mean=True, return_diagnostics=True):
    """
    Estimate and correct an unknown rotation mixing measured By and Bz.

    Assumption:
        At this spatial location, the true lab-frame By fluctuation is large
        and the true lab-frame Bz fluctuation is small.

    Parameters
    ----------
    by, bz : 1D numpy arrays
        Measured magnetic fluctuation time series.
    remove_mean : bool
        If True, subtract the mean before estimating the rotation angle.
        The final rotation is applied to the original arrays.
    return_diagnostics : bool
        If True, return useful diagnostic quantities.

    Returns
    -------
    by_prime, bz_prime : 1D numpy arrays
        Estimated lab-frame components.
    theta : float
        Rotation angle in radians.
    diagnostics : dict, optional
        Contains angle in degrees, covariance matrix, variances, etc.
    """

    by = np.asarray(by)
    bz = np.asarray(bz)

    if by.shape != bz.shape:
        raise ValueError("by and bz must have the same shape")

    if by.ndim != 1:
        raise ValueError("by and bz should be 1D arrays of shape (nt,)")

    # Use finite values only to estimate the rotation
    good = np.isfinite(by) & np.isfinite(bz)
    if np.count_nonzero(good) < 3:
        raise ValueError("Not enough finite data points to estimate rotation")

    by_fit = by[good].astype(float)
    bz_fit = bz[good].astype(float)

    if remove_mean:
        by_fit = by_fit - np.mean(by_fit)
        bz_fit = bz_fit - np.mean(bz_fit)

    # Covariance elements
    Cyy = np.mean(by_fit * by_fit)
    Czz = np.mean(bz_fit * bz_fit)
    Cyz = np.mean(by_fit * bz_fit)

    # Principal-axis rotation angle
    theta = 0.5 * np.arctan2(2.0 * Cyz, Cyy - Czz)

    c = np.cos(theta)
    s = np.sin(theta)

    # Apply rotation to the original measured data
    by_prime = c * by + s * bz
    bz_prime = -s * by + c * bz

    # Optional sign convention:
    # make By' positively correlated with the original measured By
    corr = np.nanmean((by_prime[good] - np.nanmean(by_prime[good])) *
                      (by[good] - np.nanmean(by[good])))

    # if corr < 0:
    #     by_prime *= -1
    #     bz_prime *= -1
    #     theta += np.pi

    if not return_diagnostics:
        return by_prime, bz_prime, theta

    diagnostics = {
        "theta_rad": theta,
        "theta_deg": np.degrees(theta),
        "Cyy": Cyy,
        "Czz": Czz,
        "Cyz": Cyz,
        "var_by_measured": np.nanvar(by),
        "var_bz_measured": np.nanvar(bz),
        "var_by_prime": np.nanvar(by_prime),
        "var_bz_prime": np.nanvar(bz_prime),
        "power_ratio_before": np.nanvar(by) / np.nanvar(bz),
        "power_ratio_after": np.nanvar(by_prime) / np.nanvar(bz_prime),
    }

    return by_prime, bz_prime, theta, diagnostics



# --------------   animate_vector_field ---------------

def animate_vector_field(
    x,
    y,
    u,
    v,
    t=None,
    save_path=None,
    cmap="viridis",
    indexing="xy",
    stride=1,
    scale=None,
    width=0.003,
    color_by_magnitude=True,
    colorbar_label=None,
    target_arrow_fraction=0.7,
    time_units="s",
    time_sigfigs=4,
    time_label="t",
    main_title="Vector Field",
    subtitle=None,
    x_axis_title=r"$x$",
    y_axis_title=r"$y$",
    interval=50,
    fps=30,
):
    """
    Animate a two-dimensional vector field using Matplotlib's quiver plotting.

    This function animates a vector field with components u(x, y, t) and
    v(x, y, t). It is intended for data stored as two 3D NumPy arrays, where
    the first two dimensions are spatial and the third dimension is time.

    By default, this function assumes Cartesian / image-style meshgrid
    indexing, i.e. indexing="xy". With this convention:

        X, Y = np.meshgrid(x, y, indexing="xy")

    and the expected data shape is:

        u.shape = v.shape = (len(y), len(x), nt)

    where nt is the number of time steps.

    If your data are stored using matrix / physics-style indexing, where the
    first dimension corresponds to x and the second dimension corresponds to y,
    use indexing="ij". In that case, the expected shape is:

        u.shape = v.shape = (len(x), len(y), nt)

    Parameters
    ----------
    x : array_like, shape (nx,)
        One-dimensional array of x positions.

    y : array_like, shape (ny,)
        One-dimensional array of y positions.

    u : array_like, shape (ny, nx, nt) or (nx, ny, nt)
        First vector-field component.

        For indexing="xy", this should have shape:

            (len(y), len(x), nt)

        For indexing="ij", this should have shape:

            (len(x), len(y), nt)

    v : array_like, same shape as u
        Second vector-field component.

    t : array_like, shape (nt,), optional
        One-dimensional NumPy array of floating-point time values. These values
        are used only for the displayed time annotation. The values may be in
        seconds, milliseconds, microseconds, shot-relative time, delay time, or
        any other desired floating-point time coordinate.

        If t is None, frame number is used instead.

    save_path : str or None, optional
        Path to save the animation. For example:

            "./BxBy.mp4"

        If None, the animation is not saved.

    cmap : str, optional
        Colormap used when color_by_magnitude=True.

    indexing : {"xy", "ij"}, optional
        Indexing convention used to create the coordinate grid.

        Use indexing="xy" when your data shape is:

            (len(y), len(x), nt)

        Use indexing="ij" when your data shape is:

            (len(x), len(y), nt)

        Default is "xy".

    stride : int, optional
        Plot every stride-th vector in each spatial direction. Increasing this
        value reduces the number of arrows and can make dense vector fields
        easier to view.

    scale : float or None, optional
        Matplotlib quiver scale parameter. With scale_units="xy", the displayed
        arrow length is approximately vector_magnitude / scale in data units.

        If scale is None, this function estimates a scale from the 95th
        percentile vector magnitude.

    width : float, optional
        Width of the quiver arrows.

    color_by_magnitude : bool, optional
        If True, arrow color represents vector magnitude sqrt(u**2 + v**2).
        If False, all arrows use the default single-color quiver style.

    colorbar_label : str or None, optional
        Label for the colorbar when color_by_magnitude=True.

        If None, the default label is:

            r"$\\sqrt{u^2 + v^2}$"

        This parameter is ignored when color_by_magnitude=False, because no
        colorbar is created in that case.

    target_arrow_fraction : float, optional
        Used only when scale=None. This sets the approximate size of a typical
        large arrow relative to the grid spacing. Larger values make arrows
        longer. Smaller values make arrows shorter.

    time_units : str, optional
        Units displayed after the time value. Examples:

            "s", "ms", "us", r"$\\mu$s"

    time_sigfigs : int, optional
        Number of significant figures displayed in the time annotation.

    time_label : str, optional
        Label used in the time annotation. Default is "t".

    main_title : str or None, optional
        Main title placed above the plot using fig.suptitle(). Use None to
        suppress the main title.

    subtitle : str or None, optional
        Subtitle placed directly above the axes using ax.set_title(). Use None
        for no subtitle.

    x_axis_title : str, optional
        Label for the x axis.

    y_axis_title : str, optional
        Label for the y axis.

    interval : int, optional
        Delay between animation frames in milliseconds.

    fps : int, optional
        Frames per second used when saving the animation.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object. Keep a reference to this object to prevent the
        animation from being garbage-collected before rendering.

    Notes
    -----
    Nonfinite values in u and v, such as NaN or Inf, are replaced by zero for
    plotting purposes only. The input arrays are not modified.

    The function uses quiver.set_UVC() to update the arrows efficiently without
    clearing and redrawing the axes at every frame.

    The default indexing is "xy", so the default expected array shape is:

        (len(y), len(x), nt)

    For many physics-style arrays, the data may instead be shaped as:

        (len(x), len(y), nt)

    In that case, call the function with indexing="ij".

    Examples
    --------
    For data with shape (ny, nx, nt):

        anim = animate_vector_field(
            x,
            y,
            bx,
            by,
            t=t_us,
            time_units=r"$\\mu$s",
            time_sigfigs=4,
            main_title="Magnetic Field Fluctuations",
            subtitle=r"$B_x$ and $B_y$",
            x_axis_title="x [cm]",
            y_axis_title="y [cm]",
            colorbar_label=r"$|\\delta B_\\perp|$ [G]",
            save_path="./BxBy.mp4",
        )

    For data with shape (nx, ny, nt):

        anim = animate_vector_field(
            x,
            y,
            bx,
            by,
            t=t_us,
            indexing="ij",
            time_units=r"$\\mu$s",
            colorbar_label=r"$|\\delta B_\\perp|$ [G]",
            save_path="./BxBy.mp4",
        )
    """

    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    v = np.asarray(v)

    if indexing not in ("xy", "ij"):
        raise ValueError("indexing must be either 'xy' or 'ij'.")

    if stride < 1:
        raise ValueError("stride must be an integer >= 1.")

    if u.shape != v.shape:
        raise ValueError(
            f"u and v must have the same shape. Got u.shape={u.shape} "
            f"and v.shape={v.shape}."
        )

    if u.ndim != 3:
        raise ValueError(
            f"u and v must be 3D arrays with shape (ny, nx, nt) for "
            f"indexing='xy' or (nx, ny, nt) for indexing='ij'. "
            f"Got u.ndim={u.ndim}."
        )

    nt = u.shape[2]

    if t is None:
        t = np.arange(nt, dtype=float)
        time_units = "frame"

    t = np.asarray(t, dtype=float)

    if t.ndim != 1:
        raise ValueError(f"t must be a 1D array. Got t.ndim={t.ndim}.")

    if len(t) != nt:
        raise ValueError(
            f"len(t) must equal the number of time frames. "
            f"Got len(t)={len(t)}, but nt={nt}."
        )

    if time_sigfigs < 1:
        raise ValueError("time_sigfigs must be at least 1.")

    def format_time(frame):
        return f"{time_label} = {t[frame]:.{time_sigfigs}g} {time_units}"

    # Replace NaN/Inf for plotting only.
    u_plot = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v_plot = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    X, Y = np.meshgrid(x, y, indexing=indexing)

    if u_plot.shape[:2] != X.shape:
        raise ValueError(
            f"Data shape {u_plot.shape[:2]} does not match grid shape {X.shape}. "
            f"For indexing='xy', expected data shape is "
            f"({len(y)}, {len(x)}, nt). "
            f"For indexing='ij', expected data shape is "
            f"({len(x)}, {len(y)}, nt)."
        )

    Xs = X[::stride, ::stride]
    Ys = Y[::stride, ::stride]

    us = u_plot[::stride, ::stride, :]
    vs = v_plot[::stride, ::stride, :]

    mag = np.sqrt(us**2 + vs**2)
    mag_max = np.nanmax(mag)

    if mag_max <= 0:
        raise ValueError("All plotted vectors have zero magnitude. Nothing to animate.")

    mag_ref = np.nanpercentile(mag, 95)

    frame_mags = np.nanmax(mag, axis=(0, 1))
    nonzero_frames = np.where(frame_mags > 0)[0]
    init_frame = int(nonzero_frames[0]) if len(nonzero_frames) > 0 else 0

    u0 = us[:, :, init_frame]
    v0 = vs[:, :, init_frame]
    mag0 = mag[:, :, init_frame]

    if scale is None:
        dx = np.nanmedian(np.abs(np.diff(x)))
        dy = np.nanmedian(np.abs(np.diff(y)))
        dxy = min(dx, dy)

        target_arrow_length = target_arrow_fraction * stride * dxy

        if mag_ref > 0 and target_arrow_length > 0:
            scale = mag_ref / target_arrow_length
        else:
            scale = 1.0

        print(f"Using quiver scale = {scale:.6g}")

    if color_by_magnitude:
        quiv = ax.quiver(
            Xs,
            Ys,
            u0,
            v0,
            mag0,
            cmap=cmap,
            clim=(0, mag_max),
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=width,
        )

        cbar = plt.colorbar(quiv, ax=ax)

        if colorbar_label is None:
            colorbar_label = r"$\sqrt{u^2 + v^2}$"

        cbar.set_label(colorbar_label)

    else:
        quiv = ax.quiver(
            Xs,
            Ys,
            u0,
            v0,
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=width,
        )

    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        color="black",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7),
    )

    if main_title is not None:
        fig.suptitle(main_title, fontsize=14, fontweight="bold")

    if subtitle is not None:
        ax.set_title(subtitle, fontsize=11)

    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(np.nanmin(X), np.nanmax(X))
    ax.set_ylim(np.nanmin(Y), np.nanmax(Y))

    def update(frame):
        uf = us[:, :, frame]
        vf = vs[:, :, frame]

        if color_by_magnitude:
            magf = mag[:, :, frame]
            quiv.set_UVC(uf, vf, magf)
        else:
            quiv.set_UVC(uf, vf)

        time_text.set_text(format_time(frame))

        return [quiv, time_text]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=nt,
        interval=interval,
        blit=False,
    )

    if save_path:
        anim.save(save_path, writer="ffmpeg", fps=fps)
        print(f"Saved to {save_path}")

    plt.show()

    return anim



# -------------------- animate scalar field ---------------------------

def animate_scalar_field(
    x,
    y,
    s,
    t=None,
    save_path=None,
    cmap="seismic",
    indexing="xy",
    stride=1,
    vmin=None,
    vmax=None,
    symmetric_colorbar=False,
    colorbar_label=None,
    shading="auto",
    time_units="s",
    time_sigfigs=4,
    time_label="t",
    main_title="Scalar Field",
    subtitle=None,
    x_axis_title=r"$x$",
    y_axis_title=r"$y$",
    interval=50,
    fps=30,
):
    """
    Animate a two-dimensional scalar field using Matplotlib's pcolormesh.

    This function animates a scalar field s(x, y, t). It is intended for data
    stored as a 3D NumPy array, where the first two dimensions are spatial and
    the third dimension is time.

    By default, this function assumes Cartesian / image-style meshgrid indexing,
    i.e. indexing="xy". With this convention:

        X, Y = np.meshgrid(x, y, indexing="xy")

    and the expected scalar-field shape is:

        s.shape = (len(y), len(x), nt)

    where nt is the number of time steps.

    If your data are stored using matrix / physics-style indexing, where the
    first dimension corresponds to x and the second dimension corresponds to y,
    use indexing="ij". In that case, the expected scalar-field shape is:

        s.shape = (len(x), len(y), nt)

    Parameters
    ----------
    x : array_like, shape (nx,)
        One-dimensional array of x positions.

    y : array_like, shape (ny,)
        One-dimensional array of y positions.

    s : array_like, shape (ny, nx, nt) or (nx, ny, nt)
        Scalar field to animate.

        For indexing="xy", this should have shape:

            (len(y), len(x), nt)

        For indexing="ij", this should have shape:

            (len(x), len(y), nt)

    t : array_like, shape (nt,), optional
        One-dimensional floating-point time array. These values are used only
        for the displayed time annotation.

        If t is None, frame number is used instead.

    save_path : str or None, optional
        Path to save the animation. For example:

            "./scalar_field.mp4"

        If None, the animation is not saved.

    cmap : str, optional
        Matplotlib colormap name.

    indexing : {"xy", "ij"}, optional
        Indexing convention used to create the coordinate grid.

        Use indexing="xy" when s.shape is:

            (len(y), len(x), nt)

        Use indexing="ij" when s.shape is:

            (len(x), len(y), nt)

        Default is "xy".

    stride : int, optional
        Plot every stride-th grid point in each spatial direction. This can be
        useful for very large arrays. Default is 1.

    vmin, vmax : float or None, optional
        Color limits. If either is None, it is computed from the full scalar
        field after NaN/Inf handling.

    symmetric_colorbar : bool, optional
        If True, force color limits to be symmetric around zero using the
        largest absolute value. This is often useful for fluctuation fields.

    colorbar_label : str or None, optional
        Label for the colorbar. If None, a generic label is used.

    shading : str, optional
        Shading option passed to pcolormesh. Default is "auto".

    time_units : str, optional
        Units displayed after the time value. Examples:

            "s", "ms", "us", r"$\\mu$s"

    time_sigfigs : int, optional
        Number of significant figures displayed in the time annotation.

    time_label : str, optional
        Label used in the time annotation. Default is "t".

    main_title : str or None, optional
        Main title placed above the plot using fig.suptitle(). Use None to
        suppress the main title.

    subtitle : str or None, optional
        Subtitle placed directly above the axes using ax.set_title(). Use None
        for no subtitle.

    x_axis_title : str, optional
        Label for the x axis.

    y_axis_title : str, optional
        Label for the y axis.

    interval : int, optional
        Delay between animation frames in milliseconds.

    fps : int, optional
        Frames per second used when saving the animation.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object. Keep a reference to this object to prevent the
        animation from being garbage-collected before rendering.

    Notes
    -----
    Nonfinite values in s, such as NaN or Inf, are replaced by zero for plotting
    purposes only. The input array is not modified.

    The function updates the existing pcolormesh object using set_array(), which
    is much faster than clearing and redrawing the full axes at every frame.

    Examples
    --------
    For data with shape (ny, nx, nt):

        anim = animate_scalar_field(
            x,
            y,
            density_fluctuation,
            t=t_us,
            time_units=r"$\\mu$s",
            time_sigfigs=4,
            main_title="Density Fluctuation",
            subtitle=r"$\\delta n / n_0$",
            x_axis_title="x [cm]",
            y_axis_title="y [cm]",
            colorbar_label=r"$\\delta n / n_0$",
            save_path="./density_fluctuation.mp4",
        )

    For data with shape (nx, ny, nt):

        anim = animate_scalar_field(
            x,
            y,
            density_fluctuation,
            t=t_us,
            indexing="ij",
            time_units=r"$\\mu$s",
        )
    """

    x = np.asarray(x)
    y = np.asarray(y)
    s = np.asarray(s)

    if indexing not in ("xy", "ij"):
        raise ValueError("indexing must be either 'xy' or 'ij'.")

    if s.ndim != 3:
        raise ValueError(
            f"s must be a 3D array with shape (ny, nx, nt) for "
            f"indexing='xy' or (nx, ny, nt) for indexing='ij'. "
            f"Got s.ndim={s.ndim}."
        )

    if stride < 1:
        raise ValueError("stride must be an integer >= 1.")

    nt = s.shape[2]

    if t is None:
        t = np.arange(nt, dtype=float)
        time_units = "frame"

    t = np.asarray(t, dtype=float)

    if t.ndim != 1:
        raise ValueError(f"t must be a 1D array. Got t.ndim={t.ndim}.")

    if len(t) != nt:
        raise ValueError(
            f"len(t) must equal the number of time frames. "
            f"Got len(t)={len(t)}, but nt={nt}."
        )

    if time_sigfigs < 1:
        raise ValueError("time_sigfigs must be at least 1.")

    def format_time(frame):
        return f"{time_label} = {t[frame]:.{time_sigfigs}g} {time_units}"

    # Replace NaN/Inf for plotting only.
    s_plot = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    X, Y = np.meshgrid(x, y, indexing=indexing)

    if s_plot.shape[:2] != X.shape:
        raise ValueError(
            f"Data shape {s_plot.shape[:2]} does not match grid shape {X.shape}. "
            f"For indexing='xy', expected data shape is "
            f"({len(y)}, {len(x)}, nt). "
            f"For indexing='ij', expected data shape is "
            f"({len(x)}, {len(y)}, nt)."
        )

    Xs = X[::stride, ::stride]
    Ys = Y[::stride, ::stride]
    ss = s_plot[::stride, ::stride, :]

    # Determine color limits.
    if symmetric_colorbar:
        max_abs = np.nanmax(np.abs(ss))

        if vmin is None and vmax is None:
            vmin = -max_abs
            vmax = max_abs
        elif vmin is None:
            vmin = -abs(vmax)
        elif vmax is None:
            vmax = abs(vmin)

    else:
        if vmin is None:
            vmin = np.nanmin(ss)
        if vmax is None:
            vmax = np.nanmax(ss)

    if vmin == vmax:
        # Avoid invalid color scaling for constant fields.
        delta = 1.0 if vmin == 0 else 0.01 * abs(vmin)
        vmin -= delta
        vmax += delta

    mesh = ax.pcolormesh(
        Xs,
        Ys,
        ss[:, :, 0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading=shading,
    )

    cbar = plt.colorbar(mesh, ax=ax)

    if colorbar_label is None:
        colorbar_label = "Scalar field"

    cbar.set_label(colorbar_label)

    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        color="black",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7),
    )

    if main_title is not None:
        fig.suptitle(main_title, fontsize=14, fontweight="bold")

    if subtitle is not None:
        ax.set_title(subtitle, fontsize=11)

    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(np.nanmin(X), np.nanmax(X))
    ax.set_ylim(np.nanmin(Y), np.nanmax(Y))

    def update(frame):
        mesh.set_array(ss[:, :, frame].ravel())
        time_text.set_text(format_time(frame))

        return [mesh, time_text]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=nt,
        interval=interval,
        blit=False,
    )

    if save_path:
        anim.save(save_path, writer="ffmpeg", fps=fps)
        print(f"Saved to {save_path}")

    plt.show()

    return anim