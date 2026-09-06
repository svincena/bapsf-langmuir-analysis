"""Modern Qt parameter editor and process monitor for Langmuir analysis."""

from __future__ import annotations

from pathlib import Path
import math
import sys

from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

from langmuir_analysis_config import (
    LAST_PARAMETERS_PATH,
    PARAMETER_SECTIONS,
    SUPPORTED_GEOMETRIES,
    default_parameters,
    load_last_parameters,
    save_last_parameters,
    validate_parameters,
)


GEOMETRY_TITLES = {
    "x_line": "X-line scan",
    "xy_plane": "XY-plane scan",
}

CHOICE_LABELS = {
    "manual": "Manual entry",
    "bmotion": "From HDF5 (bmotion)",
    "descending": "Positive Y to negative Y",
    "ascending": "Negative Y to positive Y",
    "high_bias_median": "High-bias regional median",
    "at_vp": "At plasma potential (ion-subtracted)",
    "individual": "Fit each shot separately",
    "average": "Average shots before fitting",
    "separate_profiles": "Separate ramp profiles",
    "ramp_time_map": "X vs ramp/time map",
}


SPLASH_DURATION_MS = 2_000
SPLASH_TITLE = "LAPD Langmuir Analysis Studio"
SPLASH_CANVAS_SIZE = QtCore.QSize(1200, 750)
ASSET_ROOT = Path(__file__).resolve().parent / "assets"
SPLASH_BACKGROUND_PATH = ASSET_ROOT / "splash" / "plasma_background.png"
SPLASH_LOGO_PATHS = {
    "bapsf": ASSET_ROOT / "branding" / "bapsf_logo.png",
    "ucla": ASSET_ROOT / "branding" / "ucla_logo.svg",
    "doe": ASSET_ROOT / "branding" / "doe_seal.svg",
}


def _splash_font(point_size, weight=QtGui.QFont.Normal, *, letter_spacing=0):
    font = QtGui.QFont("Avenir Next")
    font.setPointSizeF(float(point_size))
    font.setWeight(weight)
    if letter_spacing:
        font.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, letter_spacing)
    return font


def _aspect_fit_rect(source_size, bounds):
    """Center one image inside *bounds* without changing its proportions."""
    source_width = float(source_size.width())
    source_height = float(source_size.height())
    if source_width <= 0 or source_height <= 0:
        return QtCore.QRectF()
    scale = min(bounds.width() / source_width, bounds.height() / source_height)
    width = source_width * scale
    height = source_height * scale
    return QtCore.QRectF(
        bounds.center().x() - width / 2,
        bounds.center().y() - height / 2,
        width,
        height,
    )


def _draw_splash_asset(painter, path, bounds):
    """Draw a PNG or SVG branding asset at the highest available fidelity."""
    path = Path(path)
    if path.suffix.lower() == ".svg":
        renderer = QtSvg.QSvgRenderer(str(path))
        if not renderer.isValid():
            return False
        target = _aspect_fit_rect(renderer.defaultSize(), bounds)
        renderer.render(painter, target)
        return True

    pixmap = QtGui.QPixmap(str(path))
    if pixmap.isNull():
        return False
    target = _aspect_fit_rect(pixmap.size(), bounds)
    painter.drawPixmap(target, pixmap, QtCore.QRectF(pixmap.rect()))
    return True


def _draw_splash_title(painter):
    facility_font = _splash_font(
        9.5, QtGui.QFont.DemiBold, letter_spacing=1.8
    )
    facility_text = "BASIC PLASMA SCIENCE FACILITY  •  UCLA"
    facility_metrics = QtGui.QFontMetricsF(facility_font)
    facility_width = facility_metrics.horizontalAdvance(facility_text) + 30
    facility_rect = QtCore.QRectF(76, 42, facility_width, 30)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor(22, 72, 91, 208))
    painter.drawRoundedRect(facility_rect, 15, 15)
    painter.setFont(facility_font)
    painter.setPen(QtGui.QColor("#9CF4E2"))
    painter.drawText(facility_rect, QtCore.Qt.AlignCenter, facility_text)

    lapd_font = _splash_font(47, QtGui.QFont.Bold, letter_spacing=0.3)
    studio_font = _splash_font(34, QtGui.QFont.DemiBold, letter_spacing=0.15)
    baseline = 137
    title_x = 76
    lapd_gradient = QtGui.QLinearGradient(title_x, 85, title_x + 165, 142)
    lapd_gradient.setColorAt(0, QtGui.QColor("#88F7E1"))
    lapd_gradient.setColorAt(0.52, QtGui.QColor("#43CBE7"))
    lapd_gradient.setColorAt(1, QtGui.QColor("#51A7E8"))
    painter.setFont(lapd_font)
    painter.setPen(QtGui.QPen(QtGui.QBrush(lapd_gradient), 1))
    painter.drawText(QtCore.QPointF(title_x, baseline), "LAPD")

    lapd_width = QtGui.QFontMetricsF(lapd_font).horizontalAdvance("LAPD")
    painter.setFont(studio_font)
    painter.setPen(QtGui.QColor("#F7FBFF"))
    painter.drawText(
        QtCore.QPointF(title_x + lapd_width + 22, baseline - 2),
        "Langmuir Analysis Studio",
    )

    painter.setFont(_splash_font(10.5, QtGui.QFont.Medium, letter_spacing=1.2))
    painter.setPen(QtGui.QColor(161, 194, 217, 224))
    painter.drawText(
        QtCore.QRectF(79, 151, 800, 22),
        QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
        "PRECISION I–V DIAGNOSTICS FOR THE LARGE PLASMA DEVICE",
    )


def _draw_langmuir_trace(painter):
    card = QtCore.QRectF(76, 194, 1048, 342)
    shadow = card.translated(0, 8)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor(0, 0, 0, 72))
    painter.drawRoundedRect(shadow, 24, 24)

    card_gradient = QtGui.QLinearGradient(card.topLeft(), card.bottomRight())
    card_gradient.setColorAt(0, QtGui.QColor(7, 22, 38, 224))
    card_gradient.setColorAt(0.58, QtGui.QColor(7, 28, 46, 212))
    card_gradient.setColorAt(1, QtGui.QColor(7, 42, 54, 202))
    painter.setBrush(card_gradient)
    painter.setPen(QtGui.QPen(QtGui.QColor(86, 169, 201, 92), 1.1))
    painter.drawRoundedRect(card, 24, 24)

    painter.setFont(_splash_font(9.5, QtGui.QFont.DemiBold, letter_spacing=1.7))
    painter.setPen(QtGui.QColor(121, 228, 215, 228))
    painter.drawText(
        QtCore.QRectF(110, 211, 500, 26),
        QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
        "LANGMUIR I–V SIGNATURE",
    )
    painter.setFont(_splash_font(8.5, QtGui.QFont.Medium, letter_spacing=1.2))
    painter.setPen(QtGui.QColor(130, 165, 190, 190))
    painter.drawText(
        QtCore.QRectF(750, 211, 338, 26),
        QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
        "PROBE-BIAS SWEEP  →",
    )

    plot = QtCore.QRectF(116, 247, 968, 248)
    grid_pen = QtGui.QPen(QtGui.QColor(80, 145, 174, 36), 1)
    painter.setPen(grid_pen)
    for index in range(1, 9):
        x_pos = plot.left() + index * plot.width() / 9
        painter.drawLine(QtCore.QPointF(x_pos, plot.top()), QtCore.QPointF(x_pos, plot.bottom()))
    for index in range(1, 5):
        y_pos = plot.top() + index * plot.height() / 5
        painter.drawLine(QtCore.QPointF(plot.left(), y_pos), QtCore.QPointF(plot.right(), y_pos))

    minimum_current = -0.28
    maximum_current = 1.0

    def map_current(value):
        fraction = (maximum_current - value) / (maximum_current - minimum_current)
        return plot.top() + fraction * plot.height()

    zero_y = map_current(0)
    axis_pen = QtGui.QPen(QtGui.QColor(149, 191, 214, 112), 1.2)
    painter.setPen(axis_pen)
    painter.drawLine(
        QtCore.QPointF(plot.left(), zero_y),
        QtCore.QPointF(plot.right(), zero_y),
    )
    painter.drawLine(plot.bottomLeft(), plot.topLeft())

    painter.setFont(_splash_font(9, QtGui.QFont.DemiBold))
    painter.setPen(QtGui.QColor(165, 201, 220, 198))
    painter.drawText(QtCore.QPointF(plot.left() - 3, plot.top() - 8), "I")
    painter.drawText(QtCore.QPointF(plot.right() + 8, zero_y + 4), "V")

    def trace_current(t):
        saturation = 1.11 / (1 + math.exp(-13.5 * (t - 0.58)))
        sheath_slope = 0.055 * max(t - 0.73, 0)
        return -0.22 + saturation + sheath_slope

    trace = QtGui.QPainterPath()
    sample_count = 220
    trace_points = []
    for index in range(sample_count):
        t = index / (sample_count - 1)
        point = QtCore.QPointF(
            plot.left() + t * plot.width(), map_current(trace_current(t))
        )
        trace_points.append(point)
        if index == 0:
            trace.moveTo(point)
        else:
            trace.lineTo(point)

    painter.setBrush(QtCore.Qt.NoBrush)
    painter.setPen(
        QtGui.QPen(
            QtGui.QColor(38, 221, 234, 34),
            14,
            QtCore.Qt.SolidLine,
            QtCore.Qt.RoundCap,
            QtCore.Qt.RoundJoin,
        )
    )
    painter.drawPath(trace)
    painter.setPen(
        QtGui.QPen(
            QtGui.QColor(66, 231, 231, 100),
            7,
            QtCore.Qt.SolidLine,
            QtCore.Qt.RoundCap,
            QtCore.Qt.RoundJoin,
        )
    )
    painter.drawPath(trace)
    trace_gradient = QtGui.QLinearGradient(plot.left(), 0, plot.right(), 0)
    trace_gradient.setColorAt(0, QtGui.QColor("#43C9F1"))
    trace_gradient.setColorAt(0.55, QtGui.QColor("#69F0D2"))
    trace_gradient.setColorAt(1, QtGui.QColor("#F3C56B"))
    painter.setPen(
        QtGui.QPen(
            QtGui.QBrush(trace_gradient),
            3.2,
            QtCore.Qt.SolidLine,
            QtCore.Qt.RoundCap,
            QtCore.Qt.RoundJoin,
        )
    )
    painter.drawPath(trace)

    marker_specs = (
        (0.47, "Vf", QtGui.QColor("#78EED7")),
        (0.60, "Vp", QtGui.QColor("#F1C56D")),
    )
    for t, label, color in marker_specs:
        point = trace_points[round(t * (sample_count - 1))]
        marker_pen = QtGui.QPen(QtGui.QColor(color.red(), color.green(), color.blue(), 115), 1)
        marker_pen.setStyle(QtCore.Qt.DashLine)
        painter.setPen(marker_pen)
        painter.drawLine(
            QtCore.QPointF(point.x(), plot.top() + 12),
            QtCore.QPointF(point.x(), plot.bottom()),
        )
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(point, 4.2, 4.2)
        painter.setFont(_splash_font(9.5, QtGui.QFont.DemiBold))
        painter.setPen(color)
        painter.drawText(
            QtCore.QRectF(point.x() - 22, plot.top() + 5, 44, 20),
            QtCore.Qt.AlignCenter,
            label,
        )


def _draw_logo_rail(painter):
    rail = QtCore.QRectF(76, 561, 1048, 155)
    shadow = rail.translated(0, 7)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor(0, 0, 0, 68))
    painter.drawRoundedRect(shadow, 22, 22)

    rail_gradient = QtGui.QLinearGradient(rail.topLeft(), rail.bottomRight())
    rail_gradient.setColorAt(0, QtGui.QColor(249, 252, 253, 244))
    rail_gradient.setColorAt(0.55, QtGui.QColor(242, 248, 250, 242))
    rail_gradient.setColorAt(1, QtGui.QColor(233, 243, 246, 239))
    painter.setBrush(rail_gradient)
    painter.setPen(QtGui.QPen(QtGui.QColor(154, 211, 221, 120), 1.1))
    painter.drawRoundedRect(rail, 22, 22)

    for x_pos in (429, 794):
        painter.setPen(QtGui.QPen(QtGui.QColor(47, 88, 108, 42), 1))
        painter.drawLine(QtCore.QPointF(x_pos, 586), QtCore.QPointF(x_pos, 690))

    caption_font = _splash_font(7.8, QtGui.QFont.DemiBold, letter_spacing=1.25)
    painter.setFont(caption_font)
    painter.setPen(QtGui.QColor("#557184"))
    captions = (
        (QtCore.QRectF(101, 577, 304, 18), "FACILITY"),
        (QtCore.QRectF(454, 577, 315, 18), "UNIVERSITY"),
        (QtCore.QRectF(819, 577, 280, 18), "RESEARCH SUPPORT"),
    )
    for bounds, caption in captions:
        painter.drawText(bounds, QtCore.Qt.AlignCenter, caption)

    assets = (
        (SPLASH_LOGO_PATHS["bapsf"], QtCore.QRectF(114, 600, 278, 92)),
        (SPLASH_LOGO_PATHS["ucla"], QtCore.QRectF(478, 604, 267, 87)),
        (SPLASH_LOGO_PATHS["doe"], QtCore.QRectF(899, 597, 116, 104)),
    )
    for path, bounds in assets:
        if not _draw_splash_asset(painter, path, bounds):
            painter.setFont(_splash_font(10, QtGui.QFont.DemiBold))
            painter.setPen(QtGui.QColor("#8B2030"))
            painter.drawText(bounds, QtCore.Qt.AlignCenter, path.stem.upper())


def compose_splash_pixmap():
    """Compose the splash from exact branding assets and vector overlays."""
    canvas = QtGui.QPixmap(SPLASH_CANVAS_SIZE)
    canvas.fill(QtGui.QColor("#07111F"))
    painter = QtGui.QPainter(canvas)
    painter.setRenderHints(
        QtGui.QPainter.Antialiasing
        | QtGui.QPainter.TextAntialiasing
        | QtGui.QPainter.SmoothPixmapTransform
    )

    background = QtGui.QPixmap(str(SPLASH_BACKGROUND_PATH))
    if not background.isNull():
        painter.drawPixmap(
            QtCore.QRectF(0, 0, canvas.width(), canvas.height()),
            background,
            QtCore.QRectF(background.rect()),
        )

    veil = QtGui.QLinearGradient(0, 0, 0, canvas.height())
    veil.setColorAt(0, QtGui.QColor(2, 9, 18, 78))
    veil.setColorAt(0.24, QtGui.QColor(4, 14, 27, 35))
    veil.setColorAt(0.72, QtGui.QColor(3, 13, 23, 20))
    veil.setColorAt(1, QtGui.QColor(2, 9, 17, 92))
    painter.fillRect(canvas.rect(), veil)

    _draw_splash_title(painter)
    _draw_langmuir_trace(painter)
    _draw_logo_rail(painter)
    painter.end()
    return canvas


class StartupSplashScreen(QtWidgets.QWidget):
    """Frameless startup window with no implicit Qt splash dismissal."""

    def __init__(self, pixmap, flags, parent=None):
        super().__init__(parent, flags)
        self._pixmap = pixmap
        device_ratio = max(1.0, pixmap.devicePixelRatio())
        self.setFixedSize(
            round(pixmap.width() / device_ratio),
            round(pixmap.height() / device_ratio),
        )
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.drawPixmap(
            QtCore.QRectF(self.rect()),
            self._pixmap,
            QtCore.QRectF(self._pixmap.rect()),
        )
        event.accept()

    def finish(self, _main_window):
        self.hide()

    def mousePressEvent(self, event):
        # QSplashScreen normally hides itself on any click.  When the app is
        # started from an IDE or dock, that launch click can arrive just after
        # this window is mapped and make the splash appear to flash briefly.
        event.accept()

    def mouseDoubleClickEvent(self, event):
        event.accept()


def create_splash_screen(application):
    """Return a centered splash sized safely for the active display."""
    pixmap = compose_splash_pixmap()
    screen = application.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry().size()
        maximum = QtCore.QSize(
            max(640, round(available.width() * 0.88)),
            max(400, round(available.height() * 0.88)),
        )
        if pixmap.width() > maximum.width() or pixmap.height() > maximum.height():
            pixmap = pixmap.scaled(
                maximum,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )

    flags = (
        QtCore.Qt.Window
        | QtCore.Qt.FramelessWindowHint
        | QtCore.Qt.WindowStaysOnTopHint
    )
    splash = StartupSplashScreen(pixmap, flags)
    if screen is not None:
        available_rect = screen.availableGeometry()
        splash.move(available_rect.center() - splash.rect().center())
    splash.setObjectName("startupSplash")
    splash.setAccessibleName(SPLASH_TITLE)
    splash.setProperty("titleText", SPLASH_TITLE)
    splash.setProperty("titleBounds", QtCore.QRect(76, 42, 1048, 132))
    splash.setProperty("traceBounds", QtCore.QRect(116, 247, 968, 248))
    splash.setProperty("logoRailBounds", QtCore.QRect(76, 561, 1048, 155))
    splash.setProperty(
        "logoPaths", tuple(str(path) for path in SPLASH_LOGO_PATHS.values())
    )
    return splash


STYLE_SHEET = """
QWidget {
    color: #dce7f4;
    font-family: "Avenir Next";
    font-size: 13px;
}
QMainWindow, QWidget#root { background: #0b111b; }
QFrame#hero {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #132239, stop:0.56 #14273a, stop:1 #12342f);
    border: 1px solid #29425b;
    border-radius: 18px;
}
QLabel#eyebrow {
    color: #72dec6;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}
QLabel#title { color: #f7fbff; font-size: 28px; font-weight: 700; }
QLabel#subtitle { color: #a7bacd; font-size: 13px; }
QLabel#statusBadge {
    color: #81e6cf;
    background: #153b38;
    border: 1px solid #297164;
    border-radius: 12px;
    padding: 5px 10px;
    font-weight: 600;
}
QTabWidget::pane { border: 0; top: -1px; }
QTabBar::tab {
    color: #8fa5ba;
    background: #111a27;
    border: 1px solid #26364a;
    border-bottom: 0;
    padding: 12px 25px;
    margin-right: 5px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    font-weight: 650;
}
QTabBar::tab:selected { color: #f1f8ff; background: #182536; }
QTabBar::tab:hover:!selected { color: #d1deeb; background: #162130; }
QScrollArea { border: 0; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QFrame#sectionCard {
    background: #121c2a;
    border: 1px solid #25374c;
    border-radius: 14px;
}
QLabel#sectionTitle { color: #f3f8fd; font-size: 16px; font-weight: 700; }
QLabel#sectionDescription { color: #8399ae; font-size: 11px; }
QLabel#fieldLabel { color: #b6c6d6; font-weight: 550; }
QLineEdit, QSpinBox, QComboBox {
    color: #edf5fc;
    background: #0c1521;
    border: 1px solid #30445b;
    border-radius: 8px;
    min-height: 32px;
    padding: 0 9px;
    selection-background-color: #2b7a78;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #55cdb7;
}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {
    color: #617488; background: #111923;
}
QComboBox::drop-down { border: 0; width: 28px; }
QComboBox QAbstractItemView {
    color: #e9f2fa; background: #111c29; border: 1px solid #34495f;
    selection-background-color: #23645c;
}
QCheckBox { color: #b9c9d8; spacing: 9px; }
QCheckBox::indicator {
    width: 34px; height: 19px; border-radius: 10px;
    background: #28384a; border: 1px solid #3a4e63;
}
QCheckBox::indicator:checked {
    background: #34a995; border: 1px solid #64d3bf;
}
QPushButton {
    color: #dce8f3; background: #1b2a3b; border: 1px solid #344a61;
    border-radius: 9px; min-height: 34px; padding: 0 14px; font-weight: 600;
}
QPushButton:hover { background: #24384c; border-color: #4a657e; }
QPushButton:pressed { background: #162333; }
QPushButton:disabled { color: #607386; background: #17212d; border-color: #273646; }
QPushButton#browseButton { min-width: 36px; max-width: 36px; padding: 0; }
QPushButton#startButton {
    color: #071411;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #60dfc4, stop:1 #80e6b4);
    border: 1px solid #a0f2d2;
    border-radius: 12px;
    min-height: 52px;
    padding: 0 28px;
    font-size: 15px;
    font-weight: 750;
}
QPushButton#startButton:hover { background: #8af0d4; }
QPushButton#startButton:disabled { color: #728b84; background: #29413d; border-color: #35544e; }
QFrame#consoleCard { background: #0e1722; border: 1px solid #26384b; border-radius: 13px; }
QPlainTextEdit {
    color: #bad0df; background: #09111a; border: 0; border-radius: 8px;
    font-family: "SFMono-Regular", "Menlo", monospace; font-size: 11px;
    padding: 8px;
}
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #34485c; min-height: 30px; border-radius: 5px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { color: #eaf3fa; background: #152333; border: 1px solid #3a526a; padding: 5px; }
"""


def discover_sis_configurations(filename, digitizer):
    """Return immediate group names under one HDF5 digitizer group."""
    filename = Path(filename).expanduser()
    digitizer = str(digitizer).strip()
    if not filename.is_file():
        raise ValueError(f"Experiment HDF5 file does not exist: {filename}")
    if not digitizer:
        raise ValueError("Digitizer group cannot be empty.")

    # Keep h5py out of the GUI's lightweight splash-startup import path.
    import h5py

    group_names = ("Raw data + config", digitizer)
    try:
        with h5py.File(filename, "r") as h5_file:
            group = h5_file
            traversed = []
            for group_name in group_names:
                traversed.append(group_name)
                if group_name not in group:
                    hdf5_path = "/" + "/".join(traversed)
                    raise ValueError(f'HDF5 group "{hdf5_path}" was not found.')
                group = group[group_name]
                if not isinstance(group, h5py.Group):
                    hdf5_path = "/" + "/".join(traversed)
                    raise ValueError(f'HDF5 object "{hdf5_path}" is not a group.')

            return tuple(
                sorted(
                    (
                        name
                        for name, item in group.items()
                        if isinstance(item, h5py.Group)
                    ),
                    key=str.casefold,
                )
            )
    except OSError as error:
        raise ValueError(f"Could not open HDF5 file {filename}: {error}") from error


def discover_bmotion_configurations(filename):
    """Return bmotion configuration names mapped by bapsflib."""
    filename = Path(filename).expanduser()
    if not filename.is_file():
        raise ValueError(f"Experiment HDF5 file does not exist: {filename}")

    # Keep bapsflib and its scientific dependencies out of splash startup.
    from bapsflib import lapd

    file_obj = None
    try:
        file_obj = lapd.File(filename)
        if "bmotion" not in file_obj.controls:
            raise ValueError(f"HDF5 file {filename} has no mapped bmotion control.")
        return tuple(
            sorted(
                (str(name) for name in file_obj.controls["bmotion"].configs),
                key=str.casefold,
            )
        )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            f"Could not inspect bmotion configurations in {filename}: {error}"
        ) from error
    finally:
        if file_obj is not None:
            file_obj.close()


def infer_bmotion_geometry(shot_numbers, target_positions_cm, geometry):
    """Infer one supported regular scan geometry from bmotion target data."""
    import numpy as np

    if geometry not in {"x_line", "xy_plane"}:
        raise ValueError(f"Unsupported analysis geometry {geometry!r}.")

    shot_numbers = np.asarray(shot_numbers)
    target_positions_cm = np.asarray(target_positions_cm, dtype=float)
    if shot_numbers.ndim != 1:
        raise ValueError("bmotion shot numbers must be one-dimensional.")
    if (
        target_positions_cm.ndim != 2
        or target_positions_cm.shape[0] != shot_numbers.size
        or target_positions_cm.shape[1] < 2
    ):
        raise ValueError(
            "bmotion target positions must have shape (number of shots, 2 or more)."
        )
    if shot_numbers.size < 2:
        raise ValueError("bmotion must contain at least two scan shots.")
    if np.any(~np.isfinite(shot_numbers)) or np.any(
        shot_numbers != shot_numbers.astype(np.int64)
    ):
        raise ValueError("bmotion shot numbers must be finite integers.")
    shot_numbers = shot_numbers.astype(np.int64)
    if np.any(shot_numbers <= 0) or np.any(np.diff(shot_numbers) != 1):
        raise ValueError(
            "bmotion shots for the selected configuration must be positive and contiguous."
        )

    tolerance_cm = 1e-8
    x_values = target_positions_cm[:, 0]
    y_values = target_positions_cm[:, 1]
    if not np.all(np.isfinite(x_values)):
        raise ValueError("bmotion X target positions must be finite.")
    if geometry == "x_line":
        finite_y = y_values[np.isfinite(y_values)]
        if finite_y.size and np.ptp(finite_y) > tolerance_cm:
            raise ValueError(
                "The selected bmotion configuration varies in Y and is not an X-line scan."
            )
        run_coordinates = x_values[:, np.newaxis]
    else:
        if not np.all(np.isfinite(y_values)):
            raise ValueError("bmotion Y target positions must be finite for an XY scan.")
        run_coordinates = target_positions_cm[:, :2]

    # Every spatial position must occupy one contiguous run of repeated shots;
    # this is the acquisition ordering expected by the analysis reshapes.
    same_as_previous = np.all(
        np.isclose(
            run_coordinates[1:],
            run_coordinates[:-1],
            rtol=0.0,
            atol=tolerance_cm,
        ),
        axis=1,
    )
    run_starts = np.concatenate(([0], np.flatnonzero(~same_as_previous) + 1))
    run_stops = np.concatenate((run_starts[1:], [shot_numbers.size]))
    run_lengths = run_stops - run_starts
    if np.any(run_lengths != run_lengths[0]):
        raise ValueError(
            "bmotion does not contain a constant number of repeated shots per position."
        )
    ordered_positions = run_coordinates[run_starts]
    rounded_positions = np.round(ordered_positions, decimals=8)
    if np.unique(rounded_positions, axis=0).shape[0] != ordered_positions.shape[0]:
        raise ValueError(
            "bmotion revisits a target position in separate shot blocks; this ordering is unsupported."
        )

    result = {
        "nshots": int(run_lengths[0]),
        "data_offset": int(shot_numbers[0] - 1),
    }
    if geometry == "x_line":
        ordered_x = ordered_positions[:, 0]
        x_steps = np.diff(ordered_x)
        if ordered_x.size < 2 or np.any(x_steps <= tolerance_cm):
            raise ValueError(
                "X-line bmotion targets must be ordered at strictly increasing X positions."
            )
        if not np.allclose(
            x_steps,
            x_steps[0],
            rtol=0.0,
            atol=tolerance_cm,
        ):
            raise ValueError("X-line bmotion targets must be evenly spaced.")
        result.update(
            nx=int(ordered_x.size),
            x_min_cm=float(ordered_x[0]),
            x_max_cm=float(ordered_x[-1]),
        )
        return result

    unique_x = np.unique(rounded_positions[:, 0])
    unique_y = np.unique(rounded_positions[:, 1])
    if unique_x.size < 2 or unique_y.size < 2:
        raise ValueError(
            "XY bmotion targets must contain at least two distinct X and Y positions."
        )
    if not np.allclose(
        np.diff(unique_x),
        np.diff(unique_x)[0],
        rtol=0.0,
        atol=tolerance_cm,
    ) or not np.allclose(
        np.diff(unique_y),
        np.diff(unique_y)[0],
        rtol=0.0,
        atol=tolerance_cm,
    ):
        raise ValueError("XY bmotion targets must be evenly spaced along each axis.")
    descending_positions = np.asarray(
        [(x_value, y_value) for y_value in unique_y[::-1] for x_value in unique_x]
    )
    ascending_positions = np.asarray(
        [(x_value, y_value) for y_value in unique_y for x_value in unique_x]
    )
    if ordered_positions.shape == descending_positions.shape and np.allclose(
        ordered_positions,
        descending_positions,
        rtol=0.0,
        atol=tolerance_cm,
    ):
        y_acquisition_order = "descending"
    elif ordered_positions.shape == ascending_positions.shape and np.allclose(
        ordered_positions,
        ascending_positions,
        rtol=0.0,
        atol=tolerance_cm,
    ):
        y_acquisition_order = "ascending"
    else:
        raise ValueError(
            "XY bmotion targets must form a complete rectangular grid ordered "
            "with X increasing within each row and monotonic Y between rows."
        )
    result.update(
        nx=int(unique_x.size),
        x_min_cm=float(unique_x[0]),
        x_max_cm=float(unique_x[-1]),
        ny=int(unique_y.size),
        y_min_cm=float(unique_y[0]),
        y_max_cm=float(unique_y[-1]),
        xy_y_acquisition_order=y_acquisition_order,
    )
    return result


def read_bmotion_geometry(filename, config_name, geometry):
    """Read target positions through bapsflib and infer geometry in cm."""
    filename = Path(filename).expanduser()
    config_name = str(config_name).strip()
    if not filename.is_file():
        raise ValueError(f"Experiment HDF5 file does not exist: {filename}")
    if not config_name:
        raise ValueError("bmotion configuration cannot be empty.")

    import astropy.units as u
    import numpy as np
    from bapsflib import lapd

    file_obj = None
    try:
        file_obj = lapd.File(filename)
        if "bmotion" not in file_obj.controls:
            raise ValueError(f"HDF5 file {filename} has no mapped bmotion control.")
        control_map = file_obj.controls["bmotion"]
        if config_name not in control_map.configs:
            raise ValueError(
                f'bmotion configuration "{config_name}" is not present in {filename}.'
            )
        control_data = file_obj.read_controls(
            [("bmotion", config_name)],
            silent=True,
        )
        if "xyz_target" not in (control_data.dtype.names or ()):
            raise ValueError(
                f'bmotion configuration "{config_name}" has no xyz_target data.'
            )
        shot_numbers = np.asarray(control_data["shotnum"]).copy()
        target_positions = np.asarray(
            control_data["xyz_target"], dtype=float
        ).copy()

        config = control_map.configs[config_name]
        motion_config = config["meta"][0]["MG_CONFIG"]
        axes = motion_config["drive"]["axes"].values()
        axis_units = {
            str(axis["name"]).casefold(): str(axis["units"]) for axis in axes
        }
        required_axes = ("x",) if geometry == "x_line" else ("x", "y")
        for axis_index, axis_name in enumerate(required_axes):
            if axis_name not in axis_units:
                raise ValueError(
                    f'bmotion configuration "{config_name}" has no {axis_name.upper()} axis.'
                )
            try:
                scale_to_cm = (1.0 * u.Unit(axis_units[axis_name])).to_value(u.cm)
            except (ValueError, u.UnitConversionError) as error:
                raise ValueError(
                    f'bmotion {axis_name.upper()} units '
                    f'"{axis_units[axis_name]}" are not a length.'
                ) from error
            target_positions[:, axis_index] *= scale_to_cm
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            f"Could not read bmotion geometry from {filename}: {error}"
        ) from error
    finally:
        if file_obj is not None:
            file_obj.close()

    return infer_bmotion_geometry(shot_numbers, target_positions, geometry)


def discover_sis_sample_count(filename, digitizer, config_name):
    """Return the common trace length for a digitizer configuration."""
    filename = Path(filename).expanduser()
    digitizer = str(digitizer).strip()
    config_name = str(config_name).strip()
    if not filename.is_file():
        raise ValueError(f"Experiment HDF5 file does not exist: {filename}")
    if not digitizer:
        raise ValueError("Digitizer group cannot be empty.")
    if not config_name:
        raise ValueError("SIS configuration cannot be empty.")

    # SIS channel datasets are siblings of the configuration group and have
    # names such as "<configuration> [Slot 2: SIS 3302 ch 3]".
    import h5py

    digitizer_path = f"/Raw data + config/{digitizer}"
    try:
        with h5py.File(filename, "r") as h5_file:
            if digitizer_path not in h5_file:
                raise ValueError(f'HDF5 group "{digitizer_path}" was not found.')
            group = h5_file[digitizer_path]
            if not isinstance(group, h5py.Group):
                raise ValueError(f'HDF5 object "{digitizer_path}" is not a group.')
            if config_name not in group or not isinstance(
                group[config_name], h5py.Group
            ):
                raise ValueError(
                    f'SIS configuration "{config_name}" was not found in '
                    f'"{digitizer_path}".'
                )

            dataset_prefix = f"{config_name} ["
            sample_counts = {
                int(item.shape[-1])
                for name, item in group.items()
                if name.startswith(dataset_prefix)
                and isinstance(item, h5py.Dataset)
                and item.ndim >= 2
            }
            if not sample_counts:
                raise ValueError(
                    f'No SIS channel datasets were found for configuration '
                    f'"{config_name}".'
                )
            if len(sample_counts) != 1:
                formatted_counts = ", ".join(
                    str(count) for count in sorted(sample_counts)
                )
                raise ValueError(
                    f'SIS channel datasets for configuration "{config_name}" '
                    f"have inconsistent trace lengths: {formatted_counts}."
                )
            return sample_counts.pop()
    except OSError as error:
        raise ValueError(f"Could not open HDF5 file {filename}: {error}") from error


class PathEditor(QtWidgets.QWidget):
    """Line editor with a native file or directory picker."""

    path_selected = QtCore.Signal(str)

    def __init__(self, value, *, directory=False, parent=None):
        super().__init__(parent)
        self.directory = directory
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.line_edit = QtWidgets.QLineEdit(str(value))
        self.line_edit.setToolTip(str(value))
        self.line_edit.textChanged.connect(self.line_edit.setToolTip)
        self.browse_button = QtWidgets.QPushButton("…")
        self.browse_button.setObjectName("browseButton")
        self.browse_button.setToolTip(
            "Choose a folder" if directory else "Choose an HDF5 experiment file"
        )
        self.browse_button.clicked.connect(self._browse)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.browse_button)

    def _browse(self):
        current = Path(self.line_edit.text()).expanduser()
        start = str(current if current.is_dir() else current.parent)
        if self.directory:
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Choose output folder", start
            )
        else:
            selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Choose LAPD experiment file",
                start,
                "HDF5 files (*.hdf5 *.h5);;All files (*)",
            )
        if selected:
            self.line_edit.setText(selected)
            self.path_selected.emit(selected)

    def value(self):
        return self.line_edit.text().strip()

    def set_value(self, value):
        self.line_edit.setText(str(value))


class SisConfigurationEditor(QtWidgets.QWidget):
    """Editable SIS configuration name with HDF5-backed candidate selection."""

    def __init__(self, value, source_values, parent=None):
        super().__init__(parent)
        self.source_values = source_values
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.line_edit = QtWidgets.QLineEdit(str(value))
        self.line_edit.setToolTip(str(value))
        self.line_edit.textChanged.connect(self.line_edit.setToolTip)
        self.browse_button = QtWidgets.QPushButton("…")
        self.browse_button.setObjectName("browseButton")
        self.browse_button.setToolTip(
            "Choose a configuration found in the selected HDF5 file"
        )
        self.browse_button.clicked.connect(self._browse)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.browse_button)

    def _browse(self):
        filename, digitizer = self.source_values()
        try:
            candidates = discover_sis_configurations(filename, digitizer)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(
                self,
                "Could not find SIS configurations",
                str(error),
            )
            return

        if not candidates:
            QtWidgets.QMessageBox.information(
                self,
                "No SIS configurations found",
                "The selected digitizer group contains no configuration groups.",
            )
            return

        current_value = self.value()
        current_index = (
            candidates.index(current_value) if current_value in candidates else 0
        )
        selected, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Choose SIS configuration",
            "Available configurations:",
            candidates,
            current_index,
            False,
        )
        if accepted:
            self.set_value(selected)

    def value(self):
        return self.line_edit.text().strip()

    def set_value(self, value):
        self.line_edit.setText(str(value))


class BMotionConfigurationEditor(QtWidgets.QWidget):
    """Editable bmotion configuration with an HDF5-backed picker."""

    configuration_selected = QtCore.Signal(str)

    def __init__(self, value, source_filename, parent=None):
        super().__init__(parent)
        self.source_filename = source_filename
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.line_edit = QtWidgets.QLineEdit(str(value))
        self.line_edit.setToolTip(str(value))
        self.line_edit.textChanged.connect(self.line_edit.setToolTip)
        self.browse_button = QtWidgets.QPushButton("…")
        self.browse_button.setObjectName("browseButton")
        self.browse_button.setToolTip(
            "Choose a bmotion configuration found in the selected HDF5 file"
        )
        self.browse_button.clicked.connect(self._browse)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.browse_button)

    def _browse(self):
        try:
            candidates = discover_bmotion_configurations(self.source_filename())
        except ValueError as error:
            QtWidgets.QMessageBox.warning(
                self,
                "Could not find bmotion configurations",
                str(error),
            )
            return

        if not candidates:
            QtWidgets.QMessageBox.information(
                self,
                "No bmotion configurations found",
                "The selected HDF5 file contains no mapped bmotion configurations.",
            )
            return

        current_value = self.value()
        current_index = (
            candidates.index(current_value) if current_value in candidates else 0
        )
        selected, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Choose bmotion configuration",
            "Available motion configurations:",
            candidates,
            current_index,
            False,
        )
        if accepted:
            self.set_value(selected)
            self.configuration_selected.emit(selected)

    def value(self):
        return self.line_edit.text().strip()

    def set_value(self, value):
        self.line_edit.setText(str(value))


class DirectNumericInput(QtWidgets.QLineEdit):
    """Validated numeric text input with no scroll-wheel adjustment behavior."""

    def __init__(
        self,
        value,
        *,
        numeric_kind,
        minimum,
        maximum,
        decimals=3,
        parent=None,
    ):
        super().__init__(parent)
        if numeric_kind not in {"int", "float"}:
            raise ValueError(f"Unsupported numeric input kind {numeric_kind!r}.")

        self.numeric_kind = numeric_kind
        self.minimum = minimum
        self.maximum = maximum
        self.decimals = int(decimals)
        locale = QtCore.QLocale.c()
        locale.setNumberOptions(QtCore.QLocale.RejectGroupSeparator)

        if numeric_kind == "int":
            validator = QtGui.QIntValidator(int(minimum), int(maximum), self)
        else:
            validator = QtGui.QDoubleValidator(
                float(minimum),
                float(maximum),
                self.decimals,
                self,
            )
            validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        validator.setLocale(locale)
        self.setValidator(validator)
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.set_value(value)

    def value(self):
        text = self.text().strip()
        state, _text, _position = self.validator().validate(text, 0)
        if state != QtGui.QValidator.Acceptable:
            kind = "an integer" if self.numeric_kind == "int" else "a number"
            raise ValueError(
                f"Value must be {kind} from {self.minimum} to {self.maximum}."
            )
        return int(text) if self.numeric_kind == "int" else float(text)

    def set_value(self, value):
        if self.numeric_kind == "int":
            text = str(int(value))
        else:
            text = format(float(value), ".15g")
        self.setText(text)


class IntegerPairEditor(QtWidgets.QWidget):
    """Compact y/x integer-pair editor for two-dimensional neighborhoods."""

    def __init__(self, value, minimum, maximum, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.y_value = DirectNumericInput(
            value[0],
            numeric_kind="int",
            minimum=minimum,
            maximum=maximum,
        )
        self.x_value = DirectNumericInput(
            value[1],
            numeric_kind="int",
            minimum=minimum,
            maximum=maximum,
        )
        for label, editor in (("Y", self.y_value), ("X", self.x_value)):
            axis_label = QtWidgets.QLabel(label)
            axis_label.setObjectName("fieldLabel")
            layout.addWidget(axis_label)
            layout.addWidget(editor, 1)

    def value(self):
        return self.y_value.value(), self.x_value.value()

    def set_value(self, value):
        self.y_value.set_value(value[0])
        self.x_value.set_value(value[1])


def _make_editor(
    spec,
    *,
    sis_source_values=None,
    bmotion_source_filename=None,
):
    if spec.key == "sis_config_name":
        editor = SisConfigurationEditor(spec.default, sis_source_values)
    elif spec.key == "bmotion_config_name":
        editor = BMotionConfigurationEditor(
            spec.default,
            bmotion_source_filename,
        )
    elif spec.kind == "bool":
        editor = QtWidgets.QCheckBox("Enabled")
        editor.setChecked(bool(spec.default))
    elif spec.kind == "int":
        editor = DirectNumericInput(
            spec.default,
            numeric_kind="int",
            minimum=spec.minimum,
            maximum=spec.maximum,
        )
    elif spec.kind == "optional_int":
        editor = QtWidgets.QSpinBox()
        editor.setRange(int(spec.minimum), int(spec.maximum))
        editor.setSpecialValueText("Automatic")
        editor.setValue(0 if spec.default is None else int(spec.default))
    elif spec.kind == "float":
        editor = DirectNumericInput(
            spec.default,
            numeric_kind="float",
            minimum=spec.minimum,
            maximum=spec.maximum,
            decimals=spec.decimals,
        )
    elif spec.kind == "choice":
        editor = QtWidgets.QComboBox()
        for choice in spec.choices:
            editor.addItem(CHOICE_LABELS.get(choice, choice), choice)
        editor.setCurrentIndex(editor.findData(str(spec.default)))
    elif spec.kind in {"file", "directory"}:
        editor = PathEditor(spec.default, directory=spec.kind == "directory")
    elif spec.kind == "int_pair":
        editor = IntegerPairEditor(spec.default, spec.minimum, spec.maximum)
    else:
        editor = QtWidgets.QLineEdit(str(spec.default))

    editor.setToolTip(spec.description)
    return editor


def _editor_value(editor, spec):
    if isinstance(
        editor,
        (
            PathEditor,
            SisConfigurationEditor,
            BMotionConfigurationEditor,
            IntegerPairEditor,
        ),
    ):
        return editor.value()
    if spec.kind == "bool":
        return editor.isChecked()
    if spec.kind == "optional_int":
        return None if editor.value() == 0 else editor.value()
    if spec.kind in {"int", "float"}:
        return editor.value()
    if spec.kind == "choice":
        return editor.currentData()
    return editor.text().strip()


def _set_editor_value(editor, spec, value):
    if isinstance(
        editor,
        (
            PathEditor,
            SisConfigurationEditor,
            BMotionConfigurationEditor,
            IntegerPairEditor,
            DirectNumericInput,
        ),
    ):
        editor.set_value(value)
    elif spec.kind == "bool":
        editor.setChecked(bool(value))
    elif spec.kind == "optional_int":
        editor.setValue(0 if value is None else int(value))
    elif spec.kind == "choice":
        editor.setCurrentIndex(editor.findData(str(value)))
    else:
        editor.setText(str(value))


class ParameterTab(QtWidgets.QWidget):
    start_requested = QtCore.Signal(str)

    def __init__(self, geometry, parent=None):
        super().__init__(parent)
        self.geometry = geometry
        self.editors = {}
        self.specs = {}
        self.section_cards = {}
        self._setting_values = False

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 12, 0, 0)
        root_layout.setSpacing(12)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_body = QtWidgets.QWidget()
        card_columns_layout = QtWidgets.QHBoxLayout(scroll_body)
        card_columns_layout.setContentsMargins(0, 0, 4, 6)
        card_columns_layout.setSpacing(12)
        card_columns = [QtWidgets.QVBoxLayout(), QtWidgets.QVBoxLayout()]
        for column in card_columns:
            column.setSpacing(12)
            card_columns_layout.addLayout(column, 1)

        # Alternate ordinary cards between columns while keeping an explicitly
        # paired card directly below its predecessor in the same column.
        next_column = 0
        previous_column = None
        for section in PARAMETER_SECTIONS[geometry]:
            if section.stack_with_previous and previous_column is not None:
                column_index = previous_column
            else:
                column_index = next_column
                next_column = 1 - next_column
            card = self._build_section_card(section)
            card.setProperty("layoutColumn", column_index)
            self.section_cards[section.title] = card
            card_columns[column_index].addWidget(card)
            previous_column = column_index
        for column in card_columns:
            column.addStretch(1)
        self._wire_dependencies()
        scroll.setWidget(scroll_body)
        root_layout.addWidget(scroll, 1)

        actions = QtWidgets.QHBoxLayout()
        actions.setContentsMargins(4, 0, 4, 0)
        reset_button = QtWidgets.QPushButton("Restore this tab’s defaults")
        reset_button.clicked.connect(self.restore_defaults)
        self.start_button = QtWidgets.QPushButton(
            f"Start Analysis  ·  {GEOMETRY_TITLES[geometry]}  →"
        )
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(
            lambda: self.start_requested.emit(self.geometry)
        )
        actions.addWidget(reset_button)
        actions.addStretch(1)
        actions.addWidget(self.start_button)
        root_layout.addLayout(actions)

    def _build_section_card(self, section):
        card = QtWidgets.QFrame()
        card.setObjectName("sectionCard")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(16, 15, 16, 17)
        card_layout.setSpacing(5)

        title = QtWidgets.QLabel(section.title)
        title.setObjectName("sectionTitle")
        description = QtWidgets.QLabel(section.description)
        description.setObjectName("sectionDescription")
        description.setWordWrap(True)
        card_layout.addWidget(title)
        card_layout.addWidget(description)
        card_layout.addSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        for spec in section.parameters:
            label_text = spec.label + (f"  [{spec.unit}]" if spec.unit else "")
            label = QtWidgets.QLabel(label_text)
            label.setObjectName("fieldLabel")
            label.setToolTip(spec.description)
            editor = _make_editor(
                spec,
                sis_source_values=(
                    self._sis_source_values
                    if spec.key == "sis_config_name"
                    else None
                ),
                bmotion_source_filename=(
                    self._source_filename
                    if spec.key == "bmotion_config_name"
                    else None
                ),
            )
            self.editors[spec.key] = editor
            self.specs[spec.key] = spec
            form.addRow(label, editor)
        card_layout.addLayout(form)
        return card

    def _sis_source_values(self):
        """Read the current file and digitizer values from this tab."""
        return tuple(
            _editor_value(self.editors[key], self.specs[key])
            for key in ("filename", "digitizer")
        )

    def _source_filename(self):
        """Return the experiment filename currently displayed by this tab."""
        return _editor_value(self.editors["filename"], self.specs["filename"])

    def reconcile_bmotion_geometry(self):
        """Replace file-backed geometry and shot grouping from bmotion."""
        source = _editor_value(
            self.editors["spatial_geometry_source"],
            self.specs["spatial_geometry_source"],
        )
        if source == "manual":
            return ()

        filename = self._source_filename()
        config_editor = self.editors["bmotion_config_name"]
        config_name = config_editor.value()
        candidates = discover_bmotion_configurations(filename)
        if not candidates:
            raise ValueError(
                f"No mapped bmotion configurations were found in "
                f"{Path(filename).expanduser()}."
            )

        adjustments = []
        if config_name not in candidates:
            if len(candidates) != 1:
                valid_names = ", ".join(repr(name) for name in candidates)
                raise ValueError(
                    f'bmotion configuration "{config_name}" is not present in '
                    f"the selected HDF5 file. Choose one of: {valid_names}."
                )
            selected_name = candidates[0]
            config_editor.set_value(selected_name)
            adjustments.append(
                f'bmotion configuration: "{config_name}" → "{selected_name}"'
            )
            config_name = selected_name

        geometry_values = read_bmotion_geometry(
            filename,
            config_name,
            self.geometry,
        )
        geometry_keys = ["nx", "x_min_cm", "x_max_cm"]
        if self.geometry == "xy_plane":
            geometry_keys.extend(
                ("ny", "y_min_cm", "y_max_cm", "xy_y_acquisition_order")
            )
        geometry_keys.extend(("nshots", "data_offset"))
        for key in geometry_keys:
            detected_value = geometry_values[key]
            current_value = _editor_value(self.editors[key], self.specs[key])
            if current_value == detected_value:
                continue
            _set_editor_value(self.editors[key], self.specs[key], detected_value)
            adjustments.append(
                f"{self.specs[key].label}: {current_value} → {detected_value} "
                "(from bmotion)"
            )
        return tuple(adjustments)

    def _refresh_bmotion_geometry(self, *, show_errors=False):
        """Refresh file-backed fields after a relevant GUI selection."""
        if not self._geometry_source_is_bmotion():
            return ()
        try:
            return self.reconcile_bmotion_geometry()
        except ValueError as error:
            if show_errors:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Could not load bmotion geometry",
                    str(error),
                )
            return ()

    def _geometry_source_is_bmotion(self):
        return (
            self.editors["spatial_geometry_source"].currentData() == "bmotion"
        )

    def _update_geometry_source_controls(self, *_args):
        """Enable manual fields only when manual geometry is selected."""
        if self._setting_values:
            return
        file_backed = self._geometry_source_is_bmotion()
        self.editors["bmotion_config_name"].setEnabled(file_backed)
        geometry_keys = ["nx", "x_min_cm", "x_max_cm"]
        if self.geometry == "xy_plane":
            geometry_keys.extend(
                ("ny", "y_min_cm", "y_max_cm", "xy_y_acquisition_order")
            )
        geometry_keys.extend(("nshots", "data_offset"))
        for key in geometry_keys:
            self.editors[key].setEnabled(not file_backed)
        if file_backed:
            self._refresh_bmotion_geometry()

    def reconcile_sis_acquisition_metadata(self):
        """Synchronize unambiguous SIS metadata with the selected HDF5 file."""
        filename, digitizer = self._sis_source_values()
        config_editor = self.editors["sis_config_name"]
        config_name = config_editor.value()
        candidates = discover_sis_configurations(filename, digitizer)
        if not candidates:
            raise ValueError(
                f'No SIS configurations were found under "{digitizer}" in '
                f"{Path(filename).expanduser()}."
            )

        adjustments = []
        if config_name not in candidates:
            if len(candidates) != 1:
                valid_names = ", ".join(repr(name) for name in candidates)
                raise ValueError(
                    f'SIS configuration "{config_name}" is not present in the '
                    f"selected HDF5 file. Choose one of: {valid_names}."
                )
            selected_name = candidates[0]
            config_editor.set_value(selected_name)
            adjustments.append(
                f'SIS configuration: "{config_name}" → "{selected_name}"'
            )
            config_name = selected_name

        sample_count = discover_sis_sample_count(
            filename, digitizer, config_name
        )
        nt_editor = self.editors["nt_full"]
        configured_sample_count = _editor_value(nt_editor, self.specs["nt_full"])
        if configured_sample_count != sample_count:
            nt_editor.set_value(sample_count)
            adjustments.append(
                "Samples per trace: "
                f"{configured_sample_count} → {sample_count} (from HDF5)"
            )
        return tuple(adjustments)

    def _wire_dependencies(self):
        """Dim controls that have no effect while their feature is disabled."""
        dependencies = {
            "subtract_dc": (
                "isweep_dc_offset_start_index",
                "isweep_dc_offset_end_index",
            ),
            "calculate_shape_factor": tuple(
                key
                for key in (
                    "f_microwave_GHz",
                    "N_passes",
                    "interferometer_phase_rad",
                    "interferometer_profile_y_cm",
                )
                if key in self.editors
            ),
            "parallel_analysis": ("analysis_processes",),
            "enable_vp_spike_rejection": tuple(
                key
                for key in (
                    "vp_spike_half_window",
                    "vp_spike_threshold_V",
                    "vp_replace_flagged_with_local_interp",
                    "vp_replace_flagged_with_local_median",
                )
                if key in self.editors
            ),
            "enable_neighbor_smoothing": (
                "neighbor_smooth_half_window",
                "neighbor_smooth_sigma",
            ),
        }
        for toggle_key, dependent_keys in dependencies.items():
            toggle = self.editors[toggle_key]

            def update(enabled, keys=dependent_keys):
                for key in keys:
                    self.editors[key].setEnabled(enabled)

            toggle.toggled.connect(update)
            update(toggle.isChecked())

        geometry_source = self.editors["spatial_geometry_source"]
        geometry_source.currentIndexChanged.connect(
            self._update_geometry_source_controls
        )
        bmotion_editor = self.editors["bmotion_config_name"]
        bmotion_editor.configuration_selected.connect(
            lambda _name: self._refresh_bmotion_geometry(show_errors=True)
        )
        bmotion_editor.line_edit.editingFinished.connect(
            lambda: self._refresh_bmotion_geometry(show_errors=True)
        )
        filename_editor = self.editors["filename"]
        filename_editor.line_edit.editingFinished.connect(
            self._refresh_bmotion_geometry
        )
        filename_editor.path_selected.connect(
            lambda _path: self._refresh_bmotion_geometry()
        )
        self._update_geometry_source_controls()

    def values(self):
        return {
            key: _editor_value(self.editors[key], spec)
            for key, spec in self.specs.items()
        }

    def set_values(self, values):
        self._setting_values = True
        try:
            for key, value in values.items():
                if key in self.editors:
                    _set_editor_value(self.editors[key], self.specs[key], value)
        finally:
            self._setting_values = False
        self._update_geometry_source_controls()

    def restore_defaults(self):
        self.set_values(default_parameters(self.geometry))

    def set_running(self, running):
        self.start_button.setDisabled(running)


class LangmuirAnalysisWindow(QtWidgets.QMainWindow):
    """Tabbed analysis configuration window with a live subprocess console."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Langmuir Analysis Studio")
        self.resize(1280, 900)
        self.setMinimumSize(940, 680)
        self.process = None

        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(12)
        outer.addWidget(self._build_hero())

        self.tabs = QtWidgets.QTabWidget()
        self.parameter_tabs = {}
        for geometry in SUPPORTED_GEOMETRIES:
            tab = ParameterTab(geometry)
            tab.start_requested.connect(self.start_analysis)
            self.parameter_tabs[geometry] = tab
            self.tabs.addTab(tab, GEOMETRY_TITLES[geometry])
        self.tabs.currentChanged.connect(self._active_tab_changed)

        self.console_card = self._build_console()
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.console_card)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([680, 150])
        outer.addWidget(splitter, 1)

        saved, active_geometry, warning = load_last_parameters(LAST_PARAMETERS_PATH)
        for geometry, values in saved.items():
            self.parameter_tabs[geometry].set_values(values)
        self.tabs.setCurrentIndex(SUPPORTED_GEOMETRIES.index(active_geometry))
        if warning:
            self._append_console(warning)
        elif LAST_PARAMETERS_PATH.exists():
            self._append_console(f"Loaded settings from {LAST_PARAMETERS_PATH.name}.")
        else:
            self._append_console("No saved settings found; built-in defaults loaded.")
        self._active_tab_changed(self.tabs.currentIndex())

    def _build_hero(self):
        hero = QtWidgets.QFrame()
        hero.setObjectName("hero")
        layout = QtWidgets.QHBoxLayout(hero)
        layout.setContentsMargins(22, 18, 18, 18)
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(3)
        eyebrow = QtWidgets.QLabel("LAPD  •  LANGMUIR PROBE WORKFLOW")
        eyebrow.setObjectName("eyebrow")
        title = QtWidgets.QLabel("Langmuir Analysis Studio")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Configure acquisition, I–V physics, spatial cleanup, and interferometer calibration in one place."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        text_layout.addWidget(eyebrow)
        text_layout.addWidget(title)
        text_layout.addWidget(subtitle)
        layout.addLayout(text_layout, 1)

        controls = QtWidgets.QVBoxLayout()
        controls.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.status_badge = QtWidgets.QLabel("Ready")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setAlignment(QtCore.Qt.AlignCenter)
        save_button = QtWidgets.QPushButton("Save parameters")
        save_button.clicked.connect(self.save_parameters)
        controls.addWidget(self.status_badge)
        controls.addWidget(save_button)
        layout.addLayout(controls)
        return hero

    def _build_console(self):
        card = QtWidgets.QFrame()
        card.setObjectName("consoleCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(12, 9, 12, 12)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Analysis output")
        title.setObjectName("sectionTitle")
        self.stop_button = QtWidgets.QPushButton("Stop analysis")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_analysis)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.stop_button)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(10_000)
        layout.addLayout(header)
        layout.addWidget(self.console, 1)
        return card

    def _active_geometry(self):
        return SUPPORTED_GEOMETRIES[self.tabs.currentIndex()]

    def _active_tab_changed(self, _index):
        if self.process is None:
            geometry = self._active_geometry()
            self.status_badge.setText(f"Ready · {GEOMETRY_TITLES[geometry]}")

    def _all_values(self):
        return {geometry: tab.values() for geometry, tab in self.parameter_tabs.items()}

    def _append_console(self, text):
        self.console.appendPlainText(str(text).rstrip())
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @QtCore.Slot()
    def save_parameters(self):
        try:
            geometry_adjustments = []
            for geometry, tab in self.parameter_tabs.items():
                geometry_adjustments.extend(
                    (geometry, adjustment)
                    for adjustment in tab.reconcile_bmotion_geometry()
                )
            save_last_parameters(
                self._all_values(), self._active_geometry(), LAST_PARAMETERS_PATH
            )
        except (TypeError, ValueError, OSError) as error:
            QtWidgets.QMessageBox.critical(
                self, "Could not save parameters", str(error)
            )
            return False
        for geometry, adjustment in geometry_adjustments:
            self._append_console(
                f"Adjusted {GEOMETRY_TITLES[geometry]} {adjustment}"
            )
        self._append_console(f"Saved both tabs to {LAST_PARAMETERS_PATH.name}.")
        return True

    @QtCore.Slot(str)
    def start_analysis(self, _button_geometry):
        if self.process is not None:
            return
        # Geometry always comes from the open tab, including after keyboard or
        # programmatic tab changes.
        active_geometry = self._active_geometry()
        tab = self.parameter_tabs[active_geometry]
        try:
            metadata_adjustments = (
                tab.reconcile_bmotion_geometry()
                + tab.reconcile_sis_acquisition_metadata()
            )
            validate_parameters(
                active_geometry,
                tab.values(),
                require_input_file=True,
            )
            save_last_parameters(
                self._all_values(), active_geometry, LAST_PARAMETERS_PATH
            )
        except (TypeError, ValueError, OSError) as error:
            QtWidgets.QMessageBox.warning(self, "Check analysis parameters", str(error))
            self._append_console(f"Parameter check failed: {error}")
            return

        self.console.clear()
        self._append_console(
            f"Starting {GEOMETRY_TITLES[active_geometry]} analysis…\n"
            f"Parameters: {LAST_PARAMETERS_PATH}"
        )
        for adjustment in metadata_adjustments:
            self._append_console(f"Adjusted {adjustment}")
        process = QtCore.QProcess(self)
        process.setWorkingDirectory(str(Path(__file__).resolve().parent))
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_process_output)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                "-u",
                str(Path(__file__).with_name("langmuir_analysis.py")),
                "--run-parameters",
                str(LAST_PARAMETERS_PATH),
                "--geometry",
                active_geometry,
            ]
        )
        self.process = process
        self._set_running(True, active_geometry)
        process.start()

    def _set_running(self, running, geometry=None):
        for tab in self.parameter_tabs.values():
            tab.set_running(running)
        self.tabs.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if running:
            self.status_badge.setText(f"Running · {GEOMETRY_TITLES[geometry]}")
        else:
            self._active_tab_changed(self.tabs.currentIndex())

    @QtCore.Slot()
    def _read_process_output(self):
        if self.process is None:
            return
        raw = bytes(self.process.readAllStandardOutput())
        if raw:
            self._append_console(raw.decode("utf-8", errors="replace"))

    @QtCore.Slot(int, QtCore.QProcess.ExitStatus)
    def _process_finished(self, exit_code, exit_status):
        self._read_process_output()
        crashed = exit_status == QtCore.QProcess.CrashExit
        outcome = "crashed" if crashed else f"finished with exit code {exit_code}"
        self._append_console(f"\nAnalysis {outcome}.")
        self.process = None
        self._set_running(False)

    @QtCore.Slot(QtCore.QProcess.ProcessError)
    def _process_error(self, error):
        if self.process is not None:
            self._append_console(
                f"Process error: {self.process.errorString()} ({error})"
            )

    @QtCore.Slot()
    def stop_analysis(self):
        if self.process is None:
            return
        self._append_console("Stopping analysis…")
        process = self.process
        process.terminate()
        QtCore.QTimer.singleShot(
            3_000,
            lambda: (
                process.kill()
                if process.state() != QtCore.QProcess.NotRunning
                else None
            ),
        )

    def closeEvent(self, event):
        if self.process is not None:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Analysis is still running",
                "Stop the analysis and close the window?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.process.kill()
            self.process.waitForFinished(2_000)
        try:
            save_last_parameters(
                self._all_values(), self._active_geometry(), LAST_PARAMETERS_PATH
            )
        except (TypeError, ValueError, OSError):
            pass
        event.accept()


def launch_gui(*, splash_duration_ms=SPLASH_DURATION_MS):
    """Create the desktop application and reveal it after the startup splash."""
    application = QtWidgets.QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QtWidgets.QApplication(sys.argv[:1])
    application.setApplicationName("Langmuir Analysis Studio")
    application.setStyle("Fusion")
    application.setStyleSheet(STYLE_SHEET)

    splash = create_splash_screen(application)
    splash.show()
    splash.raise_()
    splash.activateWindow()
    # Flush the first paint before constructing the larger parameter editor.
    application.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    window = LangmuirAnalysisWindow()
    window._startup_splash = splash

    def reveal_main_window():
        window.show()
        window.raise_()
        window.activateWindow()
        splash.finish(window)
        splash.deleteLater()
        window._startup_splash = None
        window._startup_timer = None

    startup_timer = QtCore.QTimer(window)
    startup_timer.setSingleShot(True)
    startup_timer.setTimerType(QtCore.Qt.PreciseTimer)
    startup_timer.setInterval(max(0, int(splash_duration_ms)))
    startup_timer.timeout.connect(reveal_main_window)
    window._startup_timer = startup_timer
    startup_timer.start()
    if owns_application:
        return application.exec()
    return window


if __name__ == "__main__":
    raise SystemExit(launch_gui())
