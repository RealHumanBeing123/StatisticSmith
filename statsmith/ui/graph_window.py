"""Graph window: chart canvas + live graph editor (titles, colours, sizes, axes, legend ...)."""
from __future__ import annotations

import dataclasses

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox,
                               QTabWidget, QVBoxLayout, QWidget)

from ..charts import REGISTRY, ChartStyle, render
from ..charts.base import DIVERGING_CMAPS, PALETTES, SEQUENTIAL_CMAPS

FONTS = ["DejaVu Sans", "Arial", "Calibri", "Times New Roman", "Courier New", "Verdana", "Georgia", "Segoe UI"]


class ColorButton(QPushButton):
    changed = Signal(str)

    def __init__(self, color: str = "", allow_none: bool = False):
        super().__init__()
        self.allow_none = allow_none
        self.setFixedWidth(90)
        self.set_color(color)
        self.clicked.connect(self._pick)

    def set_color(self, c: str):
        self._c = c
        self.setText(c or "auto")
        self.setStyleSheet(f"background:{c};color:{'#000' if c and QColor(c).lightness() > 128 else '#fff'};" if c else "")

    def color(self) -> str:
        return self._c

    def _pick(self):
        col = QColorDialog.getColor(QColor(self._c or "#2E86C1"), self, "Choose colour")
        if col.isValid():
            self.set_color(col.name())
            self.changed.emit(self._c)

    def contextMenuEvent(self, e):
        if self.allow_none:
            self.set_color("")
            self.changed.emit("")


def _num(minv, maxv, val, step=1.0, dec=1):
    w = QDoubleSpinBox()
    w.setRange(minv, maxv); w.setDecimals(dec); w.setSingleStep(step); w.setValue(val)
    return w


class GraphWindow(QMainWindow):
    """Owns one chart. ``data`` is fetched once; the editor re-renders it with a changed :class:`ChartStyle`."""
    style_saved = Signal(object)

    def __init__(self, chart_id: str, data, params: dict, style: ChartStyle, selection: dict, dataset: str,
                 on_change_variables=None, parent=None):
        super().__init__(parent)
        self.chart_id, self.data, self.params, self.style = chart_id, data, dict(params), style
        self.selection, self.dataset = selection, dataset
        self.spec = REGISTRY[chart_id]
        self.on_change_variables = on_change_variables
        self.setWindowTitle(f"{self.spec.name} - {dataset}")
        self.resize(1200, 720)
        self.fig = Figure(figsize=(style.fig_width, style.fig_height), dpi=style.dpi)
        self.canvas = FigureCanvasQTAgg(self.fig)
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(NavigationToolbar2QT(self.canvas, self))
        v.addWidget(self.canvas, 1)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color: gray; padding: 2px 6px;")
        v.addWidget(self.notes)
        self.setCentralWidget(central)
        self._build_editor()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.redraw)
        self.redraw()

    # ------------------------------------------------------------ editor
    def _build_editor(self):
        tabs = QTabWidget()
        s = self.style
        # --- titles
        t1 = QWidget(); f = QFormLayout(t1)
        self.w_title = QLineEdit(s.title); self.w_sub = QLineEdit(s.subtitle)
        self.w_x = QLineEdit(s.xlabel); self.w_y = QLineEdit(s.ylabel); self.w_leg = QLineEdit(s.legend_title)
        for w, ph in ((self.w_title, "chart title"), (self.w_sub, "subtitle"), (self.w_x, "x-axis title (blank = automatic)"),
                      (self.w_y, "y-axis title (blank = automatic)"), (self.w_leg, "legend title")):
            w.setPlaceholderText(ph); w.textChanged.connect(self._changed)
        f.addRow("Title", self.w_title); f.addRow("Subtitle", self.w_sub); f.addRow("X-axis title", self.w_x)
        f.addRow("Y-axis title", self.w_y); f.addRow("Legend title", self.w_leg)
        tabs.addTab(t1, "Titles")
        # --- colours
        t2 = QWidget(); f = QFormLayout(t2)
        self.w_pal = QComboBox(); self.w_pal.addItems(list(PALETTES)); self.w_pal.setCurrentText(s.palette)
        self.w_pal.currentTextChanged.connect(self._changed)
        self.w_prim = ColorButton(s.primary_color, True); self.w_prim.setToolTip("Single-series colour (right-click to reset)")
        self.w_cmap = QComboBox(); self.w_cmap.addItems(SEQUENTIAL_CMAPS + DIVERGING_CMAPS); self.w_cmap.setCurrentText(s.cmap)
        self.w_bg = ColorButton(s.background); self.w_abg = ColorButton(s.axes_background)
        self.w_tc = ColorButton(s.text_color); self.w_gc = ColorButton(s.grid_color)
        self.w_alpha = _num(0.05, 1.0, s.alpha, 0.05, 2)
        for w in (self.w_prim, self.w_bg, self.w_abg, self.w_tc, self.w_gc):
            w.changed.connect(self._changed)
        self.w_cmap.currentTextChanged.connect(self._changed); self.w_alpha.valueChanged.connect(self._changed)
        f.addRow("Colour palette", self.w_pal); f.addRow("Main colour", self.w_prim); f.addRow("Colour map (heat / density)", self.w_cmap)
        f.addRow("Opacity", self.w_alpha); f.addRow("Figure background", self.w_bg); f.addRow("Plot background", self.w_abg)
        f.addRow("Text colour", self.w_tc); f.addRow("Grid colour", self.w_gc)
        self.series_box = QVBoxLayout(); f.addRow(QLabel("Colour per series / category:")); f.addRow(self.series_box)
        self.series_btns: dict[str, ColorButton] = {}
        tabs.addTab(self._scroll(t2), "Colours")
        # --- sizes
        t3 = QWidget(); f = QFormLayout(t3)
        self.w_fw = _num(2, 40, s.fig_width, 0.5); self.w_fh = _num(2, 40, s.fig_height, 0.5); self.w_dpi = QSpinBox(); self.w_dpi.setRange(50, 600); self.w_dpi.setValue(s.dpi)
        self.w_ts = _num(6, 40, s.title_size); self.w_ls = _num(6, 30, s.label_size); self.w_ks = _num(5, 24, s.tick_size)
        self.w_gs = _num(5, 24, s.legend_size); self.w_vs = _num(4, 24, s.value_label_size)
        self.w_font = QComboBox(); self.w_font.setEditable(True); self.w_font.addItems(FONTS); self.w_font.setCurrentText(s.font_family)
        self.w_lw = _num(0.2, 12, s.line_width, 0.25, 2); self.w_ms = _num(1, 30, s.marker_size, 0.5); self.w_bw = _num(0.1, 1.0, s.bar_width, 0.05, 2)
        for w in (self.w_fw, self.w_fh, self.w_dpi, self.w_ts, self.w_ls, self.w_ks, self.w_gs, self.w_vs, self.w_lw, self.w_ms, self.w_bw):
            w.valueChanged.connect(self._changed)
        self.w_font.currentTextChanged.connect(self._changed)
        for lab, w in (("Figure width (in)", self.w_fw), ("Figure height (in)", self.w_fh), ("Resolution (dpi)", self.w_dpi), ("Title size", self.w_ts),
                       ("Axis title size", self.w_ls), ("Tick label size", self.w_ks), ("Legend size", self.w_gs), ("Value label size", self.w_vs),
                       ("Font", self.w_font), ("Line width", self.w_lw), ("Marker size", self.w_ms), ("Bar width", self.w_bw)):
            f.addRow(lab, w)
        tabs.addTab(self._scroll(t3), "Sizes && fonts")
        # --- axes
        t4 = QWidget(); f = QFormLayout(t4)
        self.w_grid = QCheckBox(); self.w_grid.setChecked(s.show_grid)
        self.w_gax = QComboBox(); self.w_gax.addItems(["x", "y", "both"]); self.w_gax.setCurrentText(s.grid_axis)
        self.w_spine = QCheckBox(); self.w_spine.setChecked(s.show_spines)
        self.w_legshow = QCheckBox(); self.w_legshow.setChecked(s.show_legend)
        self.w_legloc = QComboBox(); self.w_legloc.addItems(["best", "upper right", "upper left", "lower left", "lower right", "center left", "center right", "upper center", "lower center"]); self.w_legloc.setCurrentText(s.legend_loc)
        self.w_vals = QCheckBox(); self.w_vals.setChecked(s.show_values)
        self.w_vfmt = QLineEdit(s.value_format)
        self.w_xs = QComboBox(); self.w_xs.addItems(["linear", "log", "symlog"]); self.w_xs.setCurrentText(s.xscale)
        self.w_ys = QComboBox(); self.w_ys.addItems(["linear", "log", "symlog"]); self.w_ys.setCurrentText(s.yscale)
        self.w_xlim = QLineEdit(self._lim(s.xlim)); self.w_ylim = QLineEdit(self._lim(s.ylim))
        self.w_xlim.setPlaceholderText("min, max  (blank = auto)"); self.w_ylim.setPlaceholderText("min, max  (blank = auto)")
        self.w_rot = QSpinBox(); self.w_rot.setRange(-90, 90); self.w_rot.setValue(s.rotate_xticks)
        self.w_invx = QCheckBox(); self.w_invx.setChecked(s.invert_x); self.w_invy = QCheckBox(); self.w_invy.setChecked(s.invert_y)
        for w in (self.w_grid, self.w_spine, self.w_legshow, self.w_vals, self.w_invx, self.w_invy):
            w.stateChanged.connect(self._changed)
        for w in (self.w_gax, self.w_legloc, self.w_xs, self.w_ys):
            w.currentTextChanged.connect(self._changed)
        for w in (self.w_vfmt, self.w_xlim, self.w_ylim):
            w.editingFinished.connect(self._changed)
        self.w_rot.valueChanged.connect(self._changed)
        for lab, w in (("Show grid", self.w_grid), ("Grid lines on", self.w_gax), ("Show axis frame", self.w_spine), ("Show legend", self.w_legshow),
                       ("Legend position", self.w_legloc), ("Show value labels", self.w_vals), ("Value format", self.w_vfmt),
                       ("X scale", self.w_xs), ("Y scale", self.w_ys), ("X limits", self.w_xlim), ("Y limits", self.w_ylim),
                       ("Rotate x labels (deg)", self.w_rot), ("Reverse x axis", self.w_invx), ("Reverse y axis", self.w_invy)):
            f.addRow(lab, w)
        tabs.addTab(self._scroll(t4), "Axes && legend")
        # --- chart options
        t5 = QWidget(); f = QFormLayout(t5)
        self.param_w = {}
        from .dialogs import VariableDialog
        for p in self.spec.params:
            w = VariableDialog._param_widget(p, self.params.get(p.name, p.default))
            for sig in ("valueChanged", "currentTextChanged", "stateChanged", "editingFinished"):
                if hasattr(w, sig):
                    getattr(w, sig).connect(self._params_changed)
                    break
            self.param_w[p.name] = (p, w)
            f.addRow(p.label, w)
        if self.on_change_variables:
            b = QPushButton("Change variables…")
            b.clicked.connect(lambda: self.on_change_variables(self))
            f.addRow(b)
        tabs.addTab(self._scroll(t5), "Chart options")

        bar = QWidget(); h = QHBoxLayout(bar)
        for label, fn in (("Save image…", self.save_image), ("Copy", self.copy_image), ("Reset style", self.reset_style)):
            b = QPushButton(label); b.clicked.connect(fn); h.addWidget(b)
        holder = QWidget(); vv = QVBoxLayout(holder); vv.addWidget(tabs); vv.addWidget(bar)
        dock = QDockWidget("Graph editor", self)
        dock.setWidget(holder)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setMinimumWidth(340)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    @staticmethod
    def _scroll(w: QWidget) -> QScrollArea:
        sa = QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(w); sa.setFrameShape(QScrollArea.NoFrame)
        return sa

    @staticmethod
    def _lim(v) -> str:
        return f"{v[0]:g}, {v[1]:g}" if v else ""

    @staticmethod
    def _parse_lim(t: str):
        try:
            a, b = [float(x) for x in t.replace(";", ",").split(",")]
            return (a, b) if a != b else None
        except Exception:
            return None

    # ------------------------------------------------------------ style <-> widgets
    def _changed(self, *_):
        self._timer.start()

    def _params_changed(self, *_):
        for n, (p, w) in self.param_w.items():
            if hasattr(w, "isChecked") and not hasattr(w, "currentText"):
                self.params[n] = w.isChecked()
            elif hasattr(w, "currentText"):
                t = w.currentText(); ch = [c for c in p.choices if str(c) == t]
                self.params[n] = ch[0] if ch else t
            elif hasattr(w, "value"):
                self.params[n] = w.value()
            else:
                self.params[n] = w.text()
        self._timer.start()

    def _collect(self) -> ChartStyle:
        s = dataclasses.replace(self.style)
        s.title, s.subtitle, s.xlabel, s.ylabel, s.legend_title = (self.w_title.text(), self.w_sub.text(), self.w_x.text(),
                                                                    self.w_y.text(), self.w_leg.text())
        s.palette, s.primary_color, s.cmap = self.w_pal.currentText(), self.w_prim.color(), self.w_cmap.currentText()
        s.alpha = self.w_alpha.value()
        s.background, s.axes_background, s.text_color, s.grid_color = self.w_bg.color(), self.w_abg.color(), self.w_tc.color(), self.w_gc.color()
        s.colors = {k: b.color() for k, b in self.series_btns.items() if b.color()}
        s.fig_width, s.fig_height, s.dpi = self.w_fw.value(), self.w_fh.value(), self.w_dpi.value()
        s.title_size, s.label_size, s.tick_size, s.legend_size, s.value_label_size = (self.w_ts.value(), self.w_ls.value(), self.w_ks.value(),
                                                                                       self.w_gs.value(), self.w_vs.value())
        s.font_family, s.line_width, s.marker_size, s.bar_width = self.w_font.currentText(), self.w_lw.value(), self.w_ms.value(), self.w_bw.value()
        s.show_grid, s.grid_axis, s.show_spines = self.w_grid.isChecked(), self.w_gax.currentText(), self.w_spine.isChecked()
        s.show_legend, s.legend_loc, s.show_values, s.value_format = self.w_legshow.isChecked(), self.w_legloc.currentText(), self.w_vals.isChecked(), self.w_vfmt.text() or "{:,.2f}"
        s.xscale, s.yscale = self.w_xs.currentText(), self.w_ys.currentText()
        s.xlim, s.ylim = self._parse_lim(self.w_xlim.text()), self._parse_lim(self.w_ylim.text())
        s.rotate_xticks, s.invert_x, s.invert_y = self.w_rot.value(), self.w_invx.isChecked(), self.w_invy.isChecked()
        return s

    def _load_into_widgets(self, s: ChartStyle):
        for w in self._inputs():
            w.blockSignals(True)
        self.w_title.setText(s.title); self.w_sub.setText(s.subtitle); self.w_x.setText(s.xlabel); self.w_y.setText(s.ylabel); self.w_leg.setText(s.legend_title)
        self.w_pal.setCurrentText(s.palette); self.w_prim.set_color(s.primary_color); self.w_cmap.setCurrentText(s.cmap); self.w_alpha.setValue(s.alpha)
        self.w_bg.set_color(s.background); self.w_abg.set_color(s.axes_background); self.w_tc.set_color(s.text_color); self.w_gc.set_color(s.grid_color)
        self.w_fw.setValue(s.fig_width); self.w_fh.setValue(s.fig_height); self.w_dpi.setValue(s.dpi)
        for w, v in ((self.w_ts, s.title_size), (self.w_ls, s.label_size), (self.w_ks, s.tick_size), (self.w_gs, s.legend_size), (self.w_vs, s.value_label_size),
                     (self.w_lw, s.line_width), (self.w_ms, s.marker_size), (self.w_bw, s.bar_width)):
            w.setValue(v)
        self.w_font.setCurrentText(s.font_family); self.w_grid.setChecked(s.show_grid); self.w_gax.setCurrentText(s.grid_axis)
        self.w_spine.setChecked(s.show_spines); self.w_legshow.setChecked(s.show_legend); self.w_legloc.setCurrentText(s.legend_loc)
        self.w_vals.setChecked(s.show_values); self.w_vfmt.setText(s.value_format); self.w_xs.setCurrentText(s.xscale); self.w_ys.setCurrentText(s.yscale)
        self.w_xlim.setText(self._lim(s.xlim)); self.w_ylim.setText(self._lim(s.ylim)); self.w_rot.setValue(s.rotate_xticks)
        self.w_invx.setChecked(s.invert_x); self.w_invy.setChecked(s.invert_y)
        for w in self._inputs():
            w.blockSignals(False)

    def _inputs(self):
        return [w for t in (QLineEdit, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox) for w in self.findChildren(t)]

    def reset_style(self):
        from .theme import colors
        bg = self.style.background
        self.style = ChartStyle.for_theme("dark" if bg.lower() < "#808080" else "light")
        self._load_into_widgets(self.style)
        self.series_btns.clear()
        self.redraw()

    # ------------------------------------------------------------ drawing
    def redraw(self):
        self.style = self._collect()
        self.data.notes.clear()
        try:
            self.fig.set_size_inches(self.style.fig_width, self.style.fig_height)
            self.fig.set_dpi(self.style.dpi)
            render(self.chart_id, self.data, self.style, fig=self.fig, **self.params)
            self.notes.setText("  •  ".join(self.data.notes))
        except Exception as e:  # noqa: BLE001
            self.fig.clear()
            self.fig.text(0.5, 0.5, f"Could not draw this chart:\n{str(e)[:300]}", ha="center", va="center", wrap=True)
            self.notes.setText("")
        self.canvas.draw_idle()
        self._refresh_series_buttons()

    def _refresh_series_buttons(self):
        labels: list[str] = []
        for ax in self.fig.axes:
            h, l = ax.get_legend_handles_labels()
            labels += [x for x in l if x and not x.startswith("_") and x not in labels]
        labels = labels[:30]
        if list(self.series_btns) == labels:
            return
        keep = {k: b.color() for k, b in self.series_btns.items()}
        while self.series_box.count():
            it = self.series_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.series_btns = {}
        for i, lab in enumerate(labels):
            row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(lab[:28]), 1)
            b = ColorButton(keep.get(lab) or self.style.colors.get(lab, ""), True)
            b.changed.connect(self._changed)
            h.addWidget(b)
            self.series_btns[lab] = b
            self.series_box.addWidget(row)

    # ------------------------------------------------------------ output
    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save graph", f"{self.chart_id}.png",
                                              "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;JPEG (*.jpg);;TIFF (*.tif)")
        if path:
            self.fig.savefig(path, dpi=max(self.style.dpi, 200), facecolor=self.fig.get_facecolor())
            self.last_saved = path

    def copy_image(self):
        self.canvas.draw()
        w, h = self.canvas.get_width_height(physical=True)
        img = QImage(bytes(self.canvas.buffer_rgba()), w, h, QImage.Format_RGBA8888).copy()
        QGuiApplication.clipboard().setImage(img)

    def replace_data(self, data, selection: dict, params: dict | None = None):
        self.data, self.selection = data, selection
        if params:
            self.params.update(params)
        self.redraw()
