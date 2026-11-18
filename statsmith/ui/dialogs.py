"""Variable-selection, results, settings, codebook and script dialogs."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QTextCursor, QTextDocument
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPlainTextEdit, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget, QWidgetItem)

from ..data.engine import system_resources
from ..settings import Settings

NONE_LABEL = "(none)"


def mono(size: int = 10) -> QFont:
    f = QFont("Consolas")
    f.setStyleHint(QFont.Monospace)
    f.setFixedPitch(True)
    f.setPointSize(size)
    return f


# =========================================================================== variable selection
class VariableDialog(QDialog):
    """Pick one column per single role, several for ``*_multi`` roles, plus the method's options.

    Columns highlighted in the grid are pre-assigned to the roles in order (numeric roles take numeric columns first).
    """

    def __init__(self, title: str, roles, params, dataset, preselected: list[str] | None = None, description: str = "",
                 parent=None, defaults: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(460)
        self.roles, self.params, self.ds = list(roles), list(params), dataset
        self.role_w: dict = {}
        self.param_w: dict = {}
        lay = QVBoxLayout(self)
        if description:
            lab = QLabel(description)
            lab.setWordWrap(True)
            lab.setStyleSheet("color: gray;")
            lay.addWidget(lab)
        form = QFormLayout()
        lay.addLayout(form)
        assign = self._auto_assign(preselected or [])
        for r in self.roles:
            cols = self._allowed(r)
            label = (r.label or r.name) + ("" if r.required else "  (optional)")
            if r.multi:
                w = QListWidget()
                w.setSelectionMode(QAbstractItemView.MultiSelection)
                w.setMaximumHeight(130)
                for c in cols:
                    it = QListWidgetItem(c)
                    w.addItem(it)
                    if c in assign.get(r.name, []):
                        it.setSelected(True)
            else:
                w = QComboBox()
                if not r.required:
                    w.addItem(NONE_LABEL)
                w.addItems(cols)
                sel = assign.get(r.name)
                if sel:
                    w.setCurrentText(sel[0])
            if r.help:
                w.setToolTip(r.help)
            self.role_w[r.name] = w
            form.addRow(label, w)
        if self.params:
            box = QGroupBox("Options")
            pf = QFormLayout(box)
            for p in self.params:
                w = self._param_widget(p, (defaults or {}).get(p.name, p.default))
                self.param_w[p.name] = w
                pf.addRow(p.label, w)
            lay.addWidget(box)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    # ------------------------------------------------------------ helpers
    def _allowed(self, role) -> list[str]:
        if role.kind.startswith("numeric"):
            return [c.name for c in self.ds.columns if c.is_numeric] or self.ds.names
        return self.ds.names

    def _auto_assign(self, pre: list[str]) -> dict:
        out: dict = {}
        left = list(pre)
        for r in self.roles:
            if not left:
                break
            allowed = self._allowed(r)
            take = [c for c in left if c in allowed]
            if not take:
                continue
            if r.multi:
                out[r.name] = take
                left = [c for c in left if c not in take]
            elif r.required or len(pre) >= len([x for x in self.roles if x.required]):
                out[r.name] = [take[0]]
                left.remove(take[0])
        return out

    @staticmethod
    def _param_widget(p, default):
        if p.kind == "int":
            w = QSpinBox()
            w.setRange(int(p.min if p.min is not None else -10**9), int(p.max if p.max is not None else 10**9))
            w.setValue(int(default if default is not None else 0))
        elif p.kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setRange(float(p.min if p.min is not None else -1e12), float(p.max if p.max is not None else 1e12))
            w.setSingleStep(0.01 if (p.max is not None and p.max <= 1) else 0.1)
            w.setValue(float(default if default is not None else 0))
        elif p.kind == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in p.choices])
            w.setCurrentText(str(default))
        elif p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
        else:
            w = QLineEdit(str(default or ""))
        if p.help:
            w.setToolTip(p.help)
        return w

    def selection(self) -> dict:
        out = {}
        for r in self.roles:
            w = self.role_w[r.name]
            if r.multi:
                out[r.name] = [i.text() for i in w.selectedItems()]
                out[r.name].sort(key=lambda c: self.ds.names.index(c))
            else:
                t = w.currentText()
                out[r.name] = None if t == NONE_LABEL or t == "" else t
        return out

    def values(self) -> dict:
        out = {}
        for p in self.params:
            w = self.param_w[p.name]
            if isinstance(w, QCheckBox):
                out[p.name] = w.isChecked()
            elif isinstance(w, QComboBox):
                t = w.currentText()
                ch = [c for c in p.choices if str(c) == t]
                out[p.name] = ch[0] if ch else t
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                out[p.name] = w.value()
            else:
                out[p.name] = w.text()
        return out

    def _accept(self):
        sel = self.selection()
        miss = [r.label or r.name for r in self.roles if r.required and not sel.get(r.name)]
        if miss:
            QMessageBox.warning(self, self.windowTitle(), "Please choose: " + ", ".join(miss))
            return
        self.accept()


# =========================================================================== text results
class TextWindow(QDialog):
    """Non-modal text viewer with search, copy and 'Save as .txt'."""

    def __init__(self, title: str, text: str, parent=None, size=(760, 560), suggested: str = "results.txt"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(*size)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.suggested = suggested
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.find = QLineEdit()
        self.find.setPlaceholderText("Find…  (Enter = next)")
        self.find.returnPressed.connect(self._find)
        top.addWidget(self.find)
        lay.addLayout(top)
        self.edit = QPlainTextEdit()
        self.edit.setFont(mono(10))
        self.edit.setReadOnly(True)
        self.edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.edit.setPlainText(text)
        lay.addWidget(self.edit)
        row = QHBoxLayout()
        for label, fn in (("Copy", lambda: (self.edit.selectAll(), self.edit.copy())), ("Save as text file…", self.save)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        lay.addLayout(row)

    def set_text(self, text: str):
        self.edit.setPlainText(text)

    def _find(self):
        if not self.edit.find(self.find.text()):
            self.edit.moveCursor(QTextCursor.Start)
            self.edit.find(self.find.text())

    def save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save as text file", self.suggested, "Text files (*.txt);;All files (*)")
        if path:
            Path(path).write_text(self.edit.toPlainText(), encoding="utf-8")


class CodebookDialog(TextWindow):
    def __init__(self, text: str, sidecar: str = "", parent=None, name: str = "dataset"):
        super().__init__(f"Codebook - {name}", text, parent, (860, 640), suggested=f"{name}_codebook.txt")
        if sidecar and Path(sidecar).exists():
            b = QPushButton("Open attached codebook file")
            b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(sidecar)))
            self.layout().itemAt(self.layout().count() - 1).layout().insertWidget(2, b)


# =========================================================================== script (.do) editor
class ScriptWindow(QDialog):
    """Shows the reproducible script generated from everything the user did; fully editable."""

    def __init__(self, recorder, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Script (reproducible .do / .R / .py)")
        self.resize(820, 620)
        self.rec, self.settings = recorder, settings
        self.dirty = False
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Language:"))
        self.lang = QComboBox()
        self.lang.addItems(["Stata (.do)", "R (.R)", "Python (.py)"])
        self.lang.setCurrentIndex({"stata": 0, "r": 1, "python": 2}.get(settings.do_language, 0))
        self.lang.currentIndexChanged.connect(lambda *_: self.regenerate(force=False))
        top.addWidget(self.lang)
        self.auto = QCheckBox("Update automatically as I work")
        self.auto.setChecked(settings.do_autogenerate)
        top.addWidget(self.auto)
        top.addStretch(1)
        regen = QPushButton("Regenerate (discard edits)")
        regen.clicked.connect(lambda: self.regenerate(force=True))
        top.addWidget(regen)
        lay.addLayout(top)
        self.edit = QPlainTextEdit()
        self.edit.setFont(mono(10))
        self.edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.edit.textChanged.connect(self._on_edit)
        lay.addWidget(self.edit)
        note = QLabel("You can edit this script freely (remove errors, add comments). It is only overwritten when you press "
                      "Regenerate. Run it in Stata / R / Python to verify every result.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        lay.addWidget(note)
        row = QHBoxLayout()
        b = QPushButton("Save as…")
        b.clicked.connect(self.save)
        row.addWidget(b)
        b2 = QPushButton("Open script file…")
        b2.clicked.connect(self.open_file)
        row.addWidget(b2)
        row.addStretch(1)
        c = QPushButton("Close")
        c.clicked.connect(self.close)
        row.addWidget(c)
        lay.addLayout(row)
        self._loading = False
        self.regenerate(force=True)

    @property
    def language(self) -> str:
        return ("stata", "r", "python")[self.lang.currentIndex()]

    def _on_edit(self):
        if not self._loading:
            self.dirty = True

    def regenerate(self, force: bool = False):
        if self.dirty and not force:
            r = QMessageBox.question(self, "Script", "You edited the script. Replace it with a freshly generated one?")
            if r != QMessageBox.Yes:
                return
        self._loading = True
        self.edit.setPlainText(self.rec.render(self.language))
        self._loading = False
        self.dirty = False

    def on_recorder_changed(self):
        """Called by the main window after every recorded step."""
        if self.auto.isChecked() and not self.dirty:
            sb = self.edit.verticalScrollBar()
            self._loading = True
            self.edit.setPlainText(self.rec.render(self.language))
            self._loading = False
            sb.setValue(sb.maximum())

    def save(self):
        ext = {"stata": "do", "r": "R", "python": "py"}[self.language]
        path, _ = QFileDialog.getSaveFileName(self, "Save script", f"analysis.{ext}", f"Script (*.{ext});;All files (*)")
        if path:
            Path(path).write_text(self.edit.toPlainText(), encoding="utf-8")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open script", "", "Scripts (*.do *.R *.r *.py *.txt);;All files (*)")
        if path:
            self._loading = True
            self.edit.setPlainText(Path(path).read_text(encoding="utf-8", errors="replace"))
            self._loading = False
            self.dirty = True


# =========================================================================== settings
class SettingsDialog(QDialog):
    def __init__(self, s: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self.s = s
        res = system_resources()
        self.max_ram = max(res["ram_gb"], 0.5)
        self.max_cpu = int(res["logical_cpus"])
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)
        self.theme = QComboBox()
        self.theme.addItems(["Light mode", "Dark mode"])
        self.theme.setCurrentIndex(1 if s.theme == "dark" else 0)
        form.addRow("Appearance", self.theme)
        self.font = QSpinBox(); self.font.setRange(8, 20); self.font.setValue(s.font_size)
        form.addRow("Font size", self.font)
        self.rowh = QSpinBox(); self.rowh.setRange(16, 40); self.rowh.setValue(s.grid_row_height)
        form.addRow("Grid row height", self.rowh)
        self.dec = QSpinBox(); self.dec.setRange(1, 12); self.dec.setValue(s.decimals)
        form.addRow("Decimals in text results", self.dec)
        self.seed = QSpinBox(); self.seed.setRange(0, 2**31 - 1); self.seed.setValue(s.seed)
        self.seed.setToolTip("Random seed used for sampling and permutation tests so results are reproducible.")
        form.addRow("Random seed", self.seed)

        box = QGroupBox(f"Resources  (this computer: {res['ram_gb']:.0f} GB RAM, {self.max_cpu} logical CPU cores)")
        bf = QFormLayout(box)
        rr = QHBoxLayout()
        self.ram = QDoubleSpinBox()
        self.ram.setDecimals(1); self.ram.setRange(0.25, self.max_ram); self.ram.setSuffix(" GB"); self.ram.setValue(min(s.ram_gb, self.max_ram))
        self.ram_sl = QSlider(Qt.Horizontal)
        self.ram_sl.setRange(1, int(self.max_ram * 4))
        self.ram_sl.setValue(int(self.ram.value() * 4))
        self.ram_sl.valueChanged.connect(lambda v: self.ram.setValue(v / 4))
        self.ram.valueChanged.connect(lambda v: self.ram_sl.setValue(int(v * 4)))
        rr.addWidget(self.ram_sl); rr.addWidget(self.ram)
        bf.addRow("Memory limit", rr)
        cr = QHBoxLayout()
        self.cores = QSpinBox(); self.cores.setRange(1, self.max_cpu); self.cores.setValue(min(s.cores, self.max_cpu))
        self.core_sl = QSlider(Qt.Horizontal); self.core_sl.setRange(1, self.max_cpu); self.core_sl.setValue(self.cores.value())
        self.core_sl.valueChanged.connect(self.cores.setValue)
        self.cores.valueChanged.connect(self.core_sl.setValue)
        cr.addWidget(self.core_sl); cr.addWidget(self.cores)
        bf.addRow("CPU cores", cr)
        hint = QLabel("Default: 2 GB and 4 cores. Larger datasets spill to disk automatically if they exceed the memory limit, "
                      "so a small limit is safe; raise it for speed.")
        hint.setWordWrap(True); hint.setStyleSheet("color: gray;")
        bf.addRow(hint)
        lay.addWidget(box)

        box2 = QGroupBox("Reproducible script files (.do / .R / .py)")
        f2 = QFormLayout(box2)
        self.lang = QComboBox(); self.lang.addItems(["Stata (.do)", "R (.R)", "Python (.py)"])
        self.lang.setCurrentIndex({"stata": 0, "r": 1, "python": 2}.get(s.do_language, 0))
        f2.addRow("Language", self.lang)
        self.auto = QCheckBox("Show the script live and update it as I work"); self.auto.setChecked(s.do_autogenerate)
        f2.addRow(self.auto)
        self.save_do = QCheckBox("Also write the script file next to the project whenever I save a project"); self.save_do.setChecked(s.do_autosave_with_project)
        f2.addRow(self.save_do)
        lay.addWidget(box2)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._defaults)
        lay.addWidget(bb)

    def _defaults(self):
        self.ram.setValue(min(2.0, self.max_ram)); self.cores.setValue(min(4, self.max_cpu))

    def apply_to(self, s: Settings) -> Settings:
        s.theme = "dark" if self.theme.currentIndex() == 1 else "light"
        s.font_size, s.grid_row_height, s.decimals, s.seed = self.font.value(), self.rowh.value(), self.dec.value(), self.seed.value()
        s.ram_gb, s.cores = float(self.ram.value()), int(self.cores.value())
        s.do_language = ("stata", "r", "python")[self.lang.currentIndex()]
        s.do_autogenerate, s.do_autosave_with_project = self.auto.isChecked(), self.save_do.isChecked()
        return s.clamped()


# =========================================================================== import options
class ChoiceDialog(QDialog):
    """Pick one item from a list (Excel sheet, R object)."""

    def __init__(self, title: str, label: str, items: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(label))
        self.box = QComboBox(); self.box.addItems(items)
        lay.addWidget(self.box)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def value(self) -> str:
        return self.box.currentText()
