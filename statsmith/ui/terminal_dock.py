"""Terminal panel: SQL / Python / Stata / R command line over the loaded datasets."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (QComboBox, QDockWidget, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from ..terminal import Terminal
from .dialogs import mono
from .workers import run_async

MODES = [("SQL", "sql"), ("Python", "python"), ("Stata", "stata"), ("R", "r")]


class _Input(QPlainTextEdit):
    submit = Signal()
    history = Signal(int)

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and not (e.modifiers() & Qt.ShiftModifier):
            self.submit.emit()
            return
        if e.key() in (Qt.Key_Up, Qt.Key_Down) and (e.modifiers() & Qt.ControlModifier):
            self.history.emit(-1 if e.key() == Qt.Key_Up else 1)
            return
        super().keyPressEvent(e)


class TerminalDock(QDockWidget):
    dataset_changed = Signal(str)
    switch_dataset = Signal(str)
    figure_requested = Signal(str, dict, dict)
    recorded = Signal()

    def __init__(self, terminal: Terminal, parent=None):
        super().__init__("Terminal", parent)
        self.term = terminal
        self.hist: list[tuple[str, str]] = []
        self.hpos = 0
        self.busy = False
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        top.addWidget(QLabel("Language:"))
        self.mode = QComboBox()
        for label, _ in MODES:
            self.mode.addItem(label)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        top.addWidget(self.mode)
        self.dsl = QLabel("")
        top.addWidget(self.dsl)
        top.addStretch(1)
        for label, fn in (("Help", self._help), ("Clear", lambda: self.out.clear())):
            b = QPushButton(label)
            b.clicked.connect(fn)
            top.addWidget(b)
        lay.addLayout(top)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setFont(mono(10))
        lay.addWidget(self.out, 1)
        row = QHBoxLayout()
        self.prompt = QLabel("sql>")
        self.prompt.setFont(mono(10))
        row.addWidget(self.prompt)
        self.inp = _Input()
        self.inp.setFont(mono(10))
        self.inp.setFixedHeight(64)
        self.inp.setPlaceholderText("Enter = run, Shift+Enter = new line, Ctrl+Up/Down = history, type 'help' for commands")
        self.inp.submit.connect(self.run)
        self.inp.history.connect(self._history)
        row.addWidget(self.inp, 1)
        go = QPushButton("Run")
        go.clicked.connect(self.run)
        row.addWidget(go)
        lay.addLayout(row)
        self.setWidget(w)
        self.out.appendPlainText("Statistic Smith terminal.  Pick a language above; 'help' lists the commands.\n"
                                 "Everything you run here is also written to the reproducible script.\n")

    @property
    def lang(self) -> str:
        return MODES[self.mode.currentIndex()][1]

    def set_dataset_label(self, name: str | None):
        self.dsl.setText(f"   active dataset: {name}" if name else "   (no dataset loaded)")

    def _mode_changed(self):
        self.prompt.setText(self.lang + ">")

    def _help(self):
        self.inp.setPlainText("help")
        self.run()

    def _history(self, d: int):
        if not self.hist:
            return
        self.hpos = min(max(self.hpos + d, 0), len(self.hist))
        if self.hpos == len(self.hist):
            self.inp.clear()
        else:
            m, t = self.hist[self.hpos]
            self.inp.setPlainText(t)

    def run(self):
        text = self.inp.toPlainText().strip()
        if not text or self.busy:
            return
        lang = self.lang
        self.hist.append((lang, text))
        self.hpos = len(self.hist)
        self.out.appendPlainText(f"{lang}> {text}")
        self.inp.clear()
        self.busy = True
        self.inp.setEnabled(False)

        def done(res):
            self.busy = False
            self.inp.setEnabled(True)
            self.inp.setFocus()
            if res.text:
                self.out.appendPlainText(res.text)
            self.out.appendPlainText("")
            if res.dataset_changed:
                self.dataset_changed.emit(res.dataset_changed)
            if res.switch_dataset:
                self.switch_dataset.emit(res.switch_dataset)
            if res.figure_request:
                self.figure_requested.emit(*res.figure_request)
            self.recorded.emit()

        def fail(msg, tb):
            self.busy = False
            self.inp.setEnabled(True)
            self.out.appendPlainText("error: " + msg + "\n")

        run_async(self.term.execute, done, fail, parent=self, mode=lang, line=text)
