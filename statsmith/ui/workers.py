"""Run blocking engine work off the GUI thread."""
from __future__ import annotations

import threading
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal


class Worker(QThread):
    done = Signal(object)
    failed = Signal(str, str)         # short message, traceback
    progress = Signal(float, str)

    def __init__(self, fn: Callable[..., Any], *args, with_progress: bool = False, parent: QObject | None = None, **kw):
        super().__init__(parent)
        self.fn, self.args, self.kw, self.with_progress = fn, args, kw, with_progress
        self.cancel = threading.Event()

    def run(self):
        try:
            if self.with_progress:
                self.kw["progress"] = lambda f, m: self.progress.emit(float(f), str(m))
                self.kw["cancel"] = self.cancel
            self.done.emit(self.fn(*self.args, **self.kw))
        except BaseException as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            self.failed.emit(f"{type(e).__name__}: {msg}" if not isinstance(e, (ValueError, KeyError)) else msg, traceback.format_exc())


_ACTIVE: set = set()


def run_async(fn, on_done=None, on_error=None, on_progress=None, parent=None, with_progress=False, **kw) -> Worker:
    """Start ``fn(**kw)`` in a thread; callbacks run on the GUI thread. Returns the worker (has ``.cancel``)."""
    w = Worker(fn, with_progress=with_progress, parent=parent, **kw)
    _ACTIVE.add(w)
    if on_done:
        w.done.connect(on_done)
    if on_error:
        w.failed.connect(on_error)
    if on_progress:
        w.progress.connect(on_progress)
    w.finished.connect(lambda: (_ACTIVE.discard(w), w.deleteLater()))
    w.start()
    return w
