"""Light / dark themes (Fusion style + palette + a little QSS)."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

LIGHT = dict(window="#f3f5f8", base="#ffffff", alt="#f7f9fc", text="#1d2733", button="#e9edf3", accent="#2e6db4",
             accent_text="#ffffff", grid="#d5dbe4", header="#e6ebf2", muted="#6b7785")
DARK = dict(window="#1e1f22", base="#25272b", alt="#2b2d31", text="#e6e6e6", button="#34373d", accent="#4a90d9",
            accent_text="#ffffff", grid="#3b3e44", header="#2f3238", muted="#9aa3ad")


def rgba(hexc: str, alpha_pct: int) -> str:
    q = QColor(hexc)
    return f"rgba({q.red()},{q.green()},{q.blue()},{int(alpha_pct * 2.55)})"


def colors(theme: str) -> dict:
    return DARK if theme == "dark" else LIGHT


def apply_theme(app: QApplication, theme: str, font_size: int = 10):
    c = colors(theme)
    app.setStyle("Fusion")
    pal = QPalette()
    q = QColor
    pal.setColor(QPalette.Window, q(c["window"]))
    pal.setColor(QPalette.WindowText, q(c["text"]))
    pal.setColor(QPalette.Base, q(c["base"]))
    pal.setColor(QPalette.AlternateBase, q(c["alt"]))
    pal.setColor(QPalette.Text, q(c["text"]))
    pal.setColor(QPalette.Button, q(c["button"]))
    pal.setColor(QPalette.ButtonText, q(c["text"]))
    pal.setColor(QPalette.ToolTipBase, q(c["base"]))
    pal.setColor(QPalette.ToolTipText, q(c["text"]))
    pal.setColor(QPalette.Highlight, q(c["accent"]))
    pal.setColor(QPalette.HighlightedText, q(c["accent_text"]))
    pal.setColor(QPalette.PlaceholderText, q(c["muted"]))
    pal.setColor(QPalette.Link, q(c["accent"]))
    pal.setColor(QPalette.Disabled, QPalette.Text, q(c["muted"]))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, q(c["muted"]))
    app.setPalette(pal)
    f = QFont(app.font())
    f.setPointSize(font_size)
    app.setFont(f)
    app.setStyleSheet(f"""
    QToolBar {{ background: {c['header']}; border-bottom: 1px solid {c['grid']}; spacing: 4px; padding: 3px; }}
    QToolButton {{ padding: 4px 9px; border: 1px solid transparent; border-radius: 4px; }}
    QToolButton:hover {{ background: {c['button']}; border-color: {c['grid']}; }}
    QToolButton:checked {{ background: {c['accent']}; color: {c['accent_text']}; }}
    QToolButton::menu-indicator {{ subcontrol-position: right center; }}
    QHeaderView::section {{ background: {c['header']}; color: {c['text']}; padding: 3px 6px; border: 0px;
        border-right: 1px solid {c['grid']}; border-bottom: 1px solid {c['grid']}; }}
    QTableView {{ gridline-color: {c['grid']}; selection-background-color: {rgba(c['accent'], 70)}; selection-color: {c['text']}; }}
    QTabBar::tab {{ padding: 5px 14px; border: 1px solid {c['grid']}; background: {c['button']}; }}
    QTabBar::tab:selected {{ background: {c['base']}; font-weight: bold; }}
    QStatusBar {{ background: {c['header']}; border-top: 1px solid {c['grid']}; }}
    QPlainTextEdit, QTextEdit {{ font-family: Consolas, 'DejaVu Sans Mono', monospace; }}
    QMenu::item:disabled {{ color: {c['muted']}; }}
    """)
