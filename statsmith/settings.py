"""User settings, stored as JSON (``%APPDATA%/StatisticSmith/settings.json`` on Windows)."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_RAM_GB = 2
DEFAULT_CORES = 4


def config_dir() -> Path:
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    p = base / "StatisticSmith"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    p = base / "StatisticSmith" / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def system_limits() -> tuple[float, int]:
    """(total RAM in GB, logical CPU count). Used to bound the settings sliders."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3, psutil.cpu_count(logical=True) or 1
    except Exception:
        return 128.0, os.cpu_count() or 1


@dataclass
class Settings:
    theme: str = "light"                  # light | dark
    ram_gb: float = DEFAULT_RAM_GB        # DuckDB memory budget
    cores: int = DEFAULT_CORES            # worker threads
    do_language: str = "stata"            # stata | r | python
    do_autogenerate: bool = True          # log every action into the script as the user works
    do_autosave_with_project: bool = True # write <project>.do / .R next to the project on save
    font_size: int = 10
    grid_row_height: int = 22
    decimals: int = 4                     # numbers in text output
    recent_files: list = field(default_factory=list)
    welcomed: bool = False                # first-run welcome / example dataset shown
    seed: int = 12345                     # RNG seed for reproducible results (permutation tests, sampling)

    # ------------------------------------------------------------------ persistence
    @classmethod
    def path(cls) -> Path:
        return config_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        try:
            raw = json.loads(cls.path().read_text(encoding="utf-8"))
            names = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in raw.items() if k in names}).clamped()
        except Exception:
            return cls()

    def save(self) -> None:
        try:
            self.path().write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError:
            pass

    def clamped(self) -> "Settings":
        """Resource limits can go as high as the machine allows (never below 0.25 GB / 1 core)."""
        ram, cpus = system_limits()
        self.ram_gb = float(min(max(self.ram_gb, 0.25), max(ram, 0.25)))
        self.cores = int(min(max(self.cores, 1), max(cpus, 1)))
        return self

    def add_recent(self, path: str) -> None:
        self.recent_files = [p for p in self.recent_files if p != path][:0] + [path] + [p for p in self.recent_files if p != path][:9]
