"""Guards against the packaged app silently missing modules or data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_registries_populated():
    import statsmith.charts as c, statsmith.stats as s
    assert len(s.REGISTRY) >= 77 and len(c.REGISTRY) >= 90


def test_bundled_data_present():
    d = ROOT / "statsmith" / "data"
    for f in ("examples/trial.csv", "examples/trial.dta", "examples/trial.codebook.txt", "geo/us_states.geojson", "geo/world.geojson", "geo/us_tile_grid.csv"):
        assert (d / f).exists(), f


def test_selftest_imports_examples(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from statsmith.app import selftest
    assert selftest([]) == 0


def test_gui_selftest(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from statsmith.app import gui_selftest
    assert gui_selftest([]) == 0
