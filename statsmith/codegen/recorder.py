"""Script recorder: every action the user performs becomes a line of Stata (.do), R or Python code.

The recorded *steps* are language-neutral; :meth:`Recorder.render` turns them into a script on demand, so the user
can switch language in Settings at any time and the existing history is re-rendered. Users can also edit the
rendered text in the Script Editor (the edited text then becomes the source of truth until they press "Regenerate").
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__

LANGS = ("stata", "r", "python")
EXT = {"stata": ".do", "r": ".R", "python": ".py"}


# ----------------------------------------------------------------------------- names
def stata_name(n: str) -> str:
    s = re.sub(r"\W", "_", str(n), flags=re.ASCII)
    if not s or s[0].isdigit():
        s = "_" + s
    return s[:32]


def r_name(n: str) -> str:
    """Same algorithm as R's make.names (ASCII subset)."""
    s = re.sub(r"[^A-Za-z0-9._]", ".", str(n))
    if not s or not re.match(r"[A-Za-z.]", s) or re.match(r"\.\d", s):
        s = "X" + s
    return s


def py_str(s: Any) -> str:
    return repr(str(s))


def r_str(s: Any) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _lit(lang: str, v: Any) -> str:
    """Literal of a parameter value in ``lang`` (used for `{name_r}` placeholders)."""
    if isinstance(v, bool):
        return ("TRUE" if v else "FALSE") if lang == "r" else str(v)
    if isinstance(v, (int, float)):
        return repr(v)
    if v in ("", None):
        return "NULL" if lang == "r" else "None"
    s = str(v)
    try:
        float(s)
        return s
    except ValueError:
        return r_str(s) if lang == "r" else py_str(s)


# ----------------------------------------------------------------------------- steps
@dataclass
class Step:
    kind: str                          # import | analysis | chart | sql | python | note | transform
    text: str = ""                     # note / sql / python body, title for charts
    data: dict = field(default_factory=dict)
    time: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))


class _Fmt(dict):
    """Placeholder resolver for the code templates declared by each analysis."""

    def __init__(self, lang: str, params: dict, roles: dict, names: dict, ctx: dict):
        super().__init__()
        self.lang, self.params, self.roles, self.names, self.ctx = lang, params, roles, names, ctx

    def nm(self, n: str) -> str:
        return {"stata": stata_name, "r": r_name}.get(self.lang, str)(n)

    def role_names(self, key: str) -> list[str]:
        v = self.names.get(key)
        if v is None:
            return []
        return [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]

    def __missing__(self, key: str):
        p, lang = self.params, self.lang
        if key in p and not isinstance(p[key], str):
            return str(p[key])
        if key in p:                                   # plain string parameter (alternative, method, ...)
            return str(p[key])
        if key in self.names:                          # role -> variable name(s)
            ns = self.role_names(key)
            return " ".join(self.nm(n) for n in ns) if lang == "stata" else (", ".join(self.nm(n) for n in ns) if len(ns) > 1 else (self.nm(ns[0]) if ns else ""))
        if key in self.ctx:
            return str(self.ctx[key])
        if key.endswith("_q"):                         # "a", "b"
            ns = self.role_names(key[:-2])
            q = r_str if lang == "r" else py_str
            return ", ".join(q(self.nm(n) if lang != "python" else n) for n in ns)
        if key.endswith("_plus"):
            return " + ".join(self.nm(n) for n in self.role_names(key[:-5]))
        if key.endswith("_stata"):
            base = key[:-6]
            if base == "factors":
                return "".join(f" i.{stata_name(n)}" for n in self.role_names("factors"))
            if base == "robust":
                return "" if p.get("robust", "classical") == "classical" else ", vce(robust)"
        if key == "factors_r":
            return "".join(f" + factor({r_name(n)})" for n in self.role_names("factors"))
        if key == "factors_py":
            return "".join(f" + C(Q({py_str(n)}))" if not n.isidentifier() else f" + C({n})" for n in self.role_names("factors"))
        if key in ("inter_r", "inter_py"):
            return "*" if p.get("interaction", True) else "+"
        if key == "exact_r" and "exact" not in p and "method" in p:     # rank tests: method = auto | exact | asymptotic
            m = str(p["method"]).lower()
            return {"exact": "TRUE" if lang == "r" else "True", "asymptotic": "FALSE" if lang == "r" else "False"}.get(
                m, "NULL" if lang == "r" else "None")
        if key == "ss_num":
            return {"I": "1", "II": "2", "III": "3"}.get(str(p.get("ss_type", "II")), "2")
        if key == "conf_pct":
            return str(int(round(float(p.get("conf", 0.95)) * 100)))
        if key == "break_at_r":
            return str(int(p.get("break_at", 0)) or max(self.ctx.get("n", 2) // 2, 1))
        if key == "probs_r":
            raw = str(p.get("probs", "")).strip()
            if raw:
                return raw
            k = int(self.ctx.get("k", 0)) or 2
            return ", ".join([f"1/{k}"] * k)
        if key.endswith("_r") and key[:-2] in p:       # parameter literal in the target language
            return _lit(lang, p[key[:-2]])
        if key.endswith("_r") and key[:-2] in self.names:  # role name in R-safe/py form
            ns = self.role_names(key[:-2])
            return self.nm(ns[0]) if ns else ""
        return "{" + key + "}"                         # leave unknown placeholders visible


CHART_CODE = {   # id -> {lang: template}; others fall back to a descriptive comment + exported image
    "histogram": {"stata": "histogram {x}, frequency", "r": "hist(df${x_r}, main = \"{x}\")", "python": "df[{x_q}].hist(bins='auto')"},
    "scatter": {"stata": "twoway scatter {y} {x}", "r": "plot(df${x_r}, df${y_r})", "python": "df.plot.scatter(x={x_q}, y={y_q})"},
    "box": {"stata": "graph box {y}", "r": "boxplot(df${y_r})", "python": "df.boxplot(column={y_q})"},
    "bar": {"stata": "graph hbar (count), over({x})", "r": "barplot(table(df${x_r}), horiz = TRUE)", "python": "df[{x_q}].value_counts().plot.barh()"},
    "column": {"stata": "graph bar (count), over({x})", "r": "barplot(table(df${x_r}))", "python": "df[{x_q}].value_counts().plot.bar()"},
    "pie": {"stata": "graph pie, over({x})", "r": "pie(table(df${x_r}))", "python": "df[{x_q}].value_counts().plot.pie()"},
    "line": {"stata": "twoway line {y} {x}", "r": "plot(df${x_r}, df${y_r}, type = \"l\")", "python": "df.plot.line(x={x_q}, y={y_q})"},
    "heatmap": {"stata": "* correlation heat map: see R/Python", "r": "heatmap(cor(df[, c({dims_q})], use = \"pairwise.complete.obs\"))",
                "python": "import seaborn as sns; sns.heatmap(df[[{dims_q}]].corr(), annot=True)"},
}


@dataclass
class Recorder:
    seed: int = 12345
    steps: list[Step] = field(default_factory=list)
    project: str = "Statistic Smith session"

    # ---- recording API ---------------------------------------------------------------------------
    def import_file(self, path: str, sheet: str | None = None, dataset: str = "df", **opts):
        self.steps.append(Step("import", path, {"sheet": sheet, "dataset": dataset, **opts}))

    def analysis(self, aid: str, cols_names: dict, params: dict | None = None, ctx: dict | None = None):
        self.steps.append(Step("analysis", aid, {"names": cols_names, "params": params or {}, "ctx": ctx or {}}))

    def chart(self, cid: str, cols_names: dict, title: str = "", params: dict | None = None, image: str = ""):
        self.steps.append(Step("chart", title or cid, {"id": cid, "names": cols_names, "params": params or {}, "image": image}))

    def sql(self, query: str):
        self.steps.append(Step("sql", query))

    def python(self, code: str):
        self.steps.append(Step("python", code))

    def note(self, text: str):
        self.steps.append(Step("note", text))

    def transform(self, description: str, stata: str = "", r: str = "", python: str = ""):
        self.steps.append(Step("transform", description, {"stata": stata, "r": r, "python": python}))

    def clear(self):
        self.steps.clear()

    def undo_last(self):
        if self.steps:
            self.steps.pop()

    # ---- rendering -------------------------------------------------------------------------------
    def render(self, lang: str = "stata") -> str:
        from ..stats import REGISTRY
        c = {"stata": "*", "r": "#", "python": "#"}[lang]
        out = [f"{c} {'=' * 70}", f"{c}  {self.project}", f"{c}  Generated by Statistic Smith {__version__} on "
               f"{_dt.datetime.now():%Y-%m-%d %H:%M}", f"{c}  Re-run this file in {'Stata' if lang == 'stata' else 'R' if lang == 'r' else 'Python'} "
               f"to reproduce / verify every result shown in the app.", f"{c} {'=' * 70}", ""]
        if lang == "stata":
            out += ["clear all", "set more off", f"set seed {self.seed}", ""]
        elif lang == "r":
            out += [f"set.seed({self.seed})", "options(width = 100)", ""]
        else:
            out += ["import numpy as np", "import pandas as pd", "import scipy.stats as st", f"np.random.seed({self.seed})", ""]
        for s in self.steps:
            out += self._render_step(s, lang, REGISTRY, c)
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    def _render_step(self, s: Step, lang: str, REG, c: str) -> list[str]:
        d = s.data
        if s.kind == "note":
            return [f"{c} {ln}" for ln in s.text.splitlines()]
        if s.kind == "import":
            return self._render_import(s, lang, c)
        if s.kind == "sql":
            body = s.text.strip().rstrip(";")
            if lang == "python":
                return [f"{c} SQL run in the Statistic Smith terminal", f"sql_result = con.execute({body!r}).df()   # DuckDB connection `con`"]
            if lang == "r":
                return [f"{c} SQL run in the Statistic Smith terminal", f"# library(sqldf); sqldf({r_str(body)})"]
            return [f"{c} SQL run in the Statistic Smith terminal (no Stata equivalent):"] + [f"*   {ln}" for ln in body.splitlines()]
        if s.kind == "python":
            return [f"{c} Python run in the Statistic Smith terminal"] + ([ln for ln in s.text.splitlines()] if lang == "python"
                                                                         else [f"{c}   {ln}" for ln in s.text.splitlines()])
        if s.kind == "transform":
            code = d.get(lang) or ""
            return [f"{c} {s.text}"] + ([code] if code else [])
        if s.kind == "chart":
            tpl = CHART_CODE.get(d["id"], {}).get(lang)
            hdr = f"{c} Chart: {s.text}"
            if tpl:
                f = _Fmt(lang, d["params"], {}, d["names"], {})
                code = tpl.format_map(f)
                if lang == "python" and "import matplotlib" not in code:
                    code = "import matplotlib.pyplot as plt\n" + code + "\nplt.show()"
                return [hdr, code]
            return [hdr, f"{c} (no native {lang} one-liner for '{d['id']}'; the app exported the image{': ' + d['image'] if d['image'] else ''})"]
        if s.kind == "analysis":
            spec = REG.get(s.text)
            if not spec:
                return [f"{c} unknown analysis {s.text}"]
            tpl = spec.code.get(lang)
            params = {p.name: p.default for p in spec.params}
            params.update(d["params"])
            lines = [f"{c} {spec.name}"]
            if not tpl:
                return lines + [f"{c} (no {lang} template)"]
            f = _Fmt(lang, params, {}, d["names"], d["ctx"])
            code = tpl.format_map(f)
            return lines + [code]
        return [f"{c} {s.kind}: {s.text}"]

    def _render_import(self, s: Step, lang: str, c: str) -> list[str]:
        p, d = s.text, s.data
        ext = Path(p).suffix.lower()
        win = p.replace("\\", "/")
        if lang == "stata":
            if ext in (".csv", ".tsv", ".txt", ".tab"):
                return [f'import delimited using "{win}", case(preserve) varnames(1) clear']
            if ext in (".xlsx", ".xls", ".xlsm"):
                sh = f' sheet("{d["sheet"]}")' if d.get("sheet") else ""
                return [f'import excel using "{win}",{sh} firstrow clear']
            if ext == ".dta":
                return [f'use "{win}", clear']
            return [f'* import of {Path(p).name} (R / SPSS / SAS / Parquet) has no direct Stata command; export to .dta or .csv first', f'* use "{win}", clear']
        if lang == "r":
            if ext in (".csv", ".tsv", ".txt", ".tab"):
                return ["library(readr)", f'df <- read_delim({r_str(win)}, delim = {r_str(chr(9) if ext in (".tsv", ".tab") else ",")}, show_col_types = FALSE)',
                        "names(df) <- make.names(names(df))   # same renaming the generated analysis code assumes"]
            if ext in (".xlsx", ".xls", ".xlsm"):
                sh = f", sheet = {r_str(d['sheet'])}" if d.get("sheet") else ""
                return ["library(readxl)", f"df <- read_excel({r_str(win)}{sh})", "names(df) <- make.names(names(df))"]
            if ext == ".dta":
                return ["library(haven)", f"df <- read_dta({r_str(win)})", "names(df) <- make.names(names(df))"]
            if ext in (".rds",):
                return [f"df <- readRDS({r_str(win)})"]
            if ext in (".rdata", ".rda"):
                return [f"load({r_str(win)})   # then: df <- <name of the data.frame inside>"]
            if ext == ".parquet":
                return ["library(arrow)", f"df <- read_parquet({r_str(win)})", "names(df) <- make.names(names(df))"]
            return [f"# TODO: read {win}"]
        if ext in (".csv", ".tsv", ".txt", ".tab"):
            sep = ", sep='\\t'" if ext in (".tsv", ".tab") else ""
            return [f"df = pd.read_csv({py_str(win)}{sep})"]
        if ext in (".xlsx", ".xls", ".xlsm"):
            return [f"df = pd.read_excel({py_str(win)}{', sheet_name=' + py_str(d['sheet']) if d.get('sheet') else ''})"]
        if ext == ".dta":
            return [f"df = pd.read_stata({py_str(win)})"]
        if ext == ".parquet":
            return [f"df = pd.read_parquet({py_str(win)})"]
        if ext in (".rds", ".rdata", ".rda"):
            return ["import pyreadr", f"df = next(iter(pyreadr.read_r({py_str(win)}).values()))"]
        return [f"# TODO: read {win}"]

    # ---- persistence ----------------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({"seed": self.seed, "project": self.project, "steps": [asdict(s) for s in self.steps]}, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Recorder":
        raw = json.loads(text)
        r = cls(seed=raw.get("seed", 12345), project=raw.get("project", ""))
        r.steps = [Step(**s) for s in raw.get("steps", [])]
        return r

    def save_script(self, path: str, lang: str):
        Path(path).write_text(self.render(lang), encoding="utf-8")
