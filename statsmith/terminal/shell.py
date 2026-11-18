"""Terminal back-end: SQL (DuckDB), Python, and Stata-/R-style commands.

``Terminal.execute(mode, line)`` always returns a :class:`TermResult` (never raises) so the UI just prints it.

* sql     - any DuckDB SQL; tables are the imported datasets. ``CREATE TABLE`` / ``CREATE VIEW`` adds a dataset.
* python  - persistent namespace with ``con`` (DuckDB), ``sql(q)`` -> DataFrame, ``df`` (active dataset as pandas,
            only if it is small enough, else use ``sql`` / ``sample(n)``), ``np pd st sm``, ``run(id, ...)``.
* stata   - a practical subset (see HELP_STATA); every command is executed by the same analysis code as the menus.
* r       - a practical subset (see HELP_R).
"""
from __future__ import annotations

import contextlib
import io
import re
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..codegen import Recorder
from ..data import Engine
from ..runner import draw_chart, run_analysis
from ..stats import REGISTRY as ANALYSES

PY_DF_MAX_ROWS = 5_000_000


@dataclass
class TermResult:
    text: str = ""
    ok: bool = True
    figure_request: tuple | None = None        # (chart_id, selection, params) -> UI draws it
    dataset_changed: str | None = None         # name of a dataset that was created / modified
    switch_dataset: str | None = None


HELP_STATA = """Stata-style commands (subset)
  summarize [vars]                  describe                       count
  list [in 1/10]                    tabulate v1 [v2] [, chi2 exact]
  correlate v1 v2 ...               pwcorr v1 v2 ...  , sig
  regress y x1 x2 ... [, robust]    logit y x1 x2 ...              poisson y x1 ...
  ttest v == 0                      ttest y, by(g)                 ttest a == b       (paired)
  ranksum y, by(g)                  kwallis y, by(g)               signtest y == 0
  swilk v    sfrancia v             sktest v                       oneway y g
  prtest v == 0.5                   bitest v == 0.5   (v is 0/1)   ci means v [, level(99)]
  histogram v     scatter y x       graph box v                    graph bar v     graph pie v
  set seed 123                      use <file>       import delimited <file>      import excel <file>
"""
HELP_R = """R-style commands (subset)
  summary(df$x)      mean(df$x)   sd(df$x)   median(df$x)   var(df$x)   nrow(df)   head(df)   names(df)
  table(df$a)        table(df$a, df$b)
  t.test(df$x, mu = 0)        t.test(y ~ g, data = df)        t.test(df$a, df$b, paired = TRUE)
  wilcox.test(y ~ g, data = df)    kruskal.test(y ~ g, data = df)    shapiro.test(df$x)
  cor(df$x, df$y)    cor.test(df$x, df$y)    chisq.test(table(df$a, df$b))    fisher.test(table(df$a, df$b))
  lm(y ~ a + b, data = df)      glm(y ~ a + b, data = df, family = binomial)
  aov(y ~ g, data = df)         hist(df$x)    plot(df$x, df$y)    boxplot(df$x)
"""
HELP_PY = """Python: persistent session. Available names
  con        DuckDB connection (tables = datasets)       sql("select ...")  -> pandas DataFrame
  df         active dataset as pandas (small datasets)   sample(n)          -> random n rows as DataFrame
  np pd st sm   numpy, pandas, scipy.stats, statsmodels.api
  run("t_welch", y="score", group="arm")   run any registered analysis (see Analysis menu) and print the report
  active     name of the active dataset
"""
HELP_SQL = "SQL: DuckDB dialect. Tables are your datasets (see the Data panel).  e.g.  SELECT g, avg(x) FROM mydata GROUP BY g;\n" \
           "  CREATE TABLE clean AS SELECT * FROM mydata WHERE x IS NOT NULL;   -- creates a new dataset"


class Terminal:
    def __init__(self, engine: Engine, recorder: Recorder | None = None, active: str | None = None):
        self.engine, self.recorder, self.active = engine, recorder, active
        self.ns: dict[str, Any] = {}
        self._init_ns()

    # --------------------------------------------------------------------------- helpers
    def set_active(self, name: str | None):
        self.active = name
        self._init_ns(keep_user=True)

    def _init_ns(self, keep_user: bool = False):
        import pandas as pd
        import scipy.stats as st
        import statsmodels.api as sm
        eng = self

        def sql(q: str):
            return eng.engine.sql(q).fetchdf()

        def sample(n: int = 1000, seed: int = 1):
            return eng.engine.sql(f'SELECT * FROM "{eng.active}" USING SAMPLE reservoir({int(n)} ROWS) REPEATABLE ({int(seed)})').fetchdf()

        def run(aid: str, **sel):
            params = {k: sel.pop(k) for k in list(sel) if k in {p.name for p in ANALYSES[aid].params}}
            res = run_analysis(eng.engine, eng.active, aid, sel, params, eng.recorder)
            print(res.to_text())
            return res.values

        base = {"np": np, "pd": pd, "st": st, "sm": sm, "con": self.engine.con, "sql": sql, "sample": sample,
                "run": run, "active": self.active}
        if self.active and self.active in self.engine.datasets and self.engine.datasets[self.active].n_rows <= PY_DF_MAX_ROWS:
            base["df"] = self.engine.sql(f'SELECT * FROM "{self.active}"').fetchdf()
        else:
            base.pop("df", None)
        user = {k: v for k, v in self.ns.items() if keep_user and not k.startswith("__") and k not in base}
        self.ns = {"__name__": "__statsmith__", **base, **user}

    def _resolve(self, name: str) -> str:
        ds = self.engine.datasets[self.active]
        names = ds.names
        if name in names:
            return name
        low = {n.lower(): n for n in names}
        if name.lower() in low:
            return low[name.lower()]
        from ..codegen import r_name, stata_name
        for n in names:
            if name in (stata_name(n), r_name(n)):
                return n
        raise KeyError(f"variable {name!r} not found")

    def _vars(self, text: str) -> list[str]:
        toks = text.replace(",", " ").split()
        if not toks:
            return [c.name for c in self.engine.datasets[self.active].columns]
        return [self._resolve(t.split("$")[-1]) for t in toks]

    def _out(self, aid: str, sel: dict, params: dict | None = None) -> TermResult:
        res = run_analysis(self.engine, self.active, aid, sel, params, self.recorder)
        return TermResult(res.to_text())

    # --------------------------------------------------------------------------- entry
    def execute(self, mode: str, line: str) -> TermResult:
        line = line.strip()
        if not line:
            return TermResult()
        if line.lower() in ("help", "?", "help()"):
            return TermResult({"sql": HELP_SQL, "python": HELP_PY, "stata": HELP_STATA, "r": HELP_R}[mode])
        try:
            if mode == "sql":
                return self._sql(line)
            if mode == "python":
                return self._python(line)
            if mode == "stata":
                return self._stata(line)
            if mode == "r":
                return self._r(line)
            return TermResult(f"unknown mode {mode}", ok=False)
        except Exception as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            return TermResult(f"error: {msg}", ok=False)

    # --------------------------------------------------------------------------- SQL
    def _sql(self, q: str) -> TermResult:
        before = set(self.engine.datasets)
        stmt = q.strip().rstrip(";")
        m = re.match(r"(?is)^\s*create\s+(?:or\s+replace\s+)?(table|view)\s+(\"?[\w ]+\"?)\s+as\b", stmt)
        cur = self.engine.sql(stmt)
        if self.recorder:
            self.recorder.sql(stmt)
        if m:
            name = m.group(2).strip('"')
            self.engine.refresh(name) if name in self.engine.datasets else self.engine.datasets.__setitem__(name, self.engine._describe(name, "sql"))
            return TermResult(f"OK - dataset '{name}' created ({self.engine.datasets[name].n_rows:,} rows)", dataset_changed=name)
        if re.match(r"(?is)^\s*(insert|update|delete|alter|drop)\b", stmt):
            for n in list(self.engine.datasets):
                try:
                    self.engine.refresh(n)
                except Exception:
                    self.engine.datasets.pop(n, None)
            return TermResult("OK", dataset_changed=self.active)
        try:
            df = cur.fetchdf()
        except Exception:
            return TermResult("OK")
        total = len(df)
        import pandas as pd
        with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 200):
            s = df.head(200).to_string(index=False)
        return TermResult(s + (f"\n... ({total:,} rows, first 200 shown)" if total > 200 else f"\n({total:,} rows)"))

    # --------------------------------------------------------------------------- Python
    def _python(self, code: str) -> TermResult:
        buf = io.StringIO()
        self.ns["active"] = self.active
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    val = eval(compile(code, "<terminal>", "eval"), self.ns)
                    if val is not None:
                        print(val.to_string(max_rows=100) if hasattr(val, "to_string") else repr(val.item() if isinstance(val, np.generic) else val))
                except SyntaxError:
                    exec(compile(code, "<terminal>", "exec"), self.ns)
        except Exception:
            tb = traceback.format_exc().strip().splitlines()
            return TermResult(buf.getvalue() + "\n".join(tb[-3:]), ok=False)
        if self.recorder:
            self.recorder.python(code)
        return TermResult(buf.getvalue().rstrip())

    # --------------------------------------------------------------------------- Stata subset
    def _stata(self, line: str) -> TermResult:
        line = re.sub(r"^\s*\.\s*", "", line)
        m = re.match(r"^\s*(\w+)\s*(.*?)\s*$", line)
        cmd, rest = m.group(1).lower(), m.group(2)
        body, _, opts = rest.partition(",")
        body, opts = body.strip(), opts.strip()
        eng = self.engine
        if cmd == "set" and body.lower().startswith("seed"):
            seed = int(body.split()[1])
            if self.recorder:
                self.recorder.seed = seed
            return TermResult(f"seed set to {seed}")
        if cmd == "use" or (cmd == "import" and body):
            path = body.split(None, 1)[-1] if cmd == "import" else body
            path = re.sub(r"^(delimited|excel)\s+(using\s+)?", "", path).strip().strip('"')
            ds = eng.import_file(path)
            if self.recorder:
                self.recorder.import_file(path)
            return TermResult(f"{ds.n_rows:,} observations, {len(ds.columns)} variables", dataset_changed=ds.name, switch_dataset=ds.name)
        need_data = self.active in eng.datasets
        if not need_data:
            return TermResult("no dataset is open - File > Open, or: use <file>", ok=False)
        ds = eng.datasets[self.active]
        if cmd in ("summarize", "sum", "summ"):
            vs = [v for v in self._vars(body) if ds.col(v).is_numeric]
            return self._out("summary", {"xs": vs})
        if cmd in ("describe", "des"):
            return TermResult("\n".join([f"Observations: {ds.n_rows:,}   Variables: {len(ds.columns)}"] +
                                        [f"  {c.name:<28}{c.sql_type:<14}{c.label}" for c in ds.columns]))
        if cmd == "count":
            return TermResult(f"{eng.count(self.active):,}")
        if cmd == "list":
            m2 = re.search(r"in\s+(\d+)\s*/\s*(\d+)", body)
            a, b = (int(m2.group(1)), int(m2.group(2))) if m2 else (1, 10)
            cols = [c.name for c in ds.columns[:12]]
            rows = eng.fetch(self.active, a - 1, b - a + 1, cols)
            import pandas as pd
            return TermResult(pd.DataFrame(rows, columns=cols).to_string())
        if cmd in ("tabulate", "tab", "tab1", "tab2"):
            vs = self._vars(body)
            if len(vs) == 1:
                return self._out("frequency", {"x": vs[0]})
            if len(vs) == 2:
                return self._out("fisher" if "exact" in opts else "chi2_indep", {"row": vs[0], "col": vs[1]})
        if cmd in ("correlate", "corr", "pwcorr"):
            return self._out("correlation", {"xs": [v for v in self._vars(body) if ds.col(v).is_numeric]})
        if cmd in ("regress", "reg"):
            vs = self._vars(body)
            return self._out("ols", {"y": vs[0], "xs": vs[1:]}, {"robust": "HC1" if "robust" in opts or "vce(robust)" in opts else "classical"})
        if cmd == "logit":
            vs = self._vars(body)
            return self._out("logit", {"y": vs[0], "xs": vs[1:]}, {"success": "1"})
        if cmd == "poisson":
            vs = self._vars(body)
            return self._out("poisson", {"y": vs[0], "xs": vs[1:]})
        if cmd == "ttest":
            by = re.search(r"by\((\w+)\)", opts)
            if by:
                return self._out("t_two", {"y": self._vars(body)[0], "group": self._resolve(by.group(1))})
            m3 = re.match(r"^(\w+)\s*==\s*(\S+)$", body)
            if m3:
                try:
                    return self._out("t_one", {"x": self._resolve(m3.group(1))}, {"mu0": float(m3.group(2))})
                except ValueError:
                    return self._out("t_paired", {"x": self._resolve(m3.group(1)), "y": self._resolve(m3.group(2))})
        if cmd == "ranksum":
            by = re.search(r"by\((\w+)\)", opts)
            return self._out("mann_whitney", {"y": self._vars(body)[0], "group": self._resolve(by.group(1))})
        if cmd == "kwallis":
            by = re.search(r"by\((\w+)\)", opts)
            return self._out("kruskal", {"y": self._vars(body)[0], "group": self._resolve(by.group(1))})
        if cmd == "oneway":
            vs = self._vars(body)
            return self._out("anova_one", {"y": vs[0], "group": vs[1]})
        if cmd == "signtest":
            m3 = re.match(r"^(\w+)\s*==\s*(\S+)$", body)
            return self._out("sign_test", {"x": self._resolve(m3.group(1))}, {"m0": float(m3.group(2))})
        if cmd == "swilk":
            return self._out("shapiro_wilk", {"x": self._vars(body)[0]})
        if cmd == "sfrancia":
            return self._out("shapiro_francia", {"x": self._vars(body)[0]})
        if cmd == "sktest":
            return self._out("dagostino", {"x": self._vars(body)[0]})
        if cmd in ("prtest", "bitest"):
            m3 = re.match(r"^(\w+)\s*==\s*(\S+)$", body)
            return self._out("prop_one" if cmd == "prtest" else "binomial", {"x": self._resolve(m3.group(1))}, {"p0": float(m3.group(2)), "success": "1"})
        if cmd == "ci":
            lv = re.search(r"level\((\d+)\)", opts)
            vs = self._vars(re.sub(r"^(means|mean)\s+", "", body))
            return self._out("ci_mean", {"x": vs[0]}, {"conf": int(lv.group(1)) / 100 if lv else 0.95})
        if cmd == "histogram":
            return TermResult("", figure_request=("histogram", {"x": self._vars(body)[0]}, {}))
        if cmd in ("scatter", "twoway"):
            vs = self._vars(re.sub(r"^scatter\s+", "", body))
            return TermResult("", figure_request=("scatter", {"y": vs[0], "x": vs[1]}, {}))
        if cmd == "graph":
            kind, _, v = body.partition(" ")
            v = v.strip()
            cid = {"box": "box", "bar": "column", "hbar": "bar", "pie": "pie"}.get(kind)
            if cid:
                sel = {"y": self._vars(v)[0]} if cid == "box" else {"x": self._vars(re.sub(r"^\(\w+\)\s*", "", v))[0]}
                return TermResult("", figure_request=(cid, sel, {}))
        return TermResult(f"command not supported: {cmd}   (type help)", ok=False)

    # --------------------------------------------------------------------------- R subset
    def _r(self, line: str) -> TermResult:
        eng = self.engine
        m = re.match(r"^\s*(?:\w[\w.]*\s*(?:<-|=)\s*)?([\w.]+)\s*\((.*)\)\s*$", line, re.S)
        if not m:
            return TermResult("R: only function calls like t.test(df$x, mu = 0) are supported (type help)", ok=False)
        fn, args = m.group(1), m.group(2).strip()
        if self.active not in eng.datasets:
            return TermResult("no dataset is open", ok=False)
        ds = eng.datasets[self.active]

        def cols_in(s: str) -> list[str]:
            return [self._resolve(c) for c in re.findall(r"df\$([\w.]+)|df\[\[\"([^\"]+)\"\]\]", s) and
                    [a or b for a, b in re.findall(r"df\$([\w.]+)|df\[\[\"([^\"]+)\"\]\]", s)]]

        def formula(s: str):
            f = re.match(r"^\s*([\w.]+)\s*~\s*([^,]+)", s)
            return (self._resolve(f.group(1)), [self._resolve(t.strip()) for t in re.split(r"\+", f.group(2))]) if f else None

        def opt(name: str, default=None):
            o = re.search(rf"\b{re.escape(name)}\s*=\s*([^,)]+)", args)
            return o.group(1).strip().strip('"') if o else default

        if fn == "nrow":
            return TermResult(f"[1] {ds.n_rows:,}")
        if fn == "names":
            return TermResult(" ".join(ds.names))
        if fn == "head":
            rows = eng.fetch(self.active, 0, 6, [c.name for c in ds.columns[:12]])
            import pandas as pd
            return TermResult(pd.DataFrame(rows, columns=[c.name for c in ds.columns[:12]]).to_string())
        if fn in ("mean", "sd", "median", "var", "summary", "min", "max", "sum"):
            cs = cols_in(args)
            if not cs:
                return TermResult("use df$column", ok=False)
            x = eng.column(self.active, cs[0], numeric=True)
            x = x[np.isfinite(x)]
            if fn == "summary":
                q = np.percentile(x, [0, 25, 50, 75, 100])
                return TermResult(f"   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.\n" + " ".join(f"{v:8.4g}" for v in (q[0], q[1], q[2], x.mean(), q[3], q[4])))
            v = {"mean": np.mean, "median": np.median, "min": np.min, "max": np.max, "sum": np.sum,
                 "sd": lambda a: np.std(a, ddof=1), "var": lambda a: np.var(a, ddof=1)}[fn](x)
            return TermResult(f"[1] {v:.10g}")
        if fn == "table":
            cs = cols_in(args)
            return self._out("frequency", {"x": cs[0]}) if len(cs) == 1 else self._out("chi2_indep", {"row": cs[0], "col": cs[1]})
        if fn == "t.test":
            f = formula(args)
            alt = opt("alternative", "two-sided")
            if f:
                return self._out("t_welch" if "var.equal = TRUE" not in args and "var.equal=TRUE" not in args else "t_two",
                                 {"y": f[0], "group": f[1][0]}, {"alternative": alt})
            cs = cols_in(args)
            if "paired" in args and "TRUE" in args.split("paired")[1][:12]:
                return self._out("t_paired", {"x": cs[0], "y": cs[1]})
            return self._out("t_one", {"x": cs[0]}, {"mu0": float(opt("mu", 0)), "alternative": alt})
        if fn in ("wilcox.test", "kruskal.test"):
            f = formula(args)
            return self._out("mann_whitney" if fn == "wilcox.test" else "kruskal", {"y": f[0], "group": f[1][0]})
        if fn == "shapiro.test":
            return self._out("shapiro_wilk", {"x": cols_in(args)[0]})
        if fn in ("cor", "cor.test"):
            cs = cols_in(args)
            meth = opt("method", "pearson")
            if fn == "cor.test":
                return self._out("corr_test", {"x": cs[0], "y": cs[1]}, {"method": meth})
            return self._out("correlation", {"xs": cs}, {"method": meth})
        if fn in ("chisq.test", "fisher.test"):
            cs = cols_in(args)
            return self._out("chi2_indep" if fn == "chisq.test" else "fisher", {"row": cs[0], "col": cs[1]})
        if fn in ("lm", "glm", "aov"):
            f = formula(args)
            if fn == "aov":
                return self._out("anova_one", {"y": f[0], "group": f[1][0]})
            nums = [v for v in f[1] if ds.col(v).is_numeric]
            facs = [v for v in f[1] if not ds.col(v).is_numeric]
            sel = {"y": f[0], "xs": nums}
            if facs:
                sel["factors"] = facs
            if fn == "glm" and "binomial" in args:
                return self._out("logit", {"y": f[0], "xs": nums}, {"success": "1"})
            if fn == "glm" and "poisson" in args:
                return self._out("poisson", sel)
            return self._out("ols", sel)
        if fn == "hist":
            return TermResult("", figure_request=("histogram", {"x": cols_in(args)[0]}, {}))
        if fn == "boxplot":
            return TermResult("", figure_request=("box", {"y": cols_in(args)[0]}, {}))
        if fn == "plot":
            cs = cols_in(args)
            return TermResult("", figure_request=("scatter", {"x": cs[0], "y": cs[1]}, {}))
        return TermResult(f"R function not supported: {fn}   (type help)", ok=False)
