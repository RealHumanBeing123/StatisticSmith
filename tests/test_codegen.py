"""The generated .do / R / Python scripts must (a) contain every placeholder resolved and (b) for Python,
actually run and reproduce the app's numbers - that is the whole point of the 'verify with a script' feature."""
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from statsmith.codegen import Recorder, r_name, stata_name
from statsmith.stats import REGISTRY, run


def test_name_sanitisers_match_stata_and_R_rules():
    assert r_name("my var") == "my.var" and r_name("1abc") == "X1abc" and r_name("ok_name.1") == "ok_name.1"
    assert stata_name("my var") == "my_var" and stata_name("1abc") == "_1abc" and len(stata_name("x" * 50)) == 32


def test_every_template_resolves_for_all_languages():
    """Fill every analysis template with plausible names; no `{placeholder}` may be left over."""
    bad = []
    for aid, spec in REGISTRY.items():
        names = {}
        for r in spec.roles:
            names[r.name] = ["v1", "v2", "v3"] if r.multi else ("my var" if r.name in ("x", "y") else r.name.upper())
        for lang in ("stata", "r", "python"):
            rec = Recorder()
            rec.analysis(aid, names, ctx={"k": 3, "n": 100, "a": 5, "b": 6, "c": 7, "d": 8})
            text = rec.render(lang)
            left = re.findall(r"(?<![\w$])\{(\w+)\}", text)
            left = [x for x in left if x not in ("{",)]
            if left:
                bad.append((aid, lang, left))
    assert not bad, bad[:10]


def test_script_header_and_seed_and_import_per_language():
    rec = Recorder(seed=777)
    rec.import_file("C:\\data\\survey.csv")
    rec.analysis("t_two", {"y": "score", "group": "arm"}, {"alternative": "two-sided", "conf": 0.95})
    do, r, py = rec.render("stata"), rec.render("r"), rec.render("python")
    assert "set seed 777" in do and 'import delimited using "C:/data/survey.csv"' in do and "ttest score, by(arm)" in do
    assert "set.seed(777)" in r and "read_delim" in r and "t.test(score ~ arm" in r
    assert "np.random.seed(777)" in py and "pd.read_csv" in py


def test_json_roundtrip():
    rec = Recorder(seed=5)
    rec.note("hello")
    rec.analysis("ols", {"y": "y", "xs": ["a", "b"]})
    again = Recorder.from_json(rec.to_json())
    assert again.render("r") == rec.render("r").replace(rec.render("r").splitlines()[2], again.render("r").splitlines()[2])


@pytest.mark.parametrize("aid,cols,params,keys", [
    ("t_welch", lambda r: {"y": r.normal(0, 1, 40).tolist() + r.normal(.5, 1.4, 35).tolist(), "group": ["a"] * 40 + ["b"] * 35}, {}, ["t", "p"]),
    ("t_one", lambda r: {"x": r.normal(1, 1, 50).tolist()}, {"mu0": 0.5}, ["t", "p"]),
    ("corr_test", lambda r: (lambda x: {"x": x.tolist(), "y": (x * .6 + r.normal(0, 1, 60)).tolist()})(r.normal(0, 1, 60)), {}, ["r", "p"]),
])
def test_generated_python_reproduces_app_numbers(tmp_path, aid, cols, params, keys):
    rng = np.random.default_rng(1)
    data = cols(rng)
    import pandas as pd
    df = pd.DataFrame(data)
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    names = {r.name: r.name for r in REGISTRY[aid].roles if r.name in data}
    app = run(aid, {k: np.array(v, dtype=object if isinstance(v[0], str) else float) for k, v in data.items()}, **params)
    rec = Recorder(seed=1)
    rec.import_file(str(csv))
    rec.analysis(aid, names, params)
    script = rec.render("python") + "\nprint(repr(_))\n" if False else rec.render("python")
    # execute the last template line and capture its value: wrap so the final expression is printed
    lines = script.rstrip().splitlines()
    last = lines[-1]
    body = "\n".join(lines[:-1]) + "\nres = " + last.split(";")[-1].strip() + "\nprint(res)\n"
    body = re.sub(r"^import .*? as .*?; ", "", body, flags=re.M) if False else body
    # templates are '<imports>; expression' one-liners -> turn the leading imports into real statements
    final = []
    for ln in lines[:-1]:
        final.append(ln)
    stmts = [s.strip() for s in last.split(";")]
    final += stmts[:-1] + ["res = " + stmts[-1], "print(res)"]
    p = subprocess.run([sys.executable, "-c", "\n".join(final)], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-800:]
    nums = [float(x) for x in re.findall(r"-?\d+\.\d+(?:e[+-]?\d+)?|-?\d+e[+-]?\d+", p.stdout)]
    for k in keys:
        v = float(app.values[k])
        assert any(abs(v - n) <= 1e-6 * max(1, abs(v)) or abs(abs(v) - abs(n)) <= 1e-6 * max(1, abs(v)) for n in nums), (k, v, p.stdout)
