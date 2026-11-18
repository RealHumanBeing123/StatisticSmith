"""Create sample_data/: a CSV with a codebook, an Excel file, and a Stata .dta with variable/value labels."""
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(__file__).resolve().parents[1] / "sample_data"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)
n = 1000
arm = rng.choice([0, 1], n)
age = rng.integers(18, 80, n)
bmi = rng.normal(27, 4, n)
sbp = 110 + 0.4 * age + 0.8 * bmi - 6 * arm + rng.normal(0, 8, n)
df = pd.DataFrame({"id": np.arange(1, n + 1), "arm": arm, "sex": rng.choice(["F", "M"], n), "age": age, "bmi": bmi.round(1),
                   "sbp": sbp.round(1), "event": (rng.random(n) < 1 / (1 + np.exp(-(age - 60) / 10))).astype(int)})
df.to_csv(out / "trial.csv", index=False)
(out / "trial.codebook.txt").write_text(
    "trial.csv - simulated blood-pressure trial (not real data)\n\nid     participant id\narm    0 = control, 1 = treatment\nsex    F / M\n"
    "age    years\nbmi    body-mass index, kg/m^2\nsbp    systolic blood pressure, mmHg (outcome)\nevent  1 = cardiovascular event during follow-up\n", encoding="utf-8")
df.to_excel(out / "trial.xlsx", index=False)
import pyreadstat
pyreadstat.write_dta(df, str(out / "trial.dta"), column_labels=["Participant id", "Treatment arm", "Sex", "Age (years)", "BMI (kg/m2)",
                     "Systolic BP (mmHg)", "Event during follow-up"], variable_value_labels={"arm": {0: "Control", 1: "Treatment"}, "event": {0: "No", 1: "Yes"}},
                     file_label="Simulated blood-pressure trial")
print("wrote", sorted(p.name for p in out.iterdir()))
