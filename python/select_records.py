
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "record_label_distribution.csv"
OUTPUT = ROOT / "results" / "selected_records.csv"

df = pd.read_csv(INPUT, dtype={"record_id": str})

classes = ["N", "V", "F"]
counts = df[classes].to_numpy(dtype=float)
n = len(df)

# Decision variables:
# x[i] = 1 if record i is selected
# t = largest selected class count
# d = difference between largest and smallest class count
#
# Objective: minimize imbalance while retaining
# a meaningful number of beats.

# Variables: [x_1 ... x_n, t, d]
objective = np.zeros(n + 2)

# Small penalty per selected record is avoided:
# we want diversity, not artificially few records.
objective[n + 1] = 1.0
objective[n] = 0

integrality = np.zeros(n + 2)
integrality[:n] = 1

lower = np.zeros(n + 2)
upper = np.full(n + 2, np.inf)
upper[:n] = 1

constraints = []

def add_constraint(coeff, lb, ub):
    constraints.append(
        LinearConstraint(
            np.asarray(coeff, dtype=float),
            lb,
            ub
        )
    )


# Exclude records with fewer than 100 N/V/F beats
minimum_beats = 100

for i in range(n):
    if counts[i].sum() < minimum_beats:
        row = np.zeros(n + 2)
        row[i] = 1
        add_constraint(row, 0, 0)

# Require at least 8 records containing PVC beats
row = np.zeros(n + 2)
row[:n] = (counts[:, 1] > 0).astype(float)
add_constraint(row, 8, np.inf)

# Require at least 5 selected records
row = np.zeros(n + 2)
row[:n] = 1
add_constraint(row, 5, np.inf)

# Require at least 500 Fusion beats
row = np.zeros(n + 2)
row[:n] = counts[:, 2]
add_constraint(row, 500, np.inf)

# Require at least 5 records containing Fusion
row = np.zeros(n + 2)
row[:n] = (counts[:, 2] > 0).astype(float)
add_constraint(row, 5, np.inf)

# Require at least 2000 PVC beats
row = np.zeros(n + 2)
row[:n] = counts[:, 1]
add_constraint(row, 2000, np.inf)

# t >= each class count
for j in range(3):
    row = np.zeros(n + 2)
    row[:n] = counts[:, j]
    row[n] = -1
    add_constraint(row, -np.inf, 0)

# d >= count_a - count_b for every class pair
for a in range(3):
    for b in range(3):
        row = np.zeros(n + 2)
        row[:n] = counts[:, a] - counts[:, b]
        row[n + 1] = -1
        add_constraint(row, -np.inf, 0)

result = milp(
    c=objective,
    integrality=integrality,
    bounds=Bounds(lower, upper),
    constraints=constraints,
    options={"time_limit": 60}
)

if result.x is None:
    raise RuntimeError(
        f"Optimization failed: {result.message}"
    )

selected_mask = result.x[:n] > 0.5
selected = df.loc[selected_mask].copy()

print("\nOPTIMIZATION STATUS")
print(result.message)

print("\nSELECTED RECORDS")
print(
    selected[
        ["record_id", "N", "V", "F"]
    ].to_string(index=False)
)

totals = selected[classes].sum()
percentages = totals / totals.sum() * 100

print("\nSELECTED CLASS TOTALS")
print(totals)

print("\nCLASS DISTRIBUTION (%)")
print(percentages.round(2))

print("\nNUMBER OF SELECTED RECORDS:", len(selected))

selected.to_csv(OUTPUT, index=False)
print("\nSaved to:", OUTPUT)
