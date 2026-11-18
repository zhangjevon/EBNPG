"""Replay exact primal/dual witnesses without importing the candidate solvers.

Usage: python replay_adjudication.py adjudicated.csv witnesses.json [report.json] [qp_export]
Only Python's standard library is required.
"""
import csv
import hashlib
import json
import math
import sys
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_float(text):
    return F.from_float(float.fromhex(text))


def vector(texts):
    return [exact_float(value) for value in texts]


def matrix(texts):
    return tuple(tuple(vector(row)) for row in texts)


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F(0))


@lru_cache(maxsize=128)
def factor(A):
    """A separate exact LU implementation for independent witness replay."""
    size = len(A)
    table = [list(row) for row in A]
    order = list(range(size))
    for k in range(size):
        pivot = max(range(k, size), key=lambda i: abs(table[i][k]))
        require(table[pivot][k] != 0, "singular support matrix")
        table[k], table[pivot] = table[pivot], table[k]
        order[k], order[pivot] = order[pivot], order[k]
        for i in range(k + 1, size):
            table[i][k] /= table[k][k]
            for j in range(k + 1, size):
                table[i][j] -= table[i][k] * table[k][j]
    return tuple(tuple(row) for row in table), tuple(order)


def solve(A, b):
    lu, order = factor(A)
    size = len(A)
    result = [b[i] for i in order]
    for i in range(size):
        result[i] -= sum((lu[i][j] * result[j] for j in range(i)), F(0))
    for i in range(size - 1, -1, -1):
        result[i] = (
            result[i] - sum((lu[i][j] * result[j] for j in range(i + 1, size)), F(0))
        ) / lu[i][i]
    return result


def transposed(A):
    return tuple(tuple(row) for row in zip(*A))


def root_enclosure(number, digits=65):
    require(number >= 0, "negative square")
    scale = 10 ** digits
    scaled = number.numerator * scale * scale
    lower = math.isqrt(scaled // number.denominator)
    lo = F(lower, scale)
    hi = lo if lower * lower * number.denominator == scaled else F(lower + 1, scale)
    return lo, hi


def squared_sum_at_most(a2, b2, bound):
    """Whether sqrt(a2)+sqrt(b2) <= bound, with exact rational comparisons."""
    if bound < 0:
        return False
    remainder = bound * bound - a2 - b2
    return remainder >= 0 and remainder * remainder >= 4 * a2 * b2


def squared_difference_at_least(a2, b2, bound):
    """Whether sqrt(a2)-sqrt(b2) >= nonnegative bound."""
    if bound <= 0:
        return True
    remainder = a2 - b2 - bound * bound
    return remainder >= 0 and remainder * remainder >= 4 * b2 * bound * bound


def replay_record(record, row):
    key = record["prob"], record["alg"], int(record["run"])
    data = record["model"]
    x = vector(data["x_hex"])
    gradients = [vector(values) for values in data["gradients_hex"]]
    matrices = [matrix(values) for values in data["B_hex"]]
    delta = vector(data["delta_hex"])
    lower, upper = vector(data["lower_hex"]), vector(data["upper_hex"])
    n, m = len(x), len(gradients)
    require(len(matrices) == m and len(delta) == m, "model dimension mismatch")
    require(n > 0 and m > 0 and all(len(row) == n for row in gradients),
            "gradient dimension mismatch")
    require(all(len(A) == n and all(len(row) == n for row in A) for A in matrices),
            "support matrix dimension mismatch")
    require(all(value >= 0 for value in delta), "negative support threshold")
    require(len(lower) == len(upper) == n and all(lo <= hi for lo, hi in zip(lower, upper)),
            "invalid box bounds")

    def support(i, location):
        return delta[i] * sum(map(abs, solve(transposed(matrices[i]), location)), F(0))

    H = [support(i, x) for i in range(m)]
    require(H == list(map(F, data["H_fraction"])), "support H does not replay")
    direction = list(map(F, record["primal_direction"]))
    require(len(direction) == n, "direction shape mismatch")
    location = [xi + di for xi, di in zip(x, direction)]
    require(all(lo <= yi <= hi for lo, yi, hi in zip(lower, location, upper)),
            "primal point outside box")
    U = max(
        dot(gradients[i], direction) + support(i, location) - H[i]
        for i in range(m)
    ) + dot(direction, direction) / 2

    weights = list(map(F, record["lam"]))
    W = [list(map(F, values)) for values in record["W"]]
    require(len(weights) == m and len(W) == n and all(len(row) == m for row in W),
            "dual shape mismatch")
    require(all(value >= 0 for value in weights) and sum(weights, F(0)) == 1,
            "dual weights outside simplex")
    require(all(abs(value) <= 1 for row_values in W for value in row_values),
            "dual support coefficient outside cube")
    coefficient, constant = [F(0)] * n, F(0)
    for i in range(m):
        q = [delta[i] * value for value in solve(matrices[i], [W[j][i] for j in range(n)])]
        coefficient = [
            a + weights[i] * (g + qi)
            for a, g, qi in zip(coefficient, gradients[i], q)
        ]
        constant += weights[i] * (dot(q, x) - H[i])
    inner = [
        min(max(-a, lo - xi), hi - xi)
        for a, lo, hi, xi in zip(coefficient, lower, upper, x)
    ]
    L = constant + dot(coefficient, inner) + dot(inner, inner) / 2
    require(U == F(record["U_fraction"]), "upper value does not replay")
    require(L == F(record["L_fraction"]), "lower value does not replay")
    require(U >= L, "invalid exact bracket")
    require(U - L == F(record["gap_fraction"]), "gap does not replay")

    trial_lower = None
    if "trial" in record:
        trial = record["trial"]
        trial_direction = list(map(F, trial["direction"]))
        trial_point = [xi + di for xi, di in zip(x, trial_direction)]
        require(all(lo <= yi <= hi for lo, yi, hi in zip(lower, trial_point, upper)),
                "trial point outside box")
        trial_U = max(
            dot(gradients[i], trial_direction) + support(i, trial_point) - H[i]
            for i in range(m)
        ) + dot(trial_direction, trial_direction) / 2
        require(trial_U == F(trial["U_fraction"]), "trial value does not replay")
        C = F(trial["lipschitz_upper_fraction"])
        C_check = F(0)
        for i in range(m):
            _, gradient_upper = root_enclosure(dot(gradients[i], gradients[i]))
            # Rows of B^-T are columns of B^-1. Reconstruct each independently.
            row_sum = F(0)
            for j in range(n):
                basis = [F(int(j == k)) for k in range(n)]
                column = solve(matrices[i], basis)
                _, row_upper = root_enclosure(dot(column, column))
                row_sum += row_upper
            C_check = max(C_check, gradient_upper + delta[i] * row_sum)
        require(C >= C_check and C > 0, "trial Lipschitz constant too small")
        require(trial_U < -max(F(1, 10000), C / 10000), "trial is not a failure certificate")
        trial_lower = -trial_U / C
        require(trial_lower == F(trial["norm_lower_fraction"]), "trial norm bound mismatch")

    origin = record.get("origin")
    if origin:
        require(origin["exact"] is True, "origin metadata is not exact")
        require(all(lo <= 0 <= hi for lo, hi in zip(lower, upper)), "origin outside box")
        i = int(origin["active_objective"]) - 1
        values = [-dot(gradients[j], x) - H[j] for j in range(m)]
        require(values[i] == max(values), "origin objective is not active")
        z = list(map(F, origin["supporting_cube"]))
        require(max(map(abs, z)) <= 1, "origin supporting vector outside cube")
        q = [delta[i] * value for value in solve(matrices[i], z)]
        require(all(-xi + gi + qi == 0 for xi, gi, qi in zip(x, gradients[i], q)),
                "origin stationarity does not replay")

    csv_U, csv_L = F(row["U"]), F(row["L"])
    require(csv_U >= U and csv_L <= L, "CSV objective endpoints round inward")
    require(F(row["neg_theta_lo"]) <= -U and F(row["neg_theta_hi"]) >= -L,
            "CSV criticality endpoints round inward")
    norm2, radius2 = dot(direction, direction), 2 * (U - L)
    require(F(row["radius"]) >= 0 and F(row["radius"]) ** 2 >= radius2,
            "CSV radius rounds inward")
    require(F(row["dhat"]) >= 0 and F(row["dhat"]) ** 2 >= norm2,
            "CSV norm display rounds inward")
    require(squared_sum_at_most(norm2, radius2, F(row["dnorm_hi"])),
            "CSV displacement upper endpoint rounds inward")
    dlo = F(row["dnorm_lo"])
    require(dlo >= 0 and (
        squared_difference_at_least(norm2, radius2, dlo)
        or (trial_lower is not None and dlo <= trial_lower)
    ), "CSV displacement lower endpoint rounds inward")
    tol = F(1, 10000)
    criticality = (1 if F(row["neg_theta_hi"]) <= tol
                   else 0 if F(row["neg_theta_lo"]) > tol else -1)
    displacement = (1 if F(row["dnorm_hi"]) <= tol
                    else 0 if F(row["dnorm_lo"]) > tol else -1)
    require(criticality in (0, 1) and displacement in (0, 1), "undecided CSV interval")
    require(criticality == int(row["pass_theta"]) == record["pass_theta"], "criticality mismatch")
    require(displacement == int(row["pass_dnorm"]) == record["pass_dnorm"], "displacement mismatch")
    H_old = vector(data["H_export_hex"])
    H_difference = max(abs(a - b) for a, b in zip(H, H_old))
    return criticality, displacement, bool(origin), bool(record.get("trial")), H_difference


def main():
    if len(sys.argv) not in (3, 4, 5):
        raise SystemExit("Usage: replay_adjudication.py csv witnesses [report.json] [qp_export]")
    csv_path, witness_path = map(Path, sys.argv[1:3])
    report_path = (Path(sys.argv[3]) if len(sys.argv) >= 4
                   else csv_path.with_name("independent_replay_report.json"))
    exports = Path(sys.argv[4]) if len(sys.argv) == 5 else None
    document = json.loads(witness_path.read_text(encoding="utf8"))
    require(document["schema_version"] == 1, "unknown witness schema")
    require(document["certificate"] == "exact-rational-primal-dual", "unknown proof scheme")
    require(hashlib.sha256(csv_path.read_bytes()).hexdigest() == document["csv_sha256"],
            "CSV sha256 mismatch")
    rows = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = row["prob"], row["alg"], int(row["run"])
            require(key not in rows, "duplicate CSV key")
            rows[key] = row
    require(len(rows) == 134, "CSV must contain all 134 disputed points")
    records = document["records"]
    keys = [(row["prob"], row["alg"], int(row["run"])) for row in records]
    require(len(keys) == len(set(keys)) == 134 and set(keys) == set(rows),
            "witness keys do not match CSV")
    export_hashes = {}
    totals = dict(pass_theta=0, pass_dnorm=0, exact_origins=0, trial_proofs=0)
    greatest_H_error = F(0)
    for index, record in enumerate(records, 1):
        key = record["prob"], record["alg"], int(record["run"])
        if exports is not None:
            if key[0] not in export_hashes:
                export_hashes[key[0]] = hashlib.sha256(
                    (exports / (key[0] + ".mat")).read_bytes()
                ).hexdigest()
            require(record["export_sha256"] == export_hashes[key[0]], "export sha256 mismatch")
        ct, ds, origin, trial, h_error = replay_record(record, rows[key])
        totals["pass_theta"] += ct
        totals["pass_dnorm"] += ds
        totals["exact_origins"] += origin
        totals["trial_proofs"] += trial
        greatest_H_error = max(greatest_H_error, h_error)
        if index % 20 == 0:
            print("independently replayed %d/134" % index, flush=True)
    report = dict(
        records=134, undecided=0, csv_sha256=document["csv_sha256"],
        checks="exact LU replay; model feasibility; objective bounds; origin activity/KKT; "
               "trial Lipschitz proof; outward CSV endpoints; threshold flags",
        maximum_exported_H_recomputation_difference_fraction=str(greatest_H_error),
        maximum_exported_H_recomputation_difference=float(greatest_H_error),
        **totals,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

