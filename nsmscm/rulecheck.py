"""Checks of the rules R1-R3 in the DGP with binned conditional means (prompt Section 7).

Monotonicity check: within each stratum, the means of consecutive bins of the ordering variable are
compared; a violation is a step against the rule's direction larger than TOL standard errors of the
difference. Bins with fewer than MIN_N projects are skipped.

R2 is checked along the data manifold: at fixed CPI (CPI vigintile), P(A2 = 1) must not increase with
pi (and therefore with b = pi/CPI). Same definition as the R2 penalty of M4.
"""
from __future__ import annotations

import numpy as np

TOL = 3.0
MIN_N = 100


def qbin(x, q):
    edges = np.quantile(x, np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, x, side="right")


def _strata_means(y, strata, order):
    """dict stratum -> list of (bin, mean, se, n) sorted by bin."""
    k = np.column_stack(strata + [order]) if strata else order[:, None]
    uniq, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.ravel()
    out = {}
    for i, u in enumerate(uniq):
        v = y[inv == i]
        if len(v) < MIN_N:
            continue
        out.setdefault(tuple(int(x) for x in u[:-1]), []).append(
            (int(u[-1]), float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), len(v)))
    return {s: sorted(b) for s, b in out.items()}


def monotone(y, strata, order, direction):
    """direction +1: mean must not decrease along order; -1: must not increase.
    Returns dict(comparisons, violations, worst_z, list of violating steps)."""
    res = {"comparisons": 0, "violations": 0, "worst_z": 0.0, "steps_against": 0, "examples": []}
    for s, bins in _strata_means(y, strata, order).items():
        for (b0, m0, s0, _), (b1, m1, s1, _) in zip(bins[:-1], bins[1:]):
            z = direction * (m1 - m0) / np.sqrt(s0 ** 2 + s1 ** 2)
            res["comparisons"] += 1
            res["steps_against"] += int(z < 0)
            res["worst_z"] = min(res["worst_z"], float(z))
            if z < -TOL:
                res["violations"] += 1
                if len(res["examples"]) < 5:
                    res["examples"].append({"stratum": s, "bins": [b0, b1], "means": [m0, m1], "z": float(z)})
    return res


def level(y, strata, sign):
    """sign +1: every bin mean must be >= 0 (violation: mean + TOL se < 0)."""
    res = {"bins": 0, "violations": 0, "min_mean": np.inf, "min_upper": np.inf}
    for s, bins in _strata_means(y, strata[:-1], strata[-1]).items():
        for _, m, se, _ in bins:
            res["bins"] += 1
            res["min_mean"] = min(res["min_mean"], m)
            res["min_upper"] = min(res["min_upper"], m + TOL * se)
            res["violations"] += int(sign * (m + TOL * se) < 0)
    return res


def fe_slope(y, x, groups):
    """Slope of y on x with group fixed effects (within estimator) and its HC1 t-statistic."""
    yd, xd = y.astype(float).copy(), x.astype(float).copy()
    for g in np.unique(groups):
        m = groups == g
        yd[m] -= yd[m].mean()
        xd[m] -= xd[m].mean()
    b = (xd @ yd) / (xd @ xd)
    e = yd - b * xd
    n = len(y)
    se = np.sqrt((xd ** 2 * e ** 2).sum() / (xd @ xd) ** 2 * n / (n - 1 - len(np.unique(groups))))
    return float(b), float(b / se)


def r1_interventional(z):
    """Y under do(all A = 0): by novelty decile and by size decile."""
    out = {}
    for name in ("novelty", "size"):
        d = qbin(z[name], 10)
        means = [float(z["Y"][d == k].mean()) for k in range(10)]
        ses = [float(z["Y"][d == k].std(ddof=1) / np.sqrt((d == k).sum())) for k in range(10)]
        out[name] = {"decile_means": means, "decile_se": ses,
                     "monotone_increasing": monotone(z["Y"], [], d, +1),
                     "top_minus_bottom_z": (means[-1] - means[0]) / np.hypot(ses[-1], ses[0]),
                     "max_minus_last_z": (max(means) - means[-1]) / np.hypot(ses[int(np.argmax(means))], ses[-1])}
        top = qbin(z[name], 20) == 19
        out[name]["top_vigintile_mean"] = float(z["Y"][top].mean())
        out[name]["top_vigintile_z"] = float(z["Y"][top].mean() / (z["Y"][top].std(ddof=1) / np.sqrt(top.sum())))
    return out


def r1_conditional(o):
    """l_j0(s) >= 0 and increasing in novelty, among A_j = 0, strata mode x CPI quintile x novelty decile."""
    out = {}
    cq, nq = qbin(o["CPI"], 5), qbin(o["novelty"], 10)
    mode = o["mode"].astype(int)
    for j in range(3):
        m = o["A"][:, j] == 0
        y = o["Y"][m]
        out[f"A{j + 1}"] = {"level": level(y, [mode[m], cq[m], nq[m]], +1),
                            "novelty_increasing": monotone(y, [mode[m], cq[m]], nq[m], +1)}
    return out


def r2(o):
    """At fixed CPI (vigintile): P(A2) along pi (quintiles within the CPI bin) and the FE slope on pi."""
    cv = qbin(o["CPI"], 20)
    pq = np.zeros(len(cv), dtype=int)
    for g in range(20):
        m = cv == g
        pq[m] = qbin(o["gate_pi"][m], 5)
    a2 = o["A"][:, 1]
    slope, t = fe_slope(a2, o["gate_pi"], cv)
    slope_p, t_p = fe_slope(o["P"][:, 1], o["gate_pi"], cv)
    return {"decreasing_in_pi": monotone(a2, [cv], pq, -1), "increasing_in_pi": monotone(a2, [cv], pq, +1),
            "fe_slope_A2_on_pi": slope, "fe_t": t, "fe_slope_true_propensity_on_pi": slope_p, "fe_t_true_propensity": t_p}


def r3(o):
    """l_ja decreasing in CPI: among A_j = a, strata mode x novelty tercile x pi tercile x size tercile,
    CPI quintiles."""
    out = {}
    mode = o["mode"].astype(int)
    st = [mode, qbin(o["novelty"], 3), qbin(o["gate_pi"], 3), qbin(o["size"], 3)]
    for j in range(3):
        for a in (0, 1):
            m = o["A"][:, j] == a
            cq = qbin(o["CPI"][m], 5)
            out[f"A{j + 1}={a}"] = monotone(o["Y"][m], [s[m] for s in st], cq, -1)
    return out


def all_checks(obs, zero) -> dict:
    return {"R1_interventional": r1_interventional(zero), "R1_conditional": r1_conditional(obs),
            "R2": r2(obs), "R3": r3(obs)}
