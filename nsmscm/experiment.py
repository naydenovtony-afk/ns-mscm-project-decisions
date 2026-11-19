"""One replication of one cell (scenario, N): data, M1-M5, result rows."""
from __future__ import annotations

import time

import numpy as np

from . import dgp, dml, estimators

SCEN_INDEX = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}


def rep_seed(cfg: dict, scenario: str, n: int, r: int) -> tuple:
    return (cfg["seeds"]["replications"], SCEN_INDEX[scenario], n, r)


def one_rep(cfg: dict, scenario: str, n: int, r: int) -> list[dict]:
    base = rep_seed(cfg, scenario, n, r)
    t0 = time.perf_counter()
    d = dgp.portfolio(cfg, scenario, n, np.random.SeedSequence([*base, 0]))
    X = estimators.covariates(d)
    rows = []
    if scenario == "S2":
        e = dgp.portfolio(cfg, scenario, n, np.random.SeedSequence([*base, 3]), shifted=True)
        rows += dml.dml_all(X, d["A"], d["Y"], cfg, base, estimators.covariates(e), e["A"], e["Y"])
    else:
        t1 = time.perf_counter()
        m1 = estimators.m1_ols(d["Y"], d["A"], X)
        t2 = time.perf_counter()
        m2 = estimators.m2_evm(d["Y"], d["A"], d["CPI"])
        t3 = time.perf_counter()
        for j in range(3):
            rows.append({"method": "M1", "intervention": j + 1, "estimate": m1[j][0], "se": m1[j][1],
                         "runtime_s": round((t2 - t1) / 3, 6)})
            rows.append({"method": "M2", "intervention": j + 1, "estimate": m2[j][0], "se": m2[j][1],
                         "runtime_s": round((t3 - t2) / 3, 6)})
        rows += dml.dml_all(X, d["A"], d["Y"], cfg, base)
    tot = time.perf_counter() - t0
    for row in rows:
        row.update(scenario=scenario, N=n, rep=r, rep_seconds=round(tot, 2))
    return rows
