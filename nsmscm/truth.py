"""Ground truth theta_j = E[Y | do(A_j,tau = 1)] - E[Y | do(A_j,tau = 0)], other interventions at tau
following the policy, by common random numbers on n projects (prompt Section 3). S2: shifted population."""
from __future__ import annotations

import numpy as np

from . import dgp

SCENARIOS = ["S1", "S2", "S3", "S4"]


def scenario_truth(cfg: dict, scenario: str, seed: int, n: int | None = None) -> dict:
    n = n or cfg["truth"]["n_projects"]
    ch = cfg["truth"]["chunk"]
    sh = scenario == "S2"
    run = lambda **kw: dgp.simulate_chunked(cfg, scenario, n, seed, ch, shifted=sh, **kw)  # noqa: E731
    obs = run()
    zero = run(force_all=0.0)
    out = {"scenario": scenario, "seed": seed, "n_projects": n,
           "mean_Y_observed": float(obs["Y"].mean()),
           "mean_Y_all_A0": float(zero["Y"].mean()),
           "treatment_rate": [float(x) for x in obs["A"].mean(0)],
           "propensity_min": [float(x) for x in np.nanmin(obs["P"], 0)],
           "propensity_max": [float(x) for x in np.nanmax(obs["P"], 0)],
           "share_closed_at_tmax": float(obs["closed_at_tmax"].mean()),
           "theta": [], "theta_mc_se": [], "mean_Y_do1": [], "mean_Y_do0": []}
    for j in range(3):
        y1 = run(force=(j, 1.0))["Y"]
        y0 = run(force=(j, 0.0))["Y"]
        dlt = y1 - y0
        out["theta"].append(float(dlt.mean()))
        out["theta_mc_se"].append(float(dlt.std(ddof=1) / np.sqrt(n)))
        out["mean_Y_do1"].append(float(y1.mean()))
        out["mean_Y_do0"].append(float(y0.mean()))
    return out


def summary_stats(cfg: dict, scenario: str, seed: int, n: int) -> dict:
    """Descriptive statistics of an observed portfolio (for Step 2 / the paper's DGP description)."""
    o = dgp.simulate_chunked(cfg, scenario, n, seed, cfg["truth"]["chunk"], shifted=scenario == "S2")
    q = lambda x: [float(v) for v in np.percentile(x, [5, 25, 50, 75, 95])]  # noqa: E731
    return {"scenario": scenario, "n": n,
            "Y_mean": float(o["Y"].mean()), "Y_sd": float(o["Y"].std()), "Y_q05_25_50_75_95": q(o["Y"]),
            "share_Y_positive": float((o["Y"] > 0).mean()),
            "duration_months_mean": float(o["months"].mean()), "planned_months_mean": float(o["D"].mean()),
            "tau_q05_25_50_75_95": q(o["tau"]),
            "CPI_tau_q": q(o["CPI"]), "SPI_tau_q": q(o["SPI"]), "b_tau_q": q(o["b"]), "pi_tau_q": q(o["gate_pi"]),
            "size_q": q(o["size"]), "novelty_q": q(o["novelty"]), "experience_q": q(o["experience"]),
            "mode_at_tau_share": {dgp.MODES[k]: float((o["mode"] == k).mean()) for k in range(5)},
            "treatment_rate": [float(x) for x in o["A"].mean(0)],
            "propensity_min": [float(x) for x in np.nanmin(o["P"], 0)],
            "propensity_max": [float(x) for x in np.nanmax(o["P"], 0)]}
