"""results/table_dgp_parameters.csv (Table II): every DGP parameter in config.yaml with a description."""
from __future__ import annotations

import csv

DESC = {
    "bac": "Budget at completion (every project)",
    "omega": "Share of BAC that is time-related cost; planned overhead per month = omega/D",
    "t_max": "Maximum months; open projects are closed at t_max",
    "gate_pi.low": "Stage gate: first month with pi >= g_i, g_i ~ U(low, high) per project (review schedule): low",
    "gate_pi.high": "Stage gate threshold g_i ~ U(low, high): high",
    "size.logmean": "Size ~ lognormal: mean of log size",
    "size.logsd": "Size ~ lognormal: sd of log size",
    "novelty.a": "Novelty ~ Beta(a, b): a", "novelty.b": "Novelty ~ Beta(a, b): b",
    "experience.a": "Team experience ~ Beta(a, b): a", "experience.b": "Team experience ~ Beta(a, b): b",
    "duration.base": "Planned duration D = clip(round(base * size^size_exp), min, max) months: base",
    "duration.size_exp": "Planned duration: exponent of size",
    "duration.min": "Planned duration: minimum months", "duration.max": "Planned duration: maximum months",
    "cost_factor.a0": "True cost factor kappa = 1 + delta; log delta intercept (calibrated: mean overrun 28% with all A = 0, S1)",
    "cost_factor.novelty": "log delta: coefficient of (novelty - 0.5) (planning fallacy, R1)",
    "cost_factor.log_size": "log delta: coefficient of log size (R1)",
    "cost_factor.experience": "log delta: coefficient of (experience - 0.5)",
    "cost_factor.sd": "log delta: sd of the project-level normal term (latent, unobserved)",
    "progress.novelty": "Monthly progress rate (1/D) exp(...): coefficient of (novelty - 0.5)",
    "progress.experience": "Monthly progress rate: coefficient of (experience - 0.5)",
    "progress.noise_sd": "Monthly progress noise: sd of the log-normal multiplier (mean 1)",
    "cost_noise_sd": "Monthly work-cost noise: sd of e in (1 + e), floored at 0.1",
    "mode.cost_mult": "Work-cost multiplier by mode (on track, at risk, delayed, critical)",
    "mode.progress_mult": "Progress multiplier by mode (on track, at risk, delayed, critical)",
    "mode.persistence": "Mode transition (ordered logit) eta: coefficient of current mode",
    "mode.cpi": "Mode transition eta: coefficient of (1 - CPI)",
    "mode.spi": "Mode transition eta: coefficient of (1 - SPI)",
    "mode.novelty": "Mode transition eta: coefficient of (novelty - 0.5)",
    "mode.experience": "Mode transition eta: coefficient of (experience - 0.5)",
    "mode.cuts": "Mode transition cut points: P(next <= k) = sigmoid(cut_k - eta)",
    "propensity.floor": "Propensities p = floor + (1 - 2 floor) sigmoid(logit) (positivity)",
    "propensity.A1.const": "A1 review logit: intercept",
    "propensity.A1.cpi": "A1 logit: coefficient of (1 - CPI)",
    "propensity.A1.at_risk_or_critical": "A1 logit: indicator mode at risk or critical",
    "propensity.A1.novelty": "A1 logit: coefficient of (novelty - 0.5)",
    "propensity.A2.const": "A2 re-planning logit: intercept",
    "propensity.A2.cpi": "A2 logit: coefficient of (1 - CPI)",
    "propensity.A2.b": "A2 logit: coefficient of (b - b_center); negative = escalation of commitment (R2), S1/S2",
    "propensity.A2.b_center": "A2 logit: centring of the spent share b",
    "propensity.A3.const": "A3 reinforcement logit: intercept",
    "propensity.A3.spi": "A3 logit: coefficient of (1 - SPI)",
    "propensity.A3.log_size": "A3 logit: coefficient of log size",
    "effects.A1.base": "A1: share of the remaining excess cost factor removed, intercept",
    "effects.A1.novelty": "A1: excess removed, coefficient of novelty",
    "effects.A1.mode_bonus": "A1: excess removed, added by mode at tau (on track, at risk, delayed, critical)",
    "effects.A1.transition": "A1: reduction of the mode-transition eta after tau",
    "effects.A2.base": "A2: share of the remaining excess cost factor removed, intercept",
    "effects.A2.novelty": "A2: excess removed, coefficient of novelty",
    "effects.A2.mode_bonus": "A2: excess removed, added by mode at tau",
    "effects.A2.transition": "A2: reduction of the mode-transition eta after tau",
    "effects.A3.speed_base": "A3: progress speed-up, intercept",
    "effects.A3.speed_novelty": "A3: progress speed-up, coefficient of novelty",
    "effects.A3.team_time_cost": "A3: monthly time-related cost multiplied by (1 + this)",
    "effects.A3.onboarding": "A3: added to the cost factor of remaining work",
    "effects.A3.transition": "A3: reduction of the mode-transition eta after tau",
}
SCEN_DESC = {
    "S2.shift.size_logmean": "S2 evaluation portfolio: mean of log size",
    "S2.shift.novelty_a": "S2 evaluation portfolio: novelty Beta a",
    "S2.shift.novelty_b": "S2 evaluation portfolio: novelty Beta b",
    "S3.A2_b": "S3: A2 logit coefficient of b (positive: R2 false)",
    "S4.A2_b": "S4: A2 logit coefficient of b (positive: R2 false)",
    "S4.r1_false.threshold": "S4: delta multiplied by 1 - slope (novelty - threshold)_+ : threshold (R1 false)",
    "S4.r1_false.slope": "S4: slope of that multiplier",
    "S4.r1_false.kappa_floor": "S4: lower bound of the cost factor kappa",
}


def _flat(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flat(v, key + ".")
        else:
            yield key, v


def write(cfg: dict, path: str) -> None:
    rows = []
    for k, v in _flat(cfg["dgp"]):
        if k not in DESC:
            raise KeyError(f"no description for DGP parameter {k}")
        rows.append({"block": "all scenarios", "parameter": k, "value": v, "description": DESC[k]})
    for k, v in _flat({s: (cfg["scenarios"][s] or {}) for s in cfg["scenarios"]}):
        if k not in SCEN_DESC:
            raise KeyError(f"no description for scenario parameter {k}")
        rows.append({"block": k.split(".")[0], "parameter": k, "value": v, "description": SCEN_DESC[k]})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["block", "parameter", "value", "description"])
        w.writeheader()
        w.writerows(rows)
