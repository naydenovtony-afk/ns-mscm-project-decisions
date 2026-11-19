"""Data-generating process of the NS-M-SCM project simulation (draft Section III/IV.A, prompt Section 3).

All randomness is drawn up front per project, per month and per purpose (`draw_noise`), so runs
that force an intervention at the stage gate use common random numbers. `simulate` is a
deterministic function of (noise, parameters, scenario, forcing).

Per month t = 1, 2, ... for every open project:
  progress  dpi_t = min(1 - pi, (1/D) exp(c_nov (nov-.5) + c_exp (exp-.5)) mode_mult[M] noise_t
                       * (1 + speed_A3 * A3 after tau))
  cost      dAC_t = (1-omega) dpi_t kappa_t cost_mult[M] max(1 + e_t, 0.1)
                    + (omega / D) (1 + team_time_cost * A3 after tau)
  kappa_t   = 1 + delta before tau; after tau
            1 + delta (1 - r1 A1)(1 - r2 A2) + onboarding A3
  EV = pi * BAC, PV = min(t/D, 1) * BAC, CPI = EV/AC, SPI = EV/PV, b = AC/BAC.
  mode      completed (absorbing) once pi = 1; otherwise ordered-logit transition on CPI, SPI,
            novelty, experience and the interventions.
Stage gate tau = first month with pi_t >= g_i, g_i ~ U(gate_pi.low, gate_pi.high) per project (review
schedule; not part of the state S, pi_tau is), decided on the state after month tau;
interventions act from month tau + 1. No decisions after tau (one decision point per project).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.stats import norm

MODES = ["on_track", "at_risk", "delayed", "critical", "completed"]
COMPLETED = 4


@dataclass
class Noise:
    u_size: np.ndarray       # (N,)
    u_nov: np.ndarray
    u_exp: np.ndarray
    z_kappa: np.ndarray
    u_A: np.ndarray          # (3, N) decision draws at tau
    e_prog: np.ndarray       # (t_max, N) log-normal progress noise (log scale, N(0,1))
    e_cost: np.ndarray       # (t_max, N) N(0,1)
    u_mode: np.ndarray       # (t_max, N)
    u_gate: np.ndarray       # (N,) project-specific stage-gate threshold

    @property
    def n(self) -> int:
        return len(self.u_size)


def draw_noise(n: int, rng: np.random.Generator, t_max: int) -> Noise:
    """Fixed order of draws; every purpose has its own array."""
    return Noise(
        u_size=rng.uniform(size=n), u_nov=rng.uniform(size=n), u_exp=rng.uniform(size=n),
        z_kappa=rng.standard_normal(n), u_A=rng.uniform(size=(3, n)),
        e_prog=rng.standard_normal((t_max, n)).astype(np.float32),
        e_cost=rng.standard_normal((t_max, n)).astype(np.float32),
        u_mode=rng.uniform(size=(t_max, n)).astype(np.float32),
        u_gate=rng.uniform(size=n),
    )


def scenario_params(cfg: dict, scenario: str) -> dict:
    """Flatten config into the parameters of one scenario (base X distribution unless S2 shift applied)."""
    d = cfg["dgp"]
    sc = cfg["scenarios"][scenario] or {}
    p = {"dgp": d, "A2_b": d["propensity"]["A2"]["b"], "r1_false": None, "shift": None}
    if "A2_b" in sc:
        p["A2_b"] = sc["A2_b"]
    if "r1_false" in sc:
        p["r1_false"] = sc["r1_false"]
    if "shift" in sc:
        p["shift"] = sc["shift"]
    return p


def static_x(noise: Noise, d: dict, shift: dict | None = None):
    """size, novelty, experience from the uniform draws (quantile transform: shifted portfolios
    reuse the same uniforms)."""
    lm = d["size"]["logmean"] if shift is None else shift["size_logmean"]
    size = np.exp(lm + d["size"]["logsd"] * norm.ppf(noise.u_size))
    na, nb = (d["novelty"]["a"], d["novelty"]["b"]) if shift is None else (shift["novelty_a"], shift["novelty_b"])
    nov = beta_dist.ppf(noise.u_nov, na, nb)
    exp_ = beta_dist.ppf(noise.u_exp, d["experience"]["a"], d["experience"]["b"])
    return size, nov, exp_


def propensities(st: dict, d: dict, A2_b: float) -> np.ndarray:
    """(3, n) propensities at the stage gate from the state dict."""
    pr = d["propensity"]
    f = pr["floor"]
    cpi, spi, b, mode, nov, size = st["CPI"], st["SPI"], st["b"], st["mode"], st["novelty"], st["size"]
    l1 = (pr["A1"]["const"] + pr["A1"]["cpi"] * (1 - cpi)
          + pr["A1"]["at_risk_or_critical"] * np.isin(mode, [1, 3]) + pr["A1"]["novelty"] * (nov - 0.5))
    l2 = pr["A2"]["const"] + pr["A2"]["cpi"] * (1 - cpi) + A2_b * (b - pr["A2"]["b_center"])
    l3 = pr["A3"]["const"] + pr["A3"]["spi"] * (1 - spi) + pr["A3"]["log_size"] * np.log(size)
    return f + (1 - 2 * f) * expit(np.stack([l1, l2, l3]))


def simulate(noise: Noise, params: dict, force: tuple[int, float] | None = None,
             force_all: float | None = None, shifted: bool = False, trace: bool = False) -> dict:
    """Simulate all projects. force=(j, a): A_j at tau set to a, others follow the policy.
    force_all=a: all three set to a. shifted: use the S2 shifted X distribution."""
    d = params["dgp"]
    n, t_max = noise.n, d["t_max"]
    size, nov, exp_ = static_x(noise, d, params["shift"] if shifted else None)
    D = np.clip(np.round(d["duration"]["base"] * size ** d["duration"]["size_exp"]),
                d["duration"]["min"], d["duration"]["max"])
    cf = d["cost_factor"]
    delta = np.exp(cf["a0"] + cf["novelty"] * (nov - 0.5) + cf["log_size"] * np.log(size)
                   + cf["experience"] * (exp_ - 0.5) + cf["sd"] * noise.z_kappa)
    kappa0 = 1 + delta
    if params["r1_false"] is not None:
        rf = params["r1_false"]
        g = 1 - rf["slope"] * np.maximum(nov - rf["threshold"], 0)
        kappa0 = np.maximum(1 + g * delta, rf["kappa_floor"])
    pg = d["progress"]
    base_rate = np.exp(pg["novelty"] * (nov - 0.5) + pg["experience"] * (exp_ - 0.5)) / D
    s = pg["noise_sd"]
    m = d["mode"]
    cost_mult, prog_mult = np.array(m["cost_mult"]), np.array(m["progress_mult"])
    cuts = np.array(m["cuts"])
    eff = d["effects"]
    omega = d["omega"]
    gthr = d["gate_pi"]["low"] + (d["gate_pi"]["high"] - d["gate_pi"]["low"]) * noise.u_gate
    if trace:
        pi_path = np.zeros((t_max + 1, n), dtype=np.float32)
        mode_path = np.zeros((t_max + 1, n), dtype=np.int8)

    pi = np.zeros(n)
    AC = np.zeros(n)
    mode = np.zeros(n, dtype=np.int64)
    kappa = kappa0.copy()
    tau = np.full(n, -1)
    A = np.zeros((3, n))
    P = np.full((3, n), np.nan)
    post = np.zeros(n, dtype=bool)
    speed = np.zeros(n)
    time_mult = np.ones(n)
    trans_boost = np.zeros(n)
    gate = {k: np.full(n, np.nan) for k in ("pi", "CPI", "SPI", "b", "mode")}
    done = np.zeros(n, dtype=bool)
    months = np.zeros(n)

    for t in range(1, t_max + 1):
        o = ~done
        if not o.any():
            break
        idx = np.flatnonzero(o)
        mo = mode[idx]
        prog_noise = np.exp(s * noise.e_prog[t - 1, idx] - s * s / 2)
        rate = base_rate[idx] * prog_mult[mo] * prog_noise * (1 + speed[idx])
        dpi = np.minimum(rate, 1 - pi[idx])
        work = (1 - omega) * dpi * kappa[idx] * cost_mult[mo] * np.maximum(1 + d["cost_noise_sd"] * noise.e_cost[t - 1, idx], 0.1)
        AC[idx] += work + omega / D[idx] * time_mult[idx]
        pi[idx] += dpi
        months[idx] = t
        if t == t_max:                                   # close remaining projects at t_max
            rem = 1 - pi[idx]
            AC[idx] += (1 - omega) * rem * kappa[idx] * cost_mult[mo]
            pi[idx] = 1.0
        comp = pi[idx] >= 1 - 1e-12
        pi[idx[comp]] = 1.0
        mode[idx[comp]] = COMPLETED
        done[idx[comp]] = True
        # mode transition for projects still open
        j = idx[~comp]
        if len(j):
            cpi = pi[j] / AC[j]
            spi = pi[j] / np.minimum(t / D[j], 1.0)
            eta = (m["persistence"] * mode[j] + m["cpi"] * (1 - cpi) + m["spi"] * (1 - spi)
                   + m["novelty"] * (nov[j] - 0.5) + m["experience"] * (exp_[j] - 0.5) - trans_boost[j])
            pk = expit(cuts[None, :] - eta[:, None])        # P(next <= k), k = 0..2
            u = noise.u_mode[t - 1, j]
            mode[j] = (u[:, None] > pk).sum(1)
            # stage gate
            g = j[(tau[j] < 0) & (pi[j] >= gthr[j])]
            if len(g):
                st = {"CPI": pi[g] / AC[g], "SPI": pi[g] / np.minimum(t / D[g], 1.0), "b": AC[g] / d["bac"],
                      "mode": mode[g], "novelty": nov[g], "size": size[g]}
                tau[g] = t
                for k in ("CPI", "SPI", "b"):
                    gate[k][g] = st[k]
                gate["pi"][g] = pi[g]
                gate["mode"][g] = mode[g]
                p = propensities(st, d, params["A2_b"])
                P[:, g] = p
                a = (noise.u_A[:, g] < p).astype(float)
                if force is not None:
                    a[force[0]] = force[1]
                if force_all is not None:
                    a[:] = force_all
                A[:, g] = a
                mg = mode[g]
                r1 = np.clip(eff["A1"]["base"] + eff["A1"]["novelty"] * nov[g] + np.array(eff["A1"]["mode_bonus"])[mg], 0, 0.9)
                r2 = np.clip(eff["A2"]["base"] + eff["A2"]["novelty"] * nov[g] + np.array(eff["A2"]["mode_bonus"])[mg], 0, 0.9)
                ex = kappa0[g] - 1
                kappa[g] = 1 + ex * (1 - r1 * a[0]) * (1 - r2 * a[1]) + eff["A3"]["onboarding"] * a[2]
                speed[g] = a[2] * (eff["A3"]["speed_base"] + eff["A3"]["speed_novelty"] * nov[g])
                time_mult[g] = 1 + eff["A3"]["team_time_cost"] * a[2]
                trans_boost[g] = eff["A1"]["transition"] * a[0] + eff["A2"]["transition"] * a[1] + eff["A3"]["transition"] * a[2]
                post[g] = True
        if trace:
            pi_path[t:], mode_path[t:] = pi, mode     # rows after the last simulated month keep the final state

    Y = AC / d["bac"] - 1
    return {"Y": Y, "A": A.T, "P": P.T, "tau": tau, "months": months, "D": D,
            "size": size, "novelty": nov, "experience": exp_, "kappa0": kappa0,
            "gate_pi": gate["pi"], "CPI": gate["CPI"], "SPI": gate["SPI"], "b": gate["b"],
            "mode": gate["mode"], "closed_at_tmax": (months >= t_max) & (tau >= 0), "gate_threshold": gthr,
            **({"pi_path": pi_path, "mode_path": mode_path} if trace else {})}


def simulate_chunked(cfg: dict, scenario: str, n: int, seed: int, chunk: int, **kw) -> dict:
    """Simulate n projects in chunks; chunk c uses SeedSequence([seed, c]). The chunking depends only
    on (n, chunk), so forced and unforced runs see the same random numbers."""
    params = scenario_params(cfg, scenario)
    outs = []
    for c, start in enumerate(range(0, n, chunk)):
        m = min(chunk, n - start)
        noise = draw_noise(m, np.random.default_rng(np.random.SeedSequence([seed, c])), cfg["dgp"]["t_max"])
        outs.append(simulate(noise, params, **kw))
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def portfolio(cfg: dict, scenario: str, n: int, seed_seq: np.random.SeedSequence, shifted: bool = False) -> dict:
    """One observed portfolio (estimation data)."""
    params = scenario_params(cfg, scenario)
    noise = draw_noise(n, np.random.default_rng(seed_seq), cfg["dgp"]["t_max"])
    return simulate(noise, params, shifted=shifted)
