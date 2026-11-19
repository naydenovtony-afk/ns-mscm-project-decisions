"""Step 3 tests of the DGP, the ground truth and the confounding (prompt Section 7).
Portfolios come from seeds.tests (200,000 projects); the truth stability test uses seeds.truth_check."""
import json

import numpy as np
import pytest
from conftest import ROOT

from nsmscm import dgp, estimators, rulecheck, truth
from run_all import design_hash

GT = json.loads((ROOT / "results" / "ground_truth.json").read_text(encoding="utf-8"))
SCEN = ["S1", "S2", "S3", "S4"]


# ---------------------------------------------------------------- structural
@pytest.mark.parametrize("scenario", SCEN)
def test_completed_absorbing_and_pi_monotone(cfg, scenario):
    params = dgp.scenario_params(cfg, scenario)
    noise = dgp.draw_noise(5000, np.random.default_rng(np.random.SeedSequence([cfg["seeds"]["tests"], 1])),
                           cfg["dgp"]["t_max"])
    o = dgp.simulate(noise, params, shifted=scenario == "S2", trace=True)
    pi, mode = o["pi_path"], o["mode_path"]
    assert (np.diff(pi, axis=0) >= 0).all()                          # pi never decreases
    comp = mode == dgp.COMPLETED
    assert not (comp[:-1] & ~comp[1:]).any()                         # completed is absorbing
    assert ((pi >= 1) == comp).all()                                  # completed iff pi = 1
    assert (o["tau"] > 0).all() and (o["gate_pi"] >= o["gate_threshold"]).all()


@pytest.mark.parametrize("scenario", SCEN)
def test_propensities_within_bounds(cfg, sim, scenario):
    P = sim(scenario)["P"]
    assert np.isfinite(P).all()
    assert P.min() >= 0.03 and P.max() <= 0.97


def test_propensity_function_bounded_at_extremes(cfg):
    st = {"CPI": np.array([0.01, 5.0]), "SPI": np.array([0.01, 5.0]), "b": np.array([10.0, 0.0]),
          "mode": np.array([3, 0]), "novelty": np.array([1.0, 0.0]), "size": np.array([100.0, 0.01])}
    for b in (cfg["dgp"]["propensity"]["A2"]["b"], cfg["scenarios"]["S3"]["A2_b"]):
        p = dgp.propensities(st, cfg["dgp"], b)
        assert p.min() >= 0.03 and p.max() <= 0.97


# ---------------------------------------------------------------- calibration and rules, S1
def test_s1_mean_overrun_all_zero_is_28pct(sim):
    m = sim("S1", force_all=0.0)["Y"].mean()
    assert abs(m - 0.28) <= 0.03, m
    assert abs(GT["S1"]["mean_Y_all_A0"] - 0.28) <= 0.03


def test_s1_r1_holds(sim):
    zi = rulecheck.r1_interventional(sim("S1", force_all=0.0))
    for name in ("novelty", "size"):
        r = zi[name]
        assert min(r["decile_means"]) > 0 and r["top_vigintile_z"] > 5
        assert r["monotone_increasing"]["violations"] == 0, r["monotone_increasing"]
        assert r["top_minus_bottom_z"] > 5
    rc = rulecheck.r1_conditional(sim("S1"))
    for j, r in rc.items():
        assert r["level"]["violations"] == 0, (j, r["level"])
        assert r["novelty_increasing"]["violations"] == 0, (j, r["novelty_increasing"])


def test_s1_r2_holds(sim):
    r = rulecheck.r2(sim("S1"))
    assert r["decreasing_in_pi"]["violations"] == 0, r["decreasing_in_pi"]
    assert r["fe_t"] < -5 and r["fe_t_true_propensity"] < -5


def test_s1_r3_holds(sim):
    for k, r in rulecheck.r3(sim("S1")).items():
        assert r["violations"] == 0, (k, r)


# ---------------------------------------------------------------- misspecified scenarios
@pytest.mark.parametrize("scenario", ["S3", "S4"])
def test_r2_violated(sim, scenario):
    r = rulecheck.r2(sim(scenario))
    assert r["fe_t"] > 5 and r["fe_t_true_propensity"] > 5
    assert r["decreasing_in_pi"]["violations"] > 0


def test_s4_r1_violated(sim):
    zi = rulecheck.r1_interventional(sim("S4", force_all=0.0))["novelty"]
    assert zi["max_minus_last_z"] > 5                                 # decreasing over part of the novelty range
    assert zi["monotone_increasing"]["violations"] > 0
    assert zi["top_vigintile_z"] < -3                                 # expected overrun negative at high novelty
    rc = rulecheck.r1_conditional(sim("S4"))
    assert any(r["level"]["violations"] > 0 for r in rc.values())     # l_j0 significantly negative somewhere


# ---------------------------------------------------------------- ground truth
def test_ground_truth_file_matches_design(cfg):
    for s in SCEN:
        assert GT[s]["design_sha256"] == design_hash(cfg)
        assert GT[s]["n_projects"] == cfg["truth"]["n_projects"] and GT[s]["seed"] == cfg["seeds"]["truth"]


@pytest.mark.parametrize("scenario", SCEN)
def test_ground_truth_stable_across_seeds(cfg, scenario):
    t2 = truth.scenario_truth(cfg, scenario, cfg["seeds"]["truth_check"])
    for j in range(3):
        a, b = GT[scenario]["theta"][j], t2["theta"][j]
        assert abs(a - b) < max(0.02 * abs(a), 0.002), (scenario, j, a, b)


# ---------------------------------------------------------------- confounding
def test_m1_biased_in_s1(sim):
    o = sim("S1")
    est = estimators.m1_ols(o["Y"], o["A"], estimators.covariates(o))
    z = [abs(e - t) / se for (e, se), t in zip(est, GT["S1"]["theta"])]
    assert max(z) > 5, (est, GT["S1"]["theta"], z)


