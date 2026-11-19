"""Step 5: tables, figures, paper_numbers.json and SUMMARY.md from results/raw/ (no re-simulation).

Replications used per cell: rep < experiments.R[N] (R = 100 for N in {300, 1000}, 50 for N = 5000).

Metrics per (scenario, N, method, intervention), theta from results/ground_truth.json:
  mean estimate; bias = mean - theta, MC SE = sd(est)/sqrt(R);
  RMSE, MC SE by the delta method sd((est - theta)^2) / (2 RMSE sqrt(R));
  coverage of est +- 1.96 se, MC SE sqrt(c(1 - c)/R); mean CI width 2 * 1.96 * se;
  rule violation rate (M3, M4): mean over replications of the share of held-out states with c_k > 0,
  averaged over the rule components of that intervention's nuisance functions (components reported
  separately in table_violations.csv); runtime: mean seconds per replication for that intervention.

Decision rules for the draft's hypotheses (written 2026-11-17 before any replication result, see
CHANGELOG.md). Paired differences use the same replications (common data):
  H1 (S1, correct rules, small portfolios N in {300, 1000}): per intervention, RMSE(M4) < RMSE(M3) and the
     paired mean difference of squared errors (M4 - M3) is below 0 by more than 2 MC SE.
  H2 (one nuisance rule wrong: S3 all interventions; S4 A1 and A3; every N): coverage(M4) >= 0.90 and
     |bias(M4) - bias(M3)| <= 2 MC SE of the paired difference.
  H3 (both nuisance rules wrong: S4 A2): bias(M4) at N = 5000 differs from 0 by more than 2 MC SE and
     |bias(M4)| at N = 5000 >= |bias(M4)| at N = 300 - 2 MC SE of that difference.
Amendments (committed 2026-11-17 before any phase-1 result was read, see CHANGELOG.md):
  H3 additionally requires, at N = 5000, |bias(M4)| - |bias(M3)| > 2 MC SE, paired across replications:
     with s4, s3 the signs of the two biases, the difference equals mean_r[s4 (e4_r - theta) - s3 (e3_r - theta)],
     MC SE = sd_r of that quantity / sqrt(R). (A bias shared by M3 is not attributed to the rules.)
  Every hypothesis is evaluated and reported separately for each intervention x N cell it covers
  ("supported" / "not supported"); no pooling, no counts across cells, no overall verdict.
  H2's bias criterion is a failure-to-reject criterion (no significant M4 - M3 bias difference), not
  evidence of equality, and is labelled so in SUMMARY.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

Z = 1.959963984540054
METHODS = ["M1", "M2", "M3", "M4", "M5"]
COLORS = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#eda100", "M5": "#e87ba4"}  # fixed slot order
MARKERS = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M5": "v"}
NAMES = {1: "A1 review", 2: "A2 re-planning", 3: "A3 reinforcement"}


def load_raw(res: Path, cfg: dict) -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted((res / "raw").glob("S*_N*.csv"))], ignore_index=True)
    R = {int(k): int(v) for k, v in cfg["experiments"]["R"].items()}
    d = d[d.rep < d.N.map(R)].copy()
    gt = json.loads((res / "ground_truth.json").read_text(encoding="utf-8"))
    d["truth"] = [gt[s]["theta"][j - 1] for s, j in zip(d.scenario, d.intervention)]
    return d


def _vrate(v) -> float:
    if not isinstance(v, str) or not v:
        return np.nan
    x = json.loads(v)
    return float(np.mean(list(x.values()))) if x else np.nan


def cell_metrics(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, n, m, j), g in d.groupby(["scenario", "N", "method", "intervention"]):
        R, th, est, se = len(g), g.truth.iloc[0], g.estimate.to_numpy(), g.se.to_numpy()
        err2 = (est - th) ** 2
        rmse = np.sqrt(err2.mean())
        cov = np.mean(np.abs(est - th) <= Z * se)
        vr = g.violations.map(_vrate) if "violations" in g else pd.Series(np.nan, index=g.index)
        rows.append({"scenario": s, "N": n, "method": m, "intervention": j, "R": R, "true_theta": th,
                     "mean_estimate": est.mean(), "bias": est.mean() - th, "bias_mcse": est.std(ddof=1) / np.sqrt(R),
                     "rmse": rmse, "rmse_mcse": err2.std(ddof=1) / (2 * rmse * np.sqrt(R)),
                     "coverage": cov, "coverage_mcse": np.sqrt(cov * (1 - cov) / R),
                     "mean_ci_width": (2 * Z * se).mean(),
                     "violation_rate": vr.mean() if m in ("M3", "M4") else np.nan,
                     "runtime_s": g.runtime_s.mean()})
    return pd.DataFrame(rows).sort_values(["scenario", "intervention", "method", "N"]).reset_index(drop=True)


def paired(d: pd.DataFrame, s: str, n: int, j: int, a: str = "M4", b: str = "M3") -> dict:
    x = d[(d.scenario == s) & (d.N == n) & (d.intervention == j)]
    pa = x[x.method == a].set_index("rep")
    pb = x[x.method == b].set_index("rep")
    reps = pa.index.intersection(pb.index)
    th = pa.truth.iloc[0]
    db = (pa.estimate - pb.estimate).loc[reps]
    dse = ((pa.estimate - th) ** 2 - (pb.estimate - th) ** 2).loc[reps]
    R = len(reps)
    return {"R": R, "bias_diff": db.mean(), "bias_diff_mcse": db.std(ddof=1) / np.sqrt(R),
            "sqerr_diff": dse.mean(), "sqerr_diff_mcse": dse.std(ddof=1) / np.sqrt(R)}


def violations_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, n, m, j), g in d[d.method.isin(["M3", "M4"])].groupby(["scenario", "N", "method", "intervention"]):
        comps = pd.DataFrame([json.loads(v) for v in g.violations])
        for c in comps.columns:
            rows.append({"scenario": s, "N": n, "method": m, "intervention": j, "component": c,
                         "violation_rate": comps[c].mean(), "violation_rate_mcse": comps[c].std(ddof=1) / np.sqrt(len(comps))})
    return pd.DataFrame(rows)


def mu_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, n), g in d[d.method == "M4"].groupby(["scenario", "N"]):
        for (j, v) in zip(g.intervention, g.mu_chosen):
            for x in json.loads(v):
                rows.append({"scenario": s, "N": n, "intervention": j, "nuisance": x["nuisance"], "mu": x["mu"]})
    t = pd.DataFrame(rows)
    out = t.groupby(["scenario", "N", "nuisance", "mu"]).size().rename("count").reset_index()
    out["share"] = out["count"] / out.groupby(["scenario", "N", "nuisance"])["count"].transform("sum")
    return out


def hypotheses(d: pd.DataFrame, cells: pd.DataFrame) -> list[dict]:
    out = []
    for n in (300, 1000):
        for j in (1, 2, 3):
            p = paired(d, "S1", n, j)
            c = cells.set_index(["scenario", "N", "method", "intervention"])
            r3, r4 = c.loc[("S1", n, "M3", j), "rmse"], c.loc[("S1", n, "M4", j), "rmse"]
            ok = (r4 < r3) and (p["sqerr_diff"] < -2 * p["sqerr_diff_mcse"])
            out.append({"hypothesis": "H1", "scenario": "S1", "N": n, "intervention": j, "rmse_M3": r3, "rmse_M4": r4,
                        "rmse_ratio_M4_M3": r4 / r3, **{k: p[k] for k in ("sqerr_diff", "sqerr_diff_mcse")},
                        "supported": bool(ok)})
    c = cells.set_index(["scenario", "N", "method", "intervention"])
    for s, js in (("S3", (1, 2, 3)), ("S4", (1, 3))):
        for n in (300, 1000, 5000):
            for j in js:
                p = paired(d, s, n, j)
                cov4 = c.loc[(s, n, "M4", j), "coverage"]
                ok = cov4 >= 0.90 and abs(p["bias_diff"]) <= 2 * p["bias_diff_mcse"]
                out.append({"hypothesis": "H2", "scenario": s, "N": n, "intervention": j, "coverage_M4": cov4,
                            "coverage_M3": c.loc[(s, n, "M3", j), "coverage"], "bias_M4": c.loc[(s, n, "M4", j), "bias"],
                            "bias_M3": c.loc[(s, n, "M3", j), "bias"],
                            **{k: p[k] for k in ("bias_diff", "bias_diff_mcse")}, "supported": bool(ok)})
    b5, b3 = c.loc[("S4", 5000, "M4", 2)], c.loc[("S4", 300, "M4", 2)]
    m3_5 = c.loc[("S4", 5000, "M3", 2)]
    shrink_se = np.hypot(b5.bias_mcse, b3.bias_mcse)
    a = abs_bias_diff(d, "S4", 5000, 2)
    crit = {"nonzero_at_5000": bool(abs(b5.bias) > 2 * b5.bias_mcse),
            "not_shrinking_300_to_5000": bool(abs(b5.bias) >= abs(b3.bias) - 2 * shrink_se),
            "exceeds_M3_at_5000": bool(a["abs_bias_diff"] > 2 * a["abs_bias_diff_mcse"])}
    out.append({"hypothesis": "H3", "scenario": "S4", "N": 5000, "intervention": 2,
                "bias_M4_N300": b3.bias, "bias_M4_N300_mcse": b3.bias_mcse,
                "bias_M4_N1000": c.loc[("S4", 1000, "M4", 2), "bias"],
                "bias_M4_N5000": b5.bias, "bias_M4_N5000_mcse": b5.bias_mcse,
                "bias_M3_N5000": m3_5.bias, "bias_M3_N5000_mcse": m3_5.bias_mcse, **a, **crit,
                "supported": all(crit.values())})
    return out


def abs_bias_diff(d: pd.DataFrame, s: str, n: int, j: int) -> dict:
    """|bias(M4)| - |bias(M3)|, paired across replications (signs of the two biases held fixed)."""
    x = d[(d.scenario == s) & (d.N == n) & (d.intervention == j)]
    e4 = x[x.method == "M4"].set_index("rep").estimate
    e3 = x[x.method == "M3"].set_index("rep").estimate
    reps = e4.index.intersection(e3.index)
    th = x.truth.iloc[0]
    r4, r3 = (e4.loc[reps] - th), (e3.loc[reps] - th)
    q = np.sign(r4.mean()) * r4 - np.sign(r3.mean()) * r3
    return {"abs_bias_diff": float(q.mean()), "abs_bias_diff_mcse": float(q.std(ddof=1) / np.sqrt(len(reps)))}


def fig_rmse(cells: pd.DataFrame, res: Path) -> None:
    s1 = cells[cells.scenario == "S1"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, j in zip(axes, (1, 2, 3)):
        for m in METHODS:
            g = s1[(s1.method == m) & (s1.intervention == j)].sort_values("N")
            ax.errorbar(g.N, g.rmse, yerr=Z * g.rmse_mcse, color=COLORS[m], marker=MARKERS[m], ms=5, lw=2,
                        capsize=2, label=m)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([300, 1000, 5000], ["300", "1000", "5000"])
        ax.minorticks_off()
        ax.set_title(NAMES[j], fontsize=10)
        ax.set_xlabel("portfolio size N")
        ax.grid(True, color="#e5e5e5", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("RMSE (log scale)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="center right")
    fig.suptitle("S1: RMSE against N (error bars: 95% MC interval)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    for ext in ("pdf", "png"):
        fig.savefig(res / f"fig_rmse_vs_N.{ext}", dpi=200)
    plt.close(fig)


def fig_rmse_1col(cells: pd.DataFrame, res: Path) -> None:
    """Single-column version of fig_rmse_vs_N: 3 panels stacked vertically, width 3.3 in, PNG 300 dpi.
    Same data and encoding as fig_rmse; shared y scale, y-axis label on every panel."""
    s1 = cells[cells.scenario == "S1"]
    fig, axes = plt.subplots(3, 1, figsize=(3.3, 7.2), sharex=True, sharey=True)
    for ax, j in zip(axes, (1, 2, 3)):
        for m in METHODS:
            g = s1[(s1.method == m) & (s1.intervention == j)].sort_values("N")
            ax.errorbar(g.N, g.rmse, yerr=Z * g.rmse_mcse, color=COLORS[m], marker=MARKERS[m], ms=3.5, lw=1.5,
                        capsize=1.5, label=m)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([300, 1000, 5000], ["300", "1000", "5000"])
        ax.minorticks_off()
        ax.set_title(NAMES[j], fontsize=8, loc="left", pad=3)
        ax.set_ylabel("RMSE (log)", fontsize=7.5)
        ax.tick_params(labelsize=7, labelleft=True)
        ax.grid(True, color="#e5e5e5", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("portfolio size N", fontsize=7.5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=7, loc="upper center", ncol=5, handlelength=1.4, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.6)
    fig.savefig(res / "fig_rmse_vs_N_1col.png", dpi=300)
    plt.close(fig)


def fig_coverage(cells: pd.DataFrame, res: Path) -> None:
    c = cells[cells.method.isin(["M3", "M4"])]
    Ns = sorted(c.N.unique())
    fig, axes = plt.subplots(len(Ns), 3, figsize=(11, 2.6 * len(Ns)), sharey=True, sharex=True, squeeze=False)
    sc = ["S1", "S2", "S3", "S4"]
    lo = min(0.5, float((c.coverage - Z * c.coverage_mcse).min()) - 0.02)
    for r, n in enumerate(Ns):
        for k, j in enumerate((1, 2, 3)):
            ax = axes[r][k]
            ax.axhline(0.95, color="#52514e", ls="--", lw=1)
            for off, m in ((-0.1, "M3"), (0.1, "M4")):
                g = c[(c.method == m) & (c.intervention == j) & (c.N == n)].set_index("scenario").reindex(sc)
                ax.errorbar(np.arange(4) + off, g.coverage, yerr=Z * g.coverage_mcse, fmt=MARKERS[m], color=COLORS[m],
                            ms=5, capsize=2, lw=1.2, label=m)
            ax.set_xticks(range(4), sc)
            ax.set_ylim(lo, 1.01)
            ax.grid(True, axis="y", color="#e5e5e5", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(NAMES[j], fontsize=10)
            if k == 0:
                ax.set_ylabel(f"N = {n}\ncoverage")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="center right")
    fig.suptitle("Coverage of 95% CIs by scenario (dashed: 0.95; error bars: 95% MC interval)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    for ext in ("pdf", "png"):
        fig.savefig(res / f"fig_coverage.{ext}", dpi=200)
    plt.close(fig)


def m1_bias_200k(cfg: dict, res: Path) -> list[dict]:
    """M1 on one S1 portfolio of 200,000 projects (seeds.tests), against the truth (confounding check)."""
    from . import dgp, estimators
    gt = json.loads((res / "ground_truth.json").read_text(encoding="utf-8"))
    o = dgp.simulate_chunked(cfg, "S1", cfg["truth"]["n_projects"], cfg["seeds"]["tests"], cfg["truth"]["chunk"])
    est = estimators.m1_ols(o["Y"], o["A"], estimators.covariates(o))
    return [{"intervention": j + 1, "estimate": e, "se": se, "truth": t, "bias": e - t, "z": (e - t) / se,
             "n_projects": cfg["truth"]["n_projects"], "seed": cfg["seeds"]["tests"]}
            for j, ((e, se), t) in enumerate(zip(est, gt["S1"]["theta"]))]


def _py(v):
    return v.item() if hasattr(v, "item") else v


def paper_numbers(files: dict, id_cols: dict) -> dict:
    """Every numeric cell of the listed CSV files, keyed by its identifying fields, with file and data row."""
    out = {}
    for fname, t in files.items():
        ids = id_cols[fname]
        for i, row in t.reset_index(drop=True).iterrows():
            key = ".".join(f"{c}={_py(row[c])}" for c in ids if c in t.columns and not pd.isna(row[c]))
            for col in t.columns:
                if col in ids:
                    continue
                v = row[col]
                if isinstance(v, (bool, np.bool_)):
                    v = bool(v)
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    if pd.isna(v):
                        continue
                    v = float(v)
                else:
                    continue
                out[f"{fname}|{key}|{col}"] = {"value": v, "file": f"results/{fname}", "row": i + 1,
                                               "column": col}
    return out


def _vr(x):
    return "–" if pd.isna(x) else f"{x:.3f}"


def summary_md(cells, hyp, mu, viol, m1b, R: dict, gt: dict) -> str:
    L = ["# SUMMARY", "",
         "Generated by `run_all.py --stage analyze`. Every number restates a file in results/.", "",
         f"Replications per cell: R = {R[300]} for N = 300, R = {R[1000]} for N = 1000, R = {R[5000]} for N = 5000 "
         "(phase 1: 50 for every cell; phase 2, pre-declared before any result: N = 300 and 1000 extended to 100). "
         "Monte Carlo standard errors in brackets.", "",
         "Sources: ground_truth.json, table_main.csv (S1), table_scenarios.csv (S2-S4), hypotheses.csv, "
         "m1_bias_200k.csv, table_mu.csv, table_violations.csv.", ""]
    L += ["## True effects (ground_truth.json, 200,000 projects)", "", "| scenario | A1 | A2 | A3 |", "|---|---|---|---|"]
    for s in ("S1", "S2", "S3", "S4"):
        L.append(f"| {s} | " + " | ".join(f"{t:+.4f}" for t in gt[s]["theta"]) + " |")
    hdr = "| bias [MC SE] | RMSE [MC SE] | coverage [MC SE] | CI width | violation rate |"
    L += ["", "## S1: all methods (table_main.csv)", "", "| intervention | method | N " + hdr, "|---|---|---|---|---|---|---|---|"]
    for _, r in cells[cells.scenario == "S1"].sort_values(["intervention", "method", "N"]).iterrows():
        L.append(f"| A{r.intervention} | {r.method} | {r.N} | {r.bias:+.4f} [{r.bias_mcse:.4f}] | {r.rmse:.4f} [{r.rmse_mcse:.4f}] | "
                 f"{r.coverage:.3f} [{r.coverage_mcse:.3f}] | {r.mean_ci_width:.4f} | {_vr(r.violation_rate)} |")
    L += ["", "## S2-S4: M3, M4, M5 (table_scenarios.csv)", "", "| scenario | intervention | method | N " + hdr,
          "|---|---|---|---|---|---|---|---|---|"]
    sc = cells[(cells.scenario != "S1") & cells.method.isin(["M3", "M4", "M5"])]
    for _, r in sc.sort_values(["scenario", "intervention", "method", "N"]).iterrows():
        L.append(f"| {r.scenario} | A{r.intervention} | {r.method} | {r.N} | {r.bias:+.4f} [{r.bias_mcse:.4f}] | {r.rmse:.4f} [{r.rmse_mcse:.4f}] | "
                 f"{r.coverage:.3f} [{r.coverage_mcse:.3f}] | {r.mean_ci_width:.4f} | {_vr(r.violation_rate)} |")
    L += ["", "## Hypotheses (hypotheses.csv; decision rules in nsmscm/analysis.py, fixed before any replication result)", "",
          "Each hypothesis is evaluated separately in every intervention x N cell it covers; there is no pooled or overall verdict.",
          "In S4, A1 and A3 test H2 (only the outcome rule R1 is wrong); A2 tests H3 (R1 and R2 wrong, i.e. both kinds of nuisance function).", ""]
    h1 = hyp[hyp.hypothesis == "H1"]
    L.append("**H1** (S1, N in {300, 1000}: RMSE M4 < RMSE M3 and paired squared-error difference M4 - M3 below -2 MC SE), per cell:")
    for _, r in h1.iterrows():
        L.append(f"- N = {int(r.N)}, A{int(r.intervention)}: RMSE M3 {r.rmse_M3:.4f}, M4 {r.rmse_M4:.4f} (ratio {r.rmse_ratio_M4_M3:.3f}); "
                 f"paired squared-error difference {r.sqerr_diff:+.2e} [MC SE {r.sqerr_diff_mcse:.2e}]: "
                 f"{'supported' if r.supported else 'not supported'}")
    h2 = hyp[hyp.hypothesis == "H2"]
    L += ["", "**H2** (S3 all interventions, S4 A1 and A3: coverage M4 >= 0.90 and |bias M4 - bias M3| <= 2 MC SE of the paired "
          "difference), per cell. The bias part is a failure-to-reject criterion: it means no significant M4 - M3 bias difference "
          "was detected, not that the biases are equal."]
    for _, r in h2.iterrows():
        L.append(f"- {r.scenario}, N = {int(r.N)}, A{int(r.intervention)}: coverage M4 {r.coverage_M4:.3f} (M3 {r.coverage_M3:.3f}); "
                 f"bias M4 {r.bias_M4:+.4f}, M3 {r.bias_M3:+.4f}; paired difference {r.bias_diff:+.4f} [MC SE {r.bias_diff_mcse:.4f}] "
                 f"(failure-to-reject criterion {'met' if abs(r.bias_diff) <= 2 * r.bias_diff_mcse else 'not met'}): "
                 f"{'supported' if r.supported else 'not supported'}")
    h3 = hyp[hyp.hypothesis == "H3"].iloc[0]
    yn = lambda b: "met" if b else "not met"  # noqa: E731
    L += ["", "**H3** (S4, A2, N = 5000; three criteria, all required), single cell: "
          f"{'supported' if h3.supported else 'not supported'}.",
          f"- M4 bias differs from 0 at N = 5000 by > 2 MC SE: {yn(h3.nonzero_at_5000)} "
          f"(N = 5000 {h3.bias_M4_N5000:+.4f} [{h3.bias_M4_N5000_mcse:.4f}]).",
          f"- M4 bias does not shrink from N = 300 to 5000: {yn(h3.not_shrinking_300_to_5000)} "
          f"(N = 300 {h3.bias_M4_N300:+.4f} [{h3.bias_M4_N300_mcse:.4f}], N = 1000 {h3.bias_M4_N1000:+.4f}).",
          f"- |bias M4| - |bias M3| > 2 MC SE (paired) at N = 5000: {yn(h3.exceeds_M3_at_5000)} "
          f"(difference {h3.abs_bias_diff:+.4f} [MC SE {h3.abs_bias_diff_mcse:.4f}]; M3 bias {h3.bias_M3_N5000:+.4f} [{h3.bias_M3_N5000_mcse:.4f}])."]
    L += ["", "## M1 confounding bias on 200,000 projects (S1; m1_bias_200k.csv)", ""]
    for _, r in m1b.iterrows():
        L.append(f"- A{int(r.intervention)}: estimate {r.estimate:+.4f} (SE {r.se:.4f}), truth {r.truth:+.4f}, bias {r.bias:+.4f} (z = {r.z:+.1f})")
    L += ["", "## Chosen mu (M4; share of fold-level choices; table_mu.csv)", "",
          "| scenario | N | nuisance | mu = 0.1 | mu = 1 | mu = 10 |", "|---|---|---|---|---|---|"]
    pv = mu.pivot_table(index=["scenario", "N", "nuisance"], columns="mu", values="share", fill_value=0.0)
    for (s, n, k), r in pv.iterrows():
        L.append(f"| {s} | {n} | {k} | " + " | ".join(f"{r.get(m, 0.0):.3f}" for m in (0.1, 1.0, 10.0)) + " |")
    L += ["", "## Rule violation rates of the fitted nuisance functions (held-out states; table_violations.csv)", "",
          "Mean over replications and interventions; component e:R2 exists only for A2. "
          "Per-intervention rates with MC SE are in table_violations.csv.", "",
          "| scenario | N | component | M3 | M4 |", "|---|---|---|---|---|"]
    vv = viol.groupby(["scenario", "N", "component", "method"]).violation_rate.mean().unstack("method")
    for (s, n, k), r in vv.iterrows():
        L.append(f"| {s} | {n} | {k} | {r.get('M3', np.nan):.3f} | {r.get('M4', np.nan):.3f} |")
    return "\n".join(L) + "\n"


def run(cfg: dict, res: Path = Path("results")) -> list[str]:
    d = load_raw(res, cfg)
    cells = cell_metrics(d)
    main = cells[cells.scenario == "S1"].reset_index(drop=True)
    scen = cells[(cells.scenario != "S1") & cells.method.isin(["M3", "M4", "M5"])].reset_index(drop=True)
    main.to_csv(res / "table_main.csv", index=False)
    scen.to_csv(res / "table_scenarios.csv", index=False)
    hyp = pd.DataFrame(hypotheses(d, cells))
    hyp.to_csv(res / "hypotheses.csv", index=False)
    mu = mu_table(d)
    mu.to_csv(res / "table_mu.csv", index=False)
    viol = violations_table(d)
    viol.to_csv(res / "table_violations.csv", index=False)
    m1b = pd.DataFrame(m1_bias_200k(cfg, res))
    m1b.to_csv(res / "m1_bias_200k.csv", index=False)
    fig_rmse(cells, res)
    fig_rmse_1col(cells, res)
    fig_coverage(cells, res)
    gt = json.loads((res / "ground_truth.json").read_text(encoding="utf-8"))
    files = {"table_main.csv": main, "table_scenarios.csv": scen, "hypotheses.csv": hyp, "table_mu.csv": mu,
             "table_violations.csv": viol, "m1_bias_200k.csv": m1b}
    ids = {"table_main.csv": ["scenario", "N", "method", "intervention"],
           "table_scenarios.csv": ["scenario", "N", "method", "intervention"],
           "hypotheses.csv": ["hypothesis", "scenario", "N", "intervention"],
           "table_mu.csv": ["scenario", "N", "nuisance", "mu"],
           "table_violations.csv": ["scenario", "N", "method", "intervention", "component"],
           "m1_bias_200k.csv": ["intervention"]}
    pn = paper_numbers(files, ids)
    for s in gt:
        for j in range(3):
            for col in ("theta", "theta_mc_se"):
                pn[f"ground_truth.json|scenario={s}.intervention={j + 1}|{col}"] = {
                    "value": gt[s][col][j], "file": "results/ground_truth.json", "row": f'["{s}"]["{col}"][{j}]', "column": col}
    (res / "paper_numbers.json").write_text(json.dumps(pn, indent=1), encoding="utf-8")
    R = {int(k): int(v) for k, v in cfg["experiments"]["R"].items()}
    (res / "SUMMARY.md").write_text(summary_md(cells, hyp, mu, viol, m1b, R, gt), encoding="utf-8")
    return ["table_main.csv", "table_scenarios.csv", "hypotheses.csv", "table_mu.csv", "table_violations.csv", "fig_rmse_vs_N_1col.png",
            "m1_bias_200k.csv", "fig_rmse_vs_N.pdf", "fig_rmse_vs_N.png", "fig_coverage.pdf", "fig_coverage.png",
            "paper_numbers.json", "SUMMARY.md"]
