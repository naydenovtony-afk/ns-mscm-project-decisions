"""Reproduce everything: python run_all.py --config config.yaml [--quick] [--stage ...] [--workers W].

Stages (in order; default all): truth (ground truth, Table II, DGP summary); rulecheck (R1-R3 in the DGP);
simulate (replications -> results/raw/, resumable); analyze (tables, figures, paper_numbers.json, SUMMARY.md).
--quick: 20,000 projects for truth/rulecheck, 2 replications of N = 300 into --outdir; analyze is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import csv  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

from nsmscm import dgp, params_table, rulecheck, truth  # noqa: E402
from nsmscm.experiment import one_rep  # noqa: E402

RAW_FIELDS = ["scenario", "N", "rep", "method", "intervention", "estimate", "se", "runtime_s", "rep_seconds",
              "share_e_clipped", "epochs_mean", "violations", "mu_chosen"]

RES = Path("results")


def _log(entry: dict) -> None:
    p = RES / "run_log.json"
    log = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"runs": []}
    log["runs"].append(entry)
    p.write_text(json.dumps(log, indent=2), encoding="utf-8")


def design_hash(cfg: dict) -> str:
    """SHA-256 of the DGP, scenario and truth sections (stored with the truth; checked by the tests)."""
    key = {k: cfg[k] for k in ("dgp", "scenarios", "truth")}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def stage_truth(cfg: dict, quick: bool) -> dict:
    sfx = "_quick" if quick else ""
    n = 20000 if quick else cfg["truth"]["n_projects"]
    seed = cfg["seeds"]["truth"]
    params_table.write(cfg, str(RES / "table_dgp_parameters.csv"))
    gt, summ = {}, {}
    for s in truth.SCENARIOS:
        t0 = time.time()
        gt[s] = truth.scenario_truth(cfg, s, seed, n)
        gt[s]["seconds"] = round(time.time() - t0, 1)
        gt[s]["design_sha256"] = design_hash(cfg)
        summ[s] = truth.summary_stats(cfg, s, seed, n)
        print(f"{s}: theta = {np.round(gt[s]['theta'], 5).tolist()}  ({gt[s]['seconds']} s)", flush=True)
    (RES / f"ground_truth{sfx}.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    (RES / f"dgp_summary{sfx}.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    return {"truth_seed": seed, "truth_n": n, "chunk": cfg["truth"]["chunk"],
            "outputs": [f"ground_truth{sfx}.json", f"dgp_summary{sfx}.json", "table_dgp_parameters.csv"]}


def stage_rulecheck(cfg: dict, quick: bool) -> dict:
    """R1-R3 in the DGP, binned conditional means on seeds.tests portfolios (same checks as tests/)."""
    n = 20000 if quick else cfg["truth"]["n_projects"]
    seed = cfg["seeds"]["tests"]
    out = {}
    for s in truth.SCENARIOS:
        kw = dict(shifted=s == "S2")
        obs = dgp.simulate_chunked(cfg, s, n, seed, cfg["truth"]["chunk"], **kw)
        zero = dgp.simulate_chunked(cfg, s, n, seed, cfg["truth"]["chunk"], force_all=0.0, **kw)
        out[s] = rulecheck.all_checks(obs, zero)
    name = f"dgp_rule_checks{'_quick' if quick else ''}.json"
    (RES / name).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    return {"rulecheck_seed": seed, "rulecheck_n": n, "outputs": [name]}


def _job(cfg, s, n, r):
    return s, n, r, one_rep(cfg, s, n, r)


def stage_simulate(cfg: dict, quick: bool, workers: int, reps: int | None, outdir: str, resume: bool) -> dict:
    """Replications of every cell; rows appended per finished job to <outdir>/<S>_N<N>.csv."""
    Ns = [300] if quick else cfg["experiments"]["N"]
    Rn = {n: (2 if quick else int(cfg["experiments"]["R"][n])) for n in Ns}
    if reps is not None:
        Rn = {n: min(r, reps) for n, r in Rn.items()}
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in out.glob("*.csv"):
        if not resume:
            raise SystemExit(f"{f} exists; use --resume or another --outdir")
        with open(f, newline="", encoding="utf-8") as fh:
            done |= {(r["scenario"], int(r["N"]), int(r["rep"])) for r in csv.DictReader(fh)}
    jobs = [(s, n, r) for n in sorted(Ns, reverse=True) for s in cfg["experiments"]["scenarios"] for r in range(Rn[n])
            if (s, n, r) not in done]
    print(f"{len(jobs)} jobs ({len(done)} already done), {workers} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, cfg, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            s, n, r, rows = f.result()
            path = out / f"{s}_N{n}.csv"
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
                if new:
                    w.writeheader()
                for row in rows:
                    w.writerow({k: (json.dumps(row[k]) if isinstance(row.get(k), (dict, list)) else row.get(k, ""))
                                for k in RAW_FIELDS})
            print(f"  {i}/{len(jobs)} {s} N={n} rep {r}: {rows[0]['rep_seconds']} s  (elapsed {time.time() - t0:.0f} s)",
                  flush=True)
    return {"replications": Rn, "N": Ns, "workers": workers, "outdir": outdir, "jobs_run": len(jobs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", default="all", choices=["all", "truth", "rulecheck", "simulate", "analyze"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--outdir", default="results/raw")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    stages = ["truth", "rulecheck", "simulate", "analyze"] if a.stage == "all" else [a.stage]
    info = {}
    for st in stages:
        if st == "simulate":
            info[st] = stage_simulate(cfg, a.quick, a.workers, a.reps, a.outdir, a.resume or a.stage == "all")
        elif st == "analyze":
            if a.quick:
                info[st] = {"skipped": "analyze needs the full replications"}
                continue
            from nsmscm import analysis
            info[st] = {"outputs": analysis.run(cfg)}
        else:
            info[st] = {"truth": stage_truth, "rulecheck": stage_rulecheck}[st](cfg, a.quick)
    _log({"stage": a.stage, "quick": a.quick, "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
          "seconds": round(time.time() - t0, 1), "python": sys.version.split()[0],
          "platform": platform.platform(), "numpy": np.__version__,
          "seeds": cfg["seeds"], "config": a.config, "stages": info})
    return 0


if __name__ == "__main__":
    sys.exit(main())
