"""Calibrate dgp.cost_factor.a0 so that the mean overrun with all interventions set to 0 is 0.28 in S1.

Uses seeds.calibration (never the truth or estimation seeds), n = truth.n_projects, bisection on a0
(the mean is increasing in a0). Writes the rounded value (4 decimals) into config.yaml.
Run once at design time, before the design freeze.
"""
import sys

import numpy as np
import yaml

from nsmscm import dgp

TARGET = 0.28


def mean_y0(cfg, a0):
    cfg["dgp"]["cost_factor"]["a0"] = a0
    o = dgp.simulate_chunked(cfg, "S1", cfg["truth"]["n_projects"], cfg["seeds"]["calibration"],
                             cfg["truth"]["chunk"], force_all=0.0)
    return float(o["Y"].mean())


def main(path="config.yaml"):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    lo, hi = np.log(0.02), np.log(1.0)
    for _ in range(30):
        mid = (lo + hi) / 2
        if mean_y0(cfg, mid) < TARGET:
            lo = mid
        else:
            hi = mid
    a0 = round((lo + hi) / 2, 4)
    m = mean_y0(cfg, a0)
    print(f"a0 = {a0}  mean overrun (all A = 0, calibration seed) = {m:.5f}")
    s = open(path, encoding="utf-8").read()
    old = "    a0: null"
    if old not in s:
        sys.exit("a0 already set in config.yaml; not overwriting")
    s = s.replace(old, f"    a0: {a0}")
    open(path, "w", encoding="utf-8", newline="\n").write(s)


if __name__ == "__main__":
    main()
