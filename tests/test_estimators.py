"""Step 4 test, run before any replication result: M3 and M5 agree within 2 SE at N = 5000 in S1.
SE = max(SE_M3, SE_M5) of the two estimates on the same portfolio (seeds.tests)."""
import json

import numpy as np
from conftest import ROOT

from nsmscm import dgp, dml, estimators


def test_m3_m5_agree_at_n5000_s1(cfg):
    d = dgp.portfolio(cfg, "S1", 5000, np.random.SeedSequence([cfg["seeds"]["tests"], 5000]))
    rows = dml.dml_all(estimators.covariates(d), d["A"], d["Y"], cfg, (cfg["seeds"]["tests"], 5000))
    est = {(r["method"], r["intervention"]): (r["estimate"], r["se"]) for r in rows}
    out = []
    for j in (1, 2, 3):
        (e3, s3), (e5, s5) = est[("M3", j)], est[("M5", j)]
        out.append({"intervention": j, "M3": [e3, s3], "M5": [e5, s5], "diff": e3 - e5, "diff_in_se": (e3 - e5) / max(s3, s5)})
    (ROOT / "results" / "test_m3_m5_agreement.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for o in out:
        assert abs(o["diff_in_se"]) <= 2, out
