import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsmscm import dgp  # noqa: E402

N_BIG = 200_000


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


@lru_cache(maxsize=None)
def _sim(scenario: str, seed: int, n: int, force_all=None):
    c = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    return dgp.simulate_chunked(c, scenario, n, seed, c["truth"]["chunk"], shifted=scenario == "S2",
                                force_all=force_all)


@pytest.fixture(scope="session")
def sim():
    """sim(scenario, n=200000, force_all=None): observational (or all-forced) portfolio from seeds.tests."""
    c = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    return lambda scenario, n=N_BIG, force_all=None: _sim(scenario, c["seeds"]["tests"], n, force_all)


def binned_means(y, keys):
    """Means and SEs of y by the tuple of integer bin keys. Returns dict key -> (mean, se, count)."""
    k = np.column_stack(keys)
    uniq, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.ravel()
    out = {}
    for i, u in enumerate(uniq):
        v = y[inv == i]
        out[tuple(int(x) for x in u)] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.inf, len(v))
    return out


def qbin(x, q):
    """Quantile bin index 0..q-1."""
    edges = np.quantile(x, np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, x, side="right")
