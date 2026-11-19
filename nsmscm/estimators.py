"""Estimators (prompt Section 4). Step 3 implements the covariates, M1 and M2; M3-M5 follow in Step 4."""
from __future__ import annotations

import numpy as np
from scipy import stats

MODE_COLS = ["mode_on_track", "mode_at_risk", "mode_delayed", "mode_critical"]   # "completed" never occurs at tau
CONT_COLS = ["pi", "CPI", "SPI", "b", "size", "novelty", "experience"]
COVARIATES = MODE_COLS + CONT_COLS


def covariates(d: dict) -> np.ndarray:
    """State at tau: one-hot health mode (4 non-completed modes), pi, CPI, SPI, b, size, novelty, experience."""
    mode = d["mode"].astype(int)
    oh = np.stack([(mode == k).astype(float) for k in range(4)], axis=1)
    cont = np.column_stack([d["gate_pi"], d["CPI"], d["SPI"], d["b"], d["size"], d["novelty"], d["experience"]])
    return np.column_stack([oh, cont])


def m1_ols(y: np.ndarray, A: np.ndarray, X: np.ndarray) -> list[tuple[float, float]]:
    """M1: OLS of Y on A1, A2, A3 and the covariates (intercept; on_track is the reference mode), HC1 SEs.
    Returns [(estimate, se)] for A1..A3."""
    n = len(y)
    D = np.column_stack([np.ones(n), A, X[:, 1:]])       # drop mode_on_track (reference)
    XtXi = np.linalg.inv(D.T @ D)
    beta = XtXi @ D.T @ y
    e = y - D @ beta
    k = D.shape[1]
    V = XtXi @ (D.T * e ** 2) @ D @ XtXi * n / (n - k)
    return [(float(beta[1 + j]), float(np.sqrt(V[1 + j, 1 + j]))) for j in range(3)]


def m2_evm(y: np.ndarray, A: np.ndarray, cpi: np.ndarray) -> list[tuple[float, float]]:
    """M2: forecast error Y - (1/CPI_tau - 1); difference in means treated - untreated, Welch SE."""
    fe = y - (1 / cpi - 1)
    out = []
    for j in range(3):
        a, b = fe[A[:, j] == 1], fe[A[:, j] == 0]
        out.append((float(a.mean() - b.mean()), float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))))
    return out


def t_stat_p(est: float, se: float) -> float:
    return float(2 * stats.norm.sf(abs(est / se)))
