"""M3 (DML, MLP nuisances), M4 (NS-DML, MLP + rule penalties) and M5 (DML, HistGradientBoosting).
Settings come from config.yaml["estimators"] (frozen with the design).

Nuisance functions per intervention j (fitted on the K-1 training folds, standardized there):
  e_j(s) = P(A_j = 1 | s); l_j0 on untreated, l_j1 on treated projects.
AIPW score: psi = l1 - l0 + a (y - l1)/e - (1 - a)(y - l0)/(1 - e), e clipped; theta = mean psi,
SE = sd(psi)/sqrt(n).
M4 penalties (c_k <= 0 is the rule), derivatives w.r.t. the standardized inputs:
  R1 on l_j0: c = -l_j0(s) and c = -dl_j0/dnovelty
  R2 on e_2:  c = de_2/dpi along the data manifold at fixed CPI, i.e. (in per-sd-of-pi units)
              de/dz_pi + (s_pi / (CPI s_b)) de/dz_b    (b = pi/CPI)
  R3 on l_j0 and l_j1: c = dl/dCPI
Nuisances without a rule (e_1, e_3) are the M3 networks (identical seeds), so M4 = M3 for them.
M3 and M4 share folds, the train/validation split, initialisation and minibatch order.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .estimators import COVARIATES

I_PI, I_CPI, I_B, I_NOV = (COVARIATES.index(c) for c in ("pi", "CPI", "b", "novelty"))
CONT = slice(4, len(COVARIATES))
RULES = {("l", 0): ["R1_level", "R1_novelty", "R3"], ("l", 1): ["R3"], ("e", 2): ["R2"]}


def rules_for(kind: str, j: int, a: int | None) -> list[str]:
    if kind == "l":
        return RULES[("l", a)]
    return RULES.get(("e", j + 1), [])


class Std:
    def __init__(self, X):
        self.m = X.mean(0)
        self.s = X.std(0)
        self.s[self.s == 0] = 1.0

    def __call__(self, X):
        return (X - self.m) / self.s


class MLP(torch.nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        layers, k = [], d
        for h in hidden:
            layers += [torch.nn.Linear(k, h), torch.nn.ReLU()]
            k = h
        layers.append(torch.nn.Linear(k, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def constraints(model, Z, kind, rules, std):
    """List of c_k tensors (one per rule component) at standardized states Z."""
    Z = Z.detach().clone().requires_grad_(True)
    out = model(Z)
    if kind == "e":
        out = torch.sigmoid(out)
    cs = []
    need_grad = any(r != "R1_level" for r in rules)
    g = torch.autograd.grad(out.sum(), Z, create_graph=True)[0] if need_grad else None
    for r in rules:
        if r == "R1_level":
            cs.append(-out)
        elif r == "R1_novelty":
            cs.append(-g[:, I_NOV])
        elif r == "R3":
            cs.append(g[:, I_CPI])
        elif r == "R2":
            cpi = Z[:, I_CPI].detach() * float(std.s[I_CPI]) + float(std.m[I_CPI])
            cs.append(g[:, I_PI] + (float(std.s[I_PI]) / (cpi * float(std.s[I_B]))) * g[:, I_B])
    return cs


def fit_mlp(Z, y, kind, rules, mu, mcfg, ncfg, seed, std):
    """Train one network; returns (model, best validation data loss, epochs run)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    nv = max(1, int(round(mcfg["validation_share"] * n)))
    va, tr = perm[:nv], perm[nv:]
    Zt = torch.tensor(Z, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    Zp = Z[tr].copy()
    Zp[:, CONT] += ncfg["perturbation_sd"] * rng.standard_normal(Zp[:, CONT].shape)
    Zp = torch.tensor(Zp, dtype=torch.float32)
    model = MLP(Z.shape[1], mcfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=mcfg["learning_rate"], weight_decay=mcfg["weight_decay"])
    lossf = torch.nn.MSELoss() if kind == "l" else torch.nn.BCEWithLogitsLoss()
    best, best_state, wait, ep = np.inf, None, 0, 0
    bs = mcfg["batch_size"]
    pen = bool(rules) and mu > 0
    for ep in range(1, mcfg["max_epochs"] + 1):
        model.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(tr), bs):
            bi = order[i:i + bs]
            idx = tr[bi]
            loss = lossf(model(Zt[idx]), yt[idx])
            if pen:
                states = torch.cat([Zt[idx], Zp[bi]])
                loss = loss + mu * sum(torch.relu(c).mean() for c in constraints(model, states, kind, rules, std))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(Zt[va]), yt[va]))
        if vl < best - 1e-12:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= mcfg["early_stopping_patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, ep


def predict(model, Z, kind):
    with torch.no_grad():
        out = model(torch.tensor(Z, dtype=torch.float32))
        return (torch.sigmoid(out) if kind == "e" else out).numpy().astype(float)


def violation_counts(model, Z, kind, rules, std):
    cs = constraints(model, torch.tensor(Z, dtype=torch.float32), kind, rules, std)
    return {r: int((c.detach().numpy() > 0).sum()) for r, c in zip(rules, cs)}


def _aipw(y, a, e, l0, l1, clip):
    e = np.clip(e, clip[0], clip[1])
    return l1 - l0 + a * (y - l1) / e - (1 - a) * (y - l0) / (1 - e)


def _seed(base, *k):
    return int(np.random.SeedSequence([*base, *k]).generate_state(1)[0])


def dml_all(X, A, y, cfg, seed_base, X_eval=None, A_eval=None, y_eval=None):
    """Run M3, M4, M5 for the three interventions. X/A/y: fitting portfolio. If X_eval is given (S2),
    the scores are evaluated on it (part k scored by the fold-k models); otherwise cross-fitting.
    seed_base: tuple of ints. Returns list of result rows."""
    ec = cfg["estimators"]
    mcfg, ncfg, hcfg = ec["mlp"], ec["ns"], ec["hgb"]
    clip = ec["propensity_clip"]
    K = ec["folds"]
    torch.set_num_threads(mcfg["torch_threads"])
    n = len(y)
    folds = np.random.default_rng(_seed(seed_base, 1)).permutation(n) % K
    if X_eval is None:
        Xe, Ae, ye, efold = X, A, y, folds
    else:
        Xe, Ae, ye = X_eval, A_eval, y_eval
        efold = np.random.default_rng(_seed(seed_base, 2)).permutation(len(ye)) % K
    ne = len(ye)
    rows = []
    for j in range(3):
        a_fit, a_ev = A[:, j], Ae[:, j]
        pred = {m: {k: np.zeros(ne) for k in ("e", "l0", "l1")} for m in ("M3", "M4", "M5")}
        tsec = {"M3": 0.0, "M4": 0.0, "M5": 0.0}
        viol = {m: {} for m in ("M3", "M4")}
        mus, epochs = [], {"M3": [], "M4": []}
        for k in range(K):
            trn, ev = folds != k, efold == k
            std = Std(X[trn])
            Ztr, Zev = std(X[trn]), std(Xe[ev])
            nets = [("e", None, np.ones(trn.sum(), bool)), ("l", 0, a_fit[trn] == 0), ("l", 1, a_fit[trn] == 1)]
            for kind, a, sel in nets:
                key = "e" if kind == "e" else f"l{a}"
                target = a_fit[trn] if kind == "e" else y[trn][sel]
                Zs = Ztr if kind == "e" else Ztr[sel]
                rules = rules_for(kind, j, a)
                sd = _seed(seed_base, 10, j, k, 0 if kind == "e" else 1 + a)
                t0 = time.perf_counter()
                m3, _, ep3 = fit_mlp(Zs, target.astype(float), kind, [], 0.0, mcfg, ncfg, sd, std)
                t3 = time.perf_counter() - t0
                tsec["M3"] += t3
                epochs["M3"].append(ep3)
                pred["M3"][key][ev] = predict(m3, Zev, kind)
                if rules:
                    t0 = time.perf_counter()
                    fits = [fit_mlp(Zs, target.astype(float), kind, rules, mu, mcfg, ncfg, sd, std) for mu in ncfg["mu_grid"]]
                    tsec["M4"] += time.perf_counter() - t0
                    i = int(np.argmin([f[1] for f in fits]))
                    m4 = fits[i][0]
                    mus.append({"fold": k, "nuisance": key, "mu": ncfg["mu_grid"][i],
                                "val_loss": [round(f[1], 8) for f in fits]})
                    epochs["M4"].append(fits[i][2])
                    for name, mod in (("M3", m3), ("M4", m4)):
                        for r, c in violation_counts(mod, Zev, kind, rules, std).items():
                            v = viol[name].setdefault(f"{key}:{r}", [0, 0])
                            v[0] += c
                            v[1] += len(Zev)
                else:
                    tsec["M4"] += t3
                    m4 = m3
                    epochs["M4"].append(ep3)
                pred["M4"][key][ev] = predict(m4, Zev, kind)
            # M5
            t0 = time.perf_counter()
            hk = dict(max_iter=hcfg["max_iter"], learning_rate=hcfg["learning_rate"],
                      max_leaf_nodes=hcfg["max_leaf_nodes"], min_samples_leaf=hcfg["min_samples_leaf"],
                      early_stopping=hcfg["early_stopping"])
            Xtr = X[trn]
            pred["M5"]["e"][ev] = HistGradientBoostingClassifier(random_state=_seed(seed_base, 20, j, k, 0), **hk).fit(
                Xtr, a_fit[trn]).predict_proba(Xe[ev])[:, 1]
            for a in (0, 1):
                s = a_fit[trn] == a
                pred["M5"][f"l{a}"][ev] = HistGradientBoostingRegressor(random_state=_seed(seed_base, 20, j, k, 1 + a), **hk).fit(
                    Xtr[s], y[trn][s]).predict(Xe[ev])
            tsec["M5"] += time.perf_counter() - t0
        for m in ("M3", "M4", "M5"):
            p = pred[m]
            psi = _aipw(ye, a_ev, p["e"], p["l0"], p["l1"], clip)
            row = {"method": m, "intervention": j + 1, "estimate": float(psi.mean()),
                   "se": float(psi.std(ddof=1) / np.sqrt(ne)), "runtime_s": round(tsec[m], 3),
                   "share_e_clipped": float(np.mean((p["e"] < clip[0]) | (p["e"] > clip[1])))}
            if m in viol:
                row["violations"] = {k: v[0] / v[1] for k, v in viol[m].items()}
                row["epochs_mean"] = float(np.mean(epochs[m]))
            if m == "M4":
                row["mu_chosen"] = mus
            rows.append(row)
    return rows
