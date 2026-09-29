"""Conjecture B: if the market ranking equals agent 1's ranking, some allocation
is SD-EF1 w.r.t. every agent's own ranking AND market-SD-EF1 (each block of n
consecutive market goods split one per agent), hence market EF1 for every v.

(1) exhaustive for n = 3, m <= 8: sigma_1 = identity, sigma_2, sigma_3 all pairs
    (bitmask over market-SD-EF1 allocations, as in exhaustive.py);
(2) n = 4 and n = 5: hill-climb (sigma_2..sigma_n) towards a violation.
"""
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import fd


def exhaustive(m, n=3):
    perms = list(itertools.permutations(range(m)))
    Mal = fd.market_sd_ef1_allocations(n, m)
    ident = list(range(m))
    ok1 = np.ones(len(Mal), bool)
    for j in range(1, n):
        ok1 &= fd.sd_ef1_pair(Mal, 0, j, ident)
    b1 = np.array([np.packbits(ok1 & fd.sd_ef1_pair(Mal, 1, 0, list(p)) & fd.sd_ef1_pair(Mal, 1, 2, list(p))) for p in perms])
    b2 = np.array([np.packbits(fd.sd_ef1_pair(Mal, 2, 0, list(p)) & fd.sd_ef1_pair(Mal, 2, 1, list(p))) for p in perms])
    bad = 0
    for i in range(len(perms)):
        bad += int((~((b1[i][None, :] & b2).any(axis=1))).sum())
    return {"n": n, "m": m, "profiles": len(perms) ** 2, "violations": bad}


def ls_worker(args):
    n, m, seed, budget = args
    rng = np.random.default_rng(seed)
    Mal = fd.market_sd_ef1_allocations(n, m)
    t0 = time.time(); best = None; restarts = 0
    while time.time() - t0 < budget:
        restarts += 1
        s = [list(range(m))] + [list(map(int, rng.permutation(m))) for _ in range(n - 1)]
        h = int(fd.subjective_sd_ef1(Mal, s).sum())
        for _ in range(3000):
            if h == 0 or time.time() - t0 > budget:
                break
            t = [list(x) for x in s]; i = 1 + rng.integers(n - 1)
            a, b = rng.choice(m, 2, replace=False); t[i][a], t[i][b] = t[i][b], t[i][a]
            h2 = int(fd.subjective_sd_ef1(Mal, t).sum())
            if h2 <= h or rng.random() < 0.01:
                s, h = t, h2
        best = h if best is None else min(best, h)
        if h == 0:
            return {"n": n, "m": m, "restarts": restarts, "min_h": 0, "violation": s}
    return {"n": n, "m": m, "restarts": restarts, "min_h": best, "violation": None}


if __name__ == "__main__":
    out = {"exhaustive": [], "local_search": []}
    for m in ((5, 6, 7, 8) if "--skip-exhaustive" not in sys.argv else ()):
        t = time.time(); r = exhaustive(m); r["sec"] = round(time.time() - t, 1)
        out["exhaustive"].append(r); print(r, flush=True)
    jobs = [(4, m, 40 + m + k, 300) for m in (8, 12, 16) for k in range(2)] + \
           [(5, m, 50 + m + k, 300) for m in (10, 12) for k in range(2)]
    with Pool(len(jobs)) as p:
        for r in p.map(ls_worker, jobs):
            out["local_search"].append(r); print({k: r[k] for k in ("n", "m", "restarts", "min_h")}, flush=True)
    if "--skip-exhaustive" in sys.argv:
        out["exhaustive"] = [{"n": 3, "m": m, "violations": 0, "profiles": p} for m, p in
                             ((5, 14400), (6, 518400), (7, 25401600), (8, 1625702400))]
    json.dump(out, open("../results/conjB_check.json", "w"), indent=1)
