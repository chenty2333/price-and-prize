"""n = 3: local search for counterexamples to OQ1 with m = 8..12 goods.

m <= 7 is settled exhaustively by exhaustive.py (no profile even passes the
necessary condition). For larger m the profile space is too large, so:

phase 1  hill-climb over (sigma_1, sigma_2, sigma_3) to drive
         h = #(market-SD-EF1 allocations that are subjectively SD-EF1) to 0
         (necessary for a counterexample);
phase 2  among h = 0 profiles, hill-climb to shrink |S| (fewer subjectively
         SD-EF1 allocations are easier to break with one market valuation);
check    run the cutting-plane MILP fd.find_breaking_v on each phase-2 result.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import fd

N = 3


def h_of(Mal, s):
    return int(fd.subjective_sd_ef1(Mal, s).sum())


def mutate(s, rng, m):
    t = [x[:] for x in s]
    r = rng.integers(N)
    a, b = rng.choice(m, 2, replace=False)
    t[r][a], t[r][b] = t[r][b], t[r][a]
    return t


def worker(args):
    m, seed, budget_s = args
    rng = np.random.default_rng(seed)
    Mal = fd.market_sd_ef1_allocations(N, m)
    alloc = fd.all_allocations(N, m)
    t0 = time.time()
    rec = {"m": m, "restarts": 0, "hard_found": 0, "milp_checked": 0, "best_eps": 0.0,
           "min_S": None, "S_sizes": [], "cex": [], "hard_examples": []}
    while time.time() - t0 < budget_s:
        rec["restarts"] += 1
        s = [list(rng.permutation(m)) for _ in range(N)]
        h = h_of(Mal, s)
        for _ in range(6000):
            if h == 0:
                break
            t = mutate(s, rng, m)
            h2 = h_of(Mal, t)
            if h2 <= h or rng.random() < 0.01:
                s, h = t, h2
        if h != 0:
            continue
        rec["hard_found"] += 1
        S = fd.subjective_sd_ef1(alloc, s)
        size = int(S.sum())
        for _ in range(150):  # phase 2: shrink S while keeping h = 0
            t = mutate(s, rng, m)
            if h_of(Mal, t) != 0:
                continue
            S2 = fd.subjective_sd_ef1(alloc, t)
            if int(S2.sum()) <= size:
                s, S, size = t, S2, int(S2.sum())
        eps, v, rounds = fd.find_breaking_v(alloc[S], N, m, rng=rng)
        rec["milp_checked"] += 1
        rec["best_eps"] = max(rec["best_eps"], eps)
        rec["S_sizes"].append(size)
        rec["min_S"] = size if rec["min_S"] is None else min(rec["min_S"], size)
        if len(rec["hard_examples"]) < 3:
            rec["hard_examples"].append({"sigmas": [list(map(int, x)) for x in s], "S": size})
        if eps > 1e-9:
            rec["cex"].append({"sigmas": [list(map(int, x)) for x in s], "v": v.tolist(), "eps": eps})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--ms", default="8,9,10,11,12")
    ap.add_argument("--per_m", type=int, default=3)
    args = ap.parse_args()
    ms = [int(x) for x in args.ms.split(",")]
    jobs = [(m, 7000 * m + k, args.budget) for m in ms for k in range(args.per_m)]
    t0 = time.time()
    with Pool(min(len(jobs), os.cpu_count())) as pool:
        res = pool.map(worker, jobs)
    agg = {}
    for r in res:
        a = agg.setdefault(r["m"], {"restarts": 0, "hard_found": 0, "milp_checked": 0,
                                    "best_eps": 0.0, "S_sizes": [], "cex": [], "hard_examples": []})
        for f in ("restarts", "hard_found", "milp_checked"):
            a[f] += r[f]
        a["best_eps"] = max(a["best_eps"], r["best_eps"])
        a["S_sizes"] += r["S_sizes"]
        a["cex"] += r["cex"]
        a["hard_examples"] += r["hard_examples"]
    for m in sorted(agg):
        a = agg[m]
        sz = a["S_sizes"]
        print(f"m={m}: restarts={a['restarts']} hard={a['hard_found']} milp={a['milp_checked']} "
              f"min|S|={min(sz) if sz else None} median|S|={int(np.median(sz)) if sz else None} "
              f"best_eps={a['best_eps']:.3g} counterexamples={len(a['cex'])}", flush=True)
    with open("../results/n3_search.json", "w") as f:
        json.dump({str(k): v for k, v in agg.items()}, f, indent=1)
    print(f"done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
