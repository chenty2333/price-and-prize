"""Adversarial search against Lemma B (n = 2): is S & E1 ever empty?

S & E1 = partitions where, with x = +1/-1 walks,
  walk_sigma1 >= -1, walk_market >= -1  (agent 1 SD-EF1 w.r.t. two rankings)
  walk_sigma2 <= 1                        (agent 2 SD-EF1 w.r.t. sigma2)
Hill-climb over (sigma1, sigma2) (market order fixed) to minimize |S & E1|.
If it reaches 0, also test the valuation-dependent fallback: for every v there
should be X, Y in S with v(A1) >= v(A2) >= ... (LP check), and OQ1 itself.

usage: python n2_lemmaB_adversarial.py m_list seconds_per_m
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import fd


def count_SE1(alloc, s1, s2):
    ok = fd.sd_ef1_pair(alloc, 0, 1, s1) & fd.sd_ef1_pair(alloc, 0, 1, list(range(alloc.shape[1])))
    ok &= fd.sd_ef1_pair(alloc, 1, 0, s2)
    return int(ok.sum())


def worker(args):
    m, seed, budget = args
    rng = np.random.default_rng(seed)
    alloc = fd.all_allocations(2, m)
    best, best_s, restarts, zeros = None, None, 0, []
    t0 = time.time()
    while time.time() - t0 < budget:
        restarts += 1
        s = [list(rng.permutation(m)), list(rng.permutation(m))]
        c = count_SE1(alloc, *s)
        for _ in range(3000):
            if c == 0:
                break
            t = [x[:] for x in s]
            r = rng.integers(2)
            a, b = rng.choice(m, 2, replace=False)
            t[r][a], t[r][b] = t[r][b], t[r][a]
            c2 = count_SE1(alloc, *t)
            if c2 <= c or rng.random() < 0.01:
                s, c = t, c2
        if best is None or c < best:
            best, best_s = c, [list(map(int, x)) for x in s]
        if c == 0:
            zeros.append([list(map(int, x)) for x in s])
            if len(zeros) >= 3:
                break
    return {"m": m, "restarts": restarts, "min_count": best, "argmin": best_s, "zeros": zeros}


if __name__ == "__main__":
    ms = [int(a) for a in sys.argv[1].split(",")]
    budget = float(sys.argv[2])
    jobs = [(m, 99 * m + k, budget) for m in ms for k in range(2)]
    with Pool(len(jobs)) as p:
        res = p.map(worker, jobs)
    for r in res:
        print(r["m"], "restarts", r["restarts"], "min |S&E1|", r["min_count"], "zeros", len(r["zeros"]),
              "argmin", r["argmin"])
    json.dump(res, open("../results/n2_lemmaB_adversarial.json", "w"), indent=1)
