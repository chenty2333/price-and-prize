"""Independent check of CEGAR refutation certificates (no MIP, no big-M).

A certificate is a small set C of allocations. It refutes a profile if
(1) every allocation in C is subjectively SD-EF1 (checked with fd, i.e.
    code independent of the ILP oracle), and
(2) no market valuation v (nonincreasing, v[0] = 1) makes every allocation
    in C violate market EF1 by a positive margin.
(2) is checked by depth-first search over the choice of violated pair for
each allocation. Each node solves one LP, maximising the common margin eps;
a node is pruned once eps <= tol, and a leaf with eps > tol would be a
counterexample.

Profiles checked: the hard examples of n3_search.py (m = 10..12), the
hardest profile per worker of n3_cegar_search.py (m = 13..21), and the
re-decided profiles (redecide_undecided.py).
"""
import glob
import json
import time

import numpy as np
from scipy.optimize import linprog

import cegar
import fd


def lp_margin(rows, m):
    # variables v[0..m-1], eps; maximise eps
    nv = m + 1
    A, b = [], []
    for t in range(m - 1):  # v[t+1] - v[t] <= 0
        r = np.zeros(nv); r[t + 1] = 1; r[t] = -1; A.append(r); b.append(0)
    for row in rows:  # eps - row.v <= 0
        r = np.zeros(nv); r[:m] = -row; r[m] = 1; A.append(r); b.append(0)
    bounds = [(1, 1)] + [(0, 1)] * (m - 1) + [(-(m + 1), m + 1)]
    c = np.zeros(nv); c[m] = -1
    res = linprog(c, A_ub=np.array(A), b_ub=b, bounds=bounds, method="highs")
    assert res.status == 0, res.message
    return -res.fun


def refutes(C, n, m, tol=1e-7):
    opts = [cegar.violation_rows(np.array(a, dtype=np.int8), n, m) for a in C]
    order = sorted(range(len(C)), key=lambda k: len(opts[k]))
    nodes = 0

    def dfs(d, chosen):
        nonlocal nodes
        nodes += 1
        if chosen and lp_margin(chosen, m) <= tol:
            return True  # pruned: this branch cannot break all of C
        if d == len(order):
            return False  # all of C broken with positive margin
        return all(dfs(d + 1, chosen + [row]) for row in opts[order[d]])

    return dfs(0, []), nodes


def main():
    profiles = []
    for mk, rec in json.load(open("../results/n3_search.json")).items():
        profiles += [(int(mk), ex["sigmas"]) for ex in rec["hard_examples"]]
    for f in sorted(glob.glob("../results/n3_cegar_search_w*.json")):
        rec = json.load(open(f))
        if rec.get("hardest"):
            profiles.append((rec["m"], rec["hardest"]["sigmas"]))
    for r in json.load(open("../results/n3_redecided.json")):
        profiles.append((r["m"], r["sigmas"]))
    out = []
    for m, s in profiles:
        t0 = time.time()
        r = cegar.decide(3, m, s, rng=np.random.default_rng(1))
        assert r["result"] == "refuted"
        C = r["certificate"]
        inS = bool(fd.subjective_sd_ef1(np.array(C, dtype=np.int8), s).all())
        ok, nodes = refutes(C, 3, m)
        out.append({"m": m, "cert_size": len(C), "all_in_S": inS, "lp_refutes": ok,
                    "lp_nodes": nodes, "sec": round(time.time() - t0, 1)})
        print(out[-1], flush=True)
        assert inS and ok
    json.dump(out, open("../results/verify_certificates.json", "w"), indent=1)
    print(f"verified {len(out)} certificates independently")


if __name__ == "__main__":
    main()
