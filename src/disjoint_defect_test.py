"""Test the disjoint-defect conjecture (n = 3): there exist subjectively
SD-EF1 allocations A, B, each with market-prefix counts within one at every
prefix length not divisible by 3, and at every boundary t = 3q at least one of
A, B has all counts equal to q. Feasibility ILP (own implementation). A
feasible pair is re-checked: subjective SD-EF1 with fd, and the two-cover
property with the LP-only search (verify_certificates.refutes)."""
import json, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
import cegar, fd
from two_cover_test import corpus
from verify_certificates import refutes

N = 3

def pair_ilp(m, s, time_limit=1200):
    nx = N * m; Q = m // 3
    X = lambda w, i, g: w * nx + i * m + g
    Z = lambda q: 2 * nx + q
    nv = 2 * nx + Q
    rows, lo, hi = [], [], []
    def add(c, l, h): rows.append(c); lo.append(l); hi.append(h)
    for w in (0, 1):
        for g in range(m):
            add({X(w, i, g): 1 for i in range(N)}, 1, 1)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                c = {}
                for g in s[i]:
                    c[X(w, i, g)] = c.get(X(w, i, g), 0) + 1
                    c[X(w, j, g)] = c.get(X(w, j, g), 0) - 1
                    add(dict(c), -1, np.inf)
        for t in range(1, m + 1):
            for i in range(N):
                c = {X(w, i, g): 1 for g in range(t)}
                if t % 3:
                    add(c, t // 3, -(-t // 3))
                else:
                    q = t // 3
                    zc = dict(c); zc[Z(q - 1)] = 1 if w == 0 else -1
                    # w=0: a_i <= q + z  and a_i >= q - z ; w=1 uses (1-z)
                    if w == 0:
                        add({**c, Z(q - 1): -1}, -np.inf, q)
                        add({**c, Z(q - 1): 1}, q, np.inf)
                    else:
                        add({**c, Z(q - 1): 1}, -np.inf, q + 1)
                        add({**c, Z(q - 1): -1}, q - 1, np.inf)
    A = lil_matrix((len(rows), nv))
    for r, c in enumerate(rows):
        for k, v in c.items():
            if v: A[r, k] = v
    res = milp(np.zeros(nv), constraints=LinearConstraint(A.tocsr(), lo, hi),
               integrality=np.ones(nv), bounds=Bounds(np.zeros(nv), np.ones(nv)),
               options={"time_limit": time_limit})
    if res.status == 2:
        return "infeasible", None
    if res.x is None:
        return f"unknown(status {res.status})", None
    x = np.round(res.x[:2 * nx]).reshape(2, N, m)
    return "feasible", [x[w].argmax(axis=0).astype(np.int8) for w in (0, 1)]

def work(args):
    k, (m, s, src) = args
    t0 = time.time()
    st, pair = pair_ilp(m, s)
    out = {"k": k, "m": m, "src": src, "status": st}
    if pair is not None:
        out["sd_ef1"] = bool(fd.subjective_sd_ef1(np.array(pair), s).all())
        out["two_cover_lp"] = refutes([pair[0].tolist(), pair[1].tolist()], N, m)[0]
    out["sec"] = round(time.time() - t0)
    return out

if __name__ == "__main__":
    P = corpus(); res = []
    with Pool(14) as p:
        for r in p.imap_unordered(work, list(enumerate(P))):
            res.append(r); print(r, flush=True)
            json.dump(res, open("../results/disjoint_defect_test.json", "w"), indent=1)
