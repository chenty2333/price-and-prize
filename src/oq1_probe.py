"""Probe Open Question 1 of Barman-Ebadian-Latifian-Shah (AAAI'25), n = 2.

OQ1: is there always an allocation that is SD-EF1 w.r.t. subjective utilities
and EF1 w.r.t. the (additive) market valuation?

Items 0..m-1 are indexed in market order (v nonincreasing). Agent 1 gets mask A.
For n = 2, agent i is SD-EF1 iff every prefix of sigma_i has
(#own - #other) >= -1.

Search: exhaustive over (sigma1, sigma2) with sigma1 index <= sigma2 index (agent
swap symmetry). Step 1 keeps only pairs where no subjective-SD-EF1 partition is
also SD-EF1 w.r.t. the market ranking (otherwise that partition is EF1 for every
consistent v, so no counterexample). Step 2 solves a MILP over nonincreasing v in
[0,1] that makes every surviving partition violate market EF1 by margin eps.
"""
import itertools
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

m = int(sys.argv[1])
FULL = (1 << m) - 1
masks = np.arange(1 << m)
bits = ((masks[:, None] >> np.arange(m)) & 1).astype(np.int8)  # [mask, item]
sign = 2 * bits - 1  # +1 agent 1, -1 agent 2

perms = list(itertools.permutations(range(m)))
P = len(perms)


def ok_sets(order):
    pref = np.cumsum(sign[:, list(order)], axis=1)
    return pref.min(axis=1) >= -1, pref.max(axis=1) <= 1  # agent1 ok, agent2 ok


def pack(boolvec):
    return np.packbits(boolvec.astype(np.uint8))


P1 = np.empty((P, (1 << m) // 8), np.uint8)
P2 = np.empty_like(P1)
for k, o in enumerate(perms):
    a, b = ok_sets(o)
    P1[k], P2[k] = pack(a), pack(b)

# market SD-EF1 (Lemma 1, n = 2): at most one item from each pair {2l, 2l+1}
mk = np.ones(1 << m, bool)
for l in range(0, m - 1, 2):
    mk &= bits[:, l] != bits[:, l + 1]
MK = pack(mk)

# sanity: Theorem 1 instance must survive step 1
if m == 7:
    s1, s2 = (0, 2, 1, 4, 3, 6, 5), (0, 4, 1, 2, 5, 6, 3)
    a, _ = ok_sets(s1)
    _, b = ok_sets(s2)
    assert not (a & b & mk).any(), "Theorem 1 instance should have no triple SD-EF1 partition"
    print("sanity: Theorem 1 instance has no triple-SD-EF1 partition (as in the paper)")

cands = {}
for i in range(P):
    S = P1[i][None, :] & P2[i:]
    dead = ~((S & MK[None, :]).any(axis=1))
    for off in np.nonzero(dead)[0]:
        key = S[off].tobytes()
        cands.setdefault(key, (i, i + off))
    if i % 5000 == 0:
        print(f"  step1 {i}/{P}, distinct candidate S so far: {len(cands)}", flush=True)
print(f"m={m}: distinct S surviving step 1: {len(cands)}")


def solve(S_masks):
    K = len(S_masks)
    nv = m + 1 + K  # v, eps, z
    rows, lo, hi = [], [], []
    for t in range(m - 1):  # v_t >= v_{t+1}
        r = np.zeros(nv); r[t] = 1; r[t + 1] = -1
        rows.append(r); lo.append(0); hi.append(np.inf)
    M = m + 1
    for k, A in enumerate(S_masks):
        B = FULL ^ A
        assert A and B
        a = bits[A].astype(float); b = bits[B].astype(float)
        ta = (A & -A).bit_length() - 1; tb = (B & -B).bit_length() - 1
        r1 = np.zeros(nv); r1[:m] = b - a; r1[tb] -= 1; r1[m] = -1; r1[m + 1 + k] = M
        rows.append(r1); lo.append(0); hi.append(np.inf)
        r2 = np.zeros(nv); r2[:m] = a - b; r2[ta] -= 1; r2[m] = -1; r2[m + 1 + k] = -M
        rows.append(r2); lo.append(-M); hi.append(np.inf)
    c = np.zeros(nv); c[m] = -1
    integ = np.r_[np.zeros(m + 1), np.ones(K)]
    lb = np.zeros(nv); ub = np.ones(nv)
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=integ, bounds=Bounds(lb, ub))
    return res


def verify(o1, o2, v):
    a, _ = ok_sets(o1)
    _, b = ok_sets(o2)
    for A in np.nonzero(a & b)[0]:
        B = FULL ^ A
        vA = v[bits[A] == 1].sum(); vB = v[bits[B] == 1].sum()
        ta = (A & -A).bit_length() - 1; tb = (B & -B).bit_length() - 1
        if vA >= vB - v[tb] - 1e-12 and vB >= vA - v[ta] - 1e-12:
            return False  # this partition satisfies OQ1
    return True


best = 0.0
found = 0
for n_done, (key, (i, j)) in enumerate(cands.items()):
    S_masks = np.nonzero(np.unpackbits(np.frombuffer(key, np.uint8)))[0]
    res = solve(list(map(int, S_masks)))
    eps = -res.fun if res.status == 0 else float("nan")
    best = max(best, eps)
    if eps > 1e-6:
        v = res.x[:m]
        ok = verify(perms[i], perms[j], v)
        print(f"COUNTEREXAMPLE? sigma1={perms[i]} sigma2={perms[j]} v={np.round(v, 4)} eps={eps:.4g} verified={ok}")
        found += 1
        if found >= 3:
            break
    if n_done % 2000 == 0:
        print(f"  step2 {n_done}/{len(cands)} best eps so far {best:.3g}", flush=True)
print(f"m={m}: step2 done, best eps = {best:.3g}, counterexamples found = {found}")
