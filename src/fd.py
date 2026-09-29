"""Core definitions for Open Question 1 of Barman, Ebadian, Latifian, Shah,
"Fair Division with Market Values" (AAAI 2025, arXiv 2410.23137).

OQ1: does there always exist an allocation that is SD-EF1 w.r.t. the agents'
subjective utilities and EF1 w.r.t. the additive market valuation v?

Conventions
- Items 0..m-1 are indexed in market order, v[0] >= v[1] >= ... >= v[m-1] >= 0.
- A ranking sigma is a permutation of range(m), most preferred first. Strict
  rankings are the hardest case: SD-EF1 w.r.t. a strict refinement implies
  SD-EF1 w.r.t. the tied ranking, so counterexamples may assume strictness.
- An allocation is a length-m vector a with a[g] = agent receiving g.

Facts used (prefix characterization, Section 2 of the paper):
- Agent i is SD-EF1 towards j iff, along sigma_i, every prefix satisfies
  #(items of i) - #(items of j) >= -1.
- An allocation is EF1 w.r.t. every nonincreasing v iff it is SD-EF1 w.r.t.
  the market ranking, iff each block {n*l, ..., n*l+n-1} of consecutive items
  gives at most one item to each agent (Lemma 1 of the paper).
"""
import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def all_allocations(n, m):
    """All n^m allocations as an (n^m, m) int8 array."""
    return np.array(list(itertools.product(range(n), repeat=m)), dtype=np.int8)


def market_sd_ef1_allocations(n, m):
    """Allocations giving each agent at most one item of every market block."""
    blocks = [list(range(s, min(s + n, m))) for s in range(0, m, n)]
    per_block = [list(itertools.permutations(range(n), len(b))) for b in blocks]
    out = []
    for choice in itertools.product(*per_block):
        out.append([ag for blk in choice for ag in blk])
    return np.array(out, dtype=np.int8)


def sd_ef1_pair(alloc, i, j, order):
    """Boolean mask: agent i (ranking `order`) is SD-EF1 towards agent j."""
    x = alloc[:, order]
    d = np.cumsum((x == i).astype(np.int16) - (x == j).astype(np.int16), axis=1)
    return d.min(axis=1) >= -1


def subjective_sd_ef1(alloc, sigmas):
    n = len(sigmas)
    ok = np.ones(len(alloc), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                ok &= sd_ef1_pair(alloc, i, j, sigmas[i])
    return ok


def market_ef1(alloc, v, n, tol=1e-12):
    """Boolean mask: allocation is EF1 w.r.t. the common additive valuation v."""
    v = np.asarray(v, dtype=float)
    val = np.stack([(alloc == k) @ v for k in range(n)], axis=1)
    top = np.zeros_like(val)
    for k in range(n):
        has = (alloc == k)
        first = has.argmax(axis=1)
        top[:, k] = np.where(has.any(axis=1), v[first], 0.0)
    ok = np.ones(len(alloc), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                ok &= val[:, i] >= val[:, j] - top[:, j] - tol
    return ok


def _milp_break_all(S, n, m):
    """max eps s.t. v nonincreasing in [0,1] and every allocation in S has some
    ordered pair (i,j) with v(A_j) - max_{g in A_j} v(g) - v(A_i) >= eps."""
    S = np.asarray(S)
    terms = []  # (alloc index, coefficient row over v)
    for a_idx, a in enumerate(S):
        for i in range(n):
            for j in range(n):
                if i == j or not (a == j).any():
                    continue
                row = (a == j).astype(float) - (a == i).astype(float)
                row[int(np.argmax(a == j))] -= 1.0
                terms.append((a_idx, row))
    K = len(terms)
    nv = m + 1 + K
    big = m + 1.0
    A, lo, hi = [], [], []
    for t in range(m - 1):
        r = np.zeros(nv); r[t] = 1; r[t + 1] = -1
        A.append(r); lo.append(0.0); hi.append(np.inf)
    by_alloc = {}
    for k, (a_idx, row) in enumerate(terms):
        r = np.zeros(nv); r[:m] = row; r[m] = -1; r[m + 1 + k] = -big
        A.append(r); lo.append(-big); hi.append(np.inf)
        by_alloc.setdefault(a_idx, []).append(k)
    for a_idx in range(len(S)):
        r = np.zeros(nv)
        for k in by_alloc.get(a_idx, []):
            r[m + 1 + k] = 1
        A.append(r); lo.append(1.0); hi.append(np.inf)
    c = np.zeros(nv); c[m] = -1.0
    integ = np.r_[np.zeros(m + 1), np.ones(K)]
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi),
               integrality=integ, bounds=Bounds(np.zeros(nv), np.ones(nv)))
    if res.status != 0:
        return 0.0, None
    return -res.fun, res.x[:m]


def find_breaking_v(S, n, m, max_rounds=500, add_per_round=25, rng=None):
    """Cutting-plane search for a market valuation under which every allocation
    in S (all subjective-SD-EF1 allocations) violates market EF1.

    Returns (eps, v, rounds). eps > 0 with v means a counterexample to OQ1;
    eps == 0 means none exists for this S (exact up to MILP tolerance).
    """
    rng = rng or np.random.default_rng(0)
    S = np.asarray(S)
    active = set()
    for v0 in [np.ones(m)] + [np.sort(rng.random(m))[::-1] for _ in range(5)]:
        surv = np.nonzero(market_ef1(S, v0, n))[0]
        active.update(rng.permutation(surv)[:add_per_round].tolist())
    for rounds in range(1, max_rounds + 1):
        idx = sorted(active)
        eps, v = _milp_break_all(S[idx], n, m)
        if eps <= 1e-9:
            return 0.0, None, rounds
        surv = np.nonzero(market_ef1(S, v, n, tol=1e-9))[0]
        if len(surv) == 0:
            return eps, v, rounds
        active.update(rng.permutation(surv)[:add_per_round].tolist())
    raise RuntimeError("cutting-plane loop did not converge")
