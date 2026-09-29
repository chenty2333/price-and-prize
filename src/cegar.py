"""CEGAR decision procedure for OQ1 at a fixed ranking profile, without
enumerating the set S of subjectively SD-EF1 allocations.

Question for fixed (sigma_1..sigma_n): is there a market valuation v
(nonincreasing, v[0] = 1 by scaling) such that EVERY allocation in S violates
market EF1?  A violation of pair (i, j) means
    v(A_j) - v(r_j(A)) - v(A_i) > 0,
where r_j(A) is the item removed from A_j: the most valuable item for EF1
(the first in market order), or the least valuable for EFX (the last).
EFX is supported only to validate that the pipeline does find counterexamples
(Barman et al. show SD-EF1 subjective + EFX market can be impossible).

master  : max eps over v, s.t. each allocation in the current finite set S'
          has some pair violated by >= eps (MILP, one binary per (A, pair)).
oracle  : given v, min over A in S of the largest pair violation
          (ILP over assignment variables; SD-EF1 as prefix constraints).
loop    : if master eps <= tol          -> no counterexample (S' certifies);
          if oracle value >= eps - tol   -> v breaks all of S: counterexample;
          else add the oracle's allocation to S'.

Because v[0] = 1 and v = e_0 makes every allocation exactly EF1-tight, the
master optimum is always >= 0; a counterexample needs it to stay > 0.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


class SolverError(RuntimeError):
    pass


def _milp(c, constraints, integrality, bounds, time_limit=300):
    """HiGHS occasionally reports an internal 'Solve error' on these MIPs.
    Retry with presolve off, then with rows and columns randomly permuted
    (same model, different solver path). Raise SolverError if all fail."""
    attempts = [({"time_limit": time_limit}, None), ({"time_limit": time_limit, "presolve": False}, None)]
    attempts += [({"time_limit": time_limit}, k) for k in range(3)]
    res = None
    for opts, perm_seed in attempts:
        if perm_seed is None:
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options=opts)
            x = res.x
        else:
            prng = np.random.default_rng(perm_seed)
            A = constraints.A.tocsr() if hasattr(constraints.A, "tocsr") else constraints.A
            rp = prng.permutation(A.shape[0]); cp = prng.permutation(A.shape[1])
            inv = np.argsort(cp)
            res = milp(c[cp], constraints=LinearConstraint(A[rp][:, cp], np.asarray(constraints.lb)[rp],
                                                           np.asarray(constraints.ub)[rp]),
                       integrality=np.asarray(integrality)[cp],
                       bounds=Bounds(np.asarray(bounds.lb)[cp], np.asarray(bounds.ub)[cp]), options=opts)
            x = None if res.x is None else res.x[inv]
        if x is not None and res.status == 0:
            res.x = x
            return res
    raise SolverError(f"HiGHS failed after {len(attempts)} attempts: {res.message}")


class Oracle:
    def __init__(self, n, m, sigmas, market="EF1"):
        self.n, self.m, self.market = n, m, market
        nx = n * m
        self.nvar = 2 * nx + 1  # x, f, z
        X = lambda i, g: i * m + g
        F = lambda i, g: nx + i * m + g
        self.X, self.F, self.Z = X, F, 2 * nx
        rows, lo, hi = [], [], []

        def add(coefs, l, h):
            rows.append(coefs); lo.append(l); hi.append(h)

        for g in range(m):  # each item assigned once
            add({X(i, g): 1 for i in range(n)}, 1, 1)
        for i in range(n):  # subjective SD-EF1 prefix constraints
            order = sigmas[i]
            for j in range(n):
                if j == i:
                    continue
                c = {}
                for g in order:
                    c[X(i, g)] = c.get(X(i, g), 0) + 1
                    c[X(j, g)] = c.get(X(j, g), 0) - 1
                    add(dict(c), -1, np.inf)
        for j in range(n):  # removed item r_j
            for g in range(m):
                add({F(j, g): 1, X(j, g): -1}, -np.inf, 0)
            if market == "EF1":
                add({F(j, g): 1 for g in range(m)}, -np.inf, 1)
            else:  # EFX: f picks the last item of A_j (exactly one if nonempty)
                for g in range(m):
                    for h in range(g + 1, m):
                        add({F(j, g): 1, X(j, h): 1}, -np.inf, 1)
                    add({F(j, gg): 1 for gg in range(m)} | {X(j, g): -1}, 0, np.inf)
                add({F(j, g): 1 for g in range(m)}, -np.inf, 1)
        self.base = (rows, lo, hi)
        self.pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    def solve(self, v):
        n, m = self.n, self.m
        rows, lo, hi = list(self.base[0]), list(self.base[1]), list(self.base[2])
        for i, j in self.pairs:  # z >= v(A_j) - v(r_j) - v(A_i)
            c = {self.Z: -1.0}
            for g in range(m):
                c[self.X(j, g)] = c.get(self.X(j, g), 0) + v[g]
                c[self.X(i, g)] = c.get(self.X(i, g), 0) - v[g]
                c[self.F(j, g)] = c.get(self.F(j, g), 0) - v[g]
            rows.append(c); lo.append(-np.inf); hi.append(0.0)
        A = lil_matrix((len(rows), self.nvar))
        for r, c in enumerate(rows):
            for k, val in c.items():
                if val != 0:
                    A[r, k] = val
        cost = np.zeros(self.nvar); cost[self.Z] = 1
        integ = np.r_[np.ones(2 * n * m), 0]
        lb = np.r_[np.zeros(2 * n * m), -(m + 1)]; ub = np.r_[np.ones(2 * n * m), m + 1]
        res = _milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integ,
                    bounds=Bounds(lb, ub), time_limit=120)
        x = np.round(res.x[: n * m]).reshape(n, m)
        alloc = x.argmax(axis=0).astype(np.int8)
        return float(res.fun), alloc


def violation_rows(alloc, n, m, market="EF1"):
    out = []
    for i in range(n):
        for j in range(n):
            if i == j or not (alloc == j).any():
                continue
            row = (alloc == j).astype(float) - (alloc == i).astype(float)
            idx = np.nonzero(alloc == j)[0]
            row[idx[0] if market == "EF1" else idx[-1]] -= 1.0
            out.append(row)
    return out


def master(Sp, n, m, market="EF1", vmin=0.0):
    terms = []
    for a_idx, a in enumerate(Sp):
        for row in violation_rows(a, n, m, market):
            terms.append((a_idx, row))
    K = len(terms); nv = m + 1 + K; big = m + 2.0
    A = lil_matrix((m - 1 + K + len(Sp), nv)); lo, hi = [], []
    r = 0
    for t in range(m - 1):
        A[r, t] = 1; A[r, t + 1] = -1; lo.append(0); hi.append(np.inf); r += 1
    by = {}
    for k, (a_idx, row) in enumerate(terms):
        for g in np.nonzero(row)[0]:
            A[r, g] = row[g]
        A[r, m] = -1; A[r, m + 1 + k] = -big; lo.append(-big); hi.append(np.inf); r += 1
        by.setdefault(a_idx, []).append(k)
    for a_idx in range(len(Sp)):
        for k in by.get(a_idx, []):
            A[r, m + 1 + k] = 1
        lo.append(1); hi.append(np.inf); r += 1
    cost = np.zeros(nv); cost[m] = -1
    lb = np.zeros(nv); ub = np.ones(nv)
    lb[0] = 1.0  # v[0] = 1
    lb[m - 1] = max(lb[m - 1], vmin)  # optional floor on every value (v is nonincreasing)
    lb[m], ub[m] = -(m + 1), m + 1
    integ = np.r_[np.zeros(m + 1), np.ones(K)]
    res = _milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integ,
                bounds=Bounds(lb, ub))
    return -res.fun, res.x[:m]


def decide(n, m, sigmas, market="EF1", tol=1e-6, max_rounds=2000, rng=None, init_v=(), vmin=0.0,
           exact_value=False):
    """Returns dict(result='refuted'|'counterexample', rounds, cert_size, eps, v).

    exact_value=True keeps iterating after eps <= 0 until the master's upper
    bound meets the oracle's value, returning value = max_v min_{A in S}
    (largest pair violation) over v with v[0] = 1, v >= vmin. With vmin > 0
    this value is a graded measure of how close the profile is to a
    counterexample (counterexample iff value > 0)."""
    rng = rng or np.random.default_rng(0)
    orc = Oracle(n, m, sigmas, market)
    Sp = []
    seeds = [np.ones(m), np.linspace(1, max(0.05, vmin), m)] + [np.sort(rng.random(m))[::-1] for _ in range(2)]
    for v in list(init_v) + seeds:
        v = np.maximum(np.asarray(v, float) / max(v[0], 1e-12), vmin)
        _, a = orc.solve(v)
        if not any((a == b).all() for b in Sp):
            Sp.append(a)
    for rounds in range(1, max_rounds + 1):
        eps, v = master(Sp, n, m, market, vmin)
        if eps <= tol and not exact_value:
            return {"result": "refuted", "rounds": rounds, "cert_size": len(Sp), "eps": eps,
                    "certificate": [a.tolist() for a in Sp]}
        z, a = orc.solve(v)
        if z > tol:  # every allocation in S violates some pair by >= z under v
            return {"result": "counterexample", "rounds": rounds, "cert_size": len(Sp),
                    "eps": eps, "oracle_min_violation": z, "v": v.tolist()}
        if exact_value and z >= eps - tol:  # upper bound met: value = eps
            return {"result": "refuted", "rounds": rounds, "cert_size": len(Sp), "eps": eps,
                    "value": eps, "v": v.tolist(), "certificate": [a.tolist() for a in Sp]}
        Sp.append(a)  # a has violation < eps at v, so it is new
    raise RuntimeError("CEGAR did not converge")


def market_balanced_in_S(n, m, sigmas):
    """True iff some subjectively SD-EF1 allocation is also market-SD-EF1
    (each block of n consecutive market items split one per agent). Such an
    allocation is EF1 for every market valuation, so the profile cannot be a
    counterexample. Solved as an ILP feasibility problem (no enumeration)."""
    orc = Oracle(n, m, sigmas)
    rows, lo, hi = list(orc.base[0]), list(orc.base[1]), list(orc.base[2])
    for s in range(0, m, n):
        for i in range(n):
            rows.append({orc.X(i, g): 1 for g in range(s, min(s + n, m))}); lo.append(-np.inf); hi.append(1)
    A = lil_matrix((len(rows), orc.nvar))
    for r, c in enumerate(rows):
        for k, val in c.items():
            A[r, k] = val
    integ = np.r_[np.ones(2 * n * m), 0]
    lb = np.r_[np.zeros(2 * n * m), -(m + 1)]; ub = np.r_[np.ones(2 * n * m), m + 1]
    try:
        _milp(np.zeros(orc.nvar), constraints=LinearConstraint(A.tocsr(), lo, hi),
              integrality=integ, bounds=Bounds(lb, ub), time_limit=120)
        return True
    except SolverError:
        # infeasible models also end here (status != 0); distinguish by a direct solve
        res = milp(np.zeros(orc.nvar), constraints=LinearConstraint(A.tocsr(), lo, hi),
                   integrality=integ, bounds=Bounds(lb, ub), options={"time_limit": 120})
        if res.status == 2:
            return False
        raise


def market_imbalance(n, m, sigmas):
    """min over subjectively SD-EF1 allocations of the number of excess items in
    market blocks (sum over blocks and agents of (items in block - 1)^+).
    0 iff a market-SD-EF1 allocation lies in S; a graded hardness measure."""
    orc = Oracle(n, m, sigmas)
    rows, lo, hi = list(orc.base[0]), list(orc.base[1]), list(orc.base[2])
    blocks = [list(range(s, min(s + n, m))) for s in range(0, m, n)]
    ne = len(blocks) * n
    base_nv = orc.nvar
    for b, blk in enumerate(blocks):
        for i in range(n):
            c = {orc.X(i, g): 1 for g in blk}; c[base_nv + b * n + i] = -1
            rows.append(c); lo.append(-np.inf); hi.append(1)
    nv = base_nv + ne
    A = lil_matrix((len(rows), nv))
    for r, c in enumerate(rows):
        for k, val in c.items():
            A[r, k] = val
    cost = np.r_[np.zeros(base_nv), np.ones(ne)]
    integ = np.r_[np.ones(2 * n * m), 0, np.ones(ne)]
    lb = np.r_[np.zeros(2 * n * m), -(m + 1), np.zeros(ne)]
    ub = np.r_[np.ones(2 * n * m), m + 1, np.full(ne, n)]
    res = _milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integ,
                bounds=Bounds(lb, ub), time_limit=120)
    return int(round(res.fun))


# ---------------------------------------------------------------------------
# Robust variant for hard instances: uses incumbents and MIP dual bounds, so a
# time-limited solve still yields a valid step whenever possible.
#   master : upper bound on eps <= tol  -> refuted (certified by the bound);
#            incumbent eps > tol        -> use its v (any v breaking S' works).
#   oracle : incumbent violation < eps - tol -> valid new cut (need not be optimal);
#            lower bound on violation > tol  -> counterexample (certified).
# ---------------------------------------------------------------------------

def _solve_raw(c, A, lo, hi, integ, lb, ub, time_limit):
    """Solve; on HiGHS internal 'Solve error' (status 4) retry without presolve
    and then on randomly permuted copies of the same model. Time-limited
    results (status 1) are returned as they are (incumbent + dual bound)."""
    lo, hi, lb, ub, integ = map(np.asarray, (lo, hi, lb, ub, integ))
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(lb, ub), options={"time_limit": time_limit})
    if res.status != 4:
        return res
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(lb, ub), options={"time_limit": time_limit, "presolve": False})
    if res.status != 4:
        return res
    for seed in range(4):
        prng = np.random.default_rng(seed)
        rp = prng.permutation(A.shape[0]); cp = prng.permutation(A.shape[1]); inv = np.argsort(cp)
        r2 = milp(c[cp], constraints=LinearConstraint(A[rp][:, cp], lo[rp], hi[rp]),
                  integrality=integ[cp], bounds=Bounds(lb[cp], ub[cp]),
                  options={"time_limit": time_limit})
        if r2.status != 4:
            if r2.x is not None:
                r2.x = r2.x[inv]
            return r2
    return res


def master_robust(Sp, n, m, time_limit=1800, tol=1e-6):
    terms = []
    for a_idx, a in enumerate(Sp):
        for row in violation_rows(a, n, m):
            terms.append((a_idx, row))
    K = len(terms); nv = m + 1 + K; big = m + 2.0
    A = lil_matrix((m - 1 + K + len(Sp), nv)); lo, hi = [], []
    r = 0
    for t in range(m - 1):
        A[r, t] = 1; A[r, t + 1] = -1; lo.append(0); hi.append(np.inf); r += 1
    by = {}
    for k, (a_idx, row) in enumerate(terms):
        for g in np.nonzero(row)[0]:
            A[r, g] = row[g]
        A[r, m] = -1; A[r, m + 1 + k] = -big; lo.append(-big); hi.append(np.inf); r += 1
        by.setdefault(a_idx, []).append(k)
    for a_idx in range(len(Sp)):
        for k in by.get(a_idx, []):
            A[r, m + 1 + k] = 1
        lo.append(1); hi.append(np.inf); r += 1
    cost = np.zeros(nv); cost[m] = -1
    lb = np.zeros(nv); ub = np.ones(nv); lb[0] = 1.0; lb[m], ub[m] = -(m + 1), m + 1
    integ = np.r_[np.zeros(m + 1), np.ones(K)]
    res = _solve_raw(cost, A.tocsr(), lo, hi, integ, lb, ub, time_limit)
    upper = -res.mip_dual_bound if getattr(res, "mip_dual_bound", None) is not None else np.inf
    if res.status == 0:
        return {"eps": -res.fun, "v": res.x[:m], "upper": -res.fun}
    if upper <= tol:
        return {"eps": None, "v": None, "upper": upper}
    if res.x is not None and -res.fun > tol:
        return {"eps": -res.fun, "v": res.x[:m], "upper": upper}
    raise SolverError(f"master inconclusive: status {res.status}, upper {upper}")


def oracle_robust(orc, v, eps, time_limit=1800, tol=1e-6):
    n, m = orc.n, orc.m
    rows, lo, hi = list(orc.base[0]), list(orc.base[1]), list(orc.base[2])
    for i, j in orc.pairs:
        c = {orc.Z: -1.0}
        for g in range(m):
            c[orc.X(j, g)] = c.get(orc.X(j, g), 0) + v[g]
            c[orc.X(i, g)] = c.get(orc.X(i, g), 0) - v[g]
            c[orc.F(j, g)] = c.get(orc.F(j, g), 0) - v[g]
        rows.append(c); lo.append(-np.inf); hi.append(0.0)
    A = lil_matrix((len(rows), orc.nvar))
    for r, c in enumerate(rows):
        for k, val in c.items():
            if val != 0:
                A[r, k] = val
    cost = np.zeros(orc.nvar); cost[orc.Z] = 1
    integ = np.r_[np.ones(2 * n * m), 0]
    lb = np.r_[np.zeros(2 * n * m), -(m + 1)]; ub = np.r_[np.ones(2 * n * m), m + 1]
    res = _solve_raw(cost, A.tocsr(), lo, hi, integ, lb, ub, time_limit)
    lower = res.mip_dual_bound if getattr(res, "mip_dual_bound", None) is not None else -np.inf
    alloc = None
    if res.x is not None:
        alloc = np.round(res.x[: n * m]).reshape(n, m).argmax(axis=0).astype(np.int8)
    if res.status == 0:
        return {"z": res.fun, "alloc": alloc, "lower": res.fun}
    if res.x is not None and res.fun < eps - tol:
        return {"z": res.fun, "alloc": alloc, "lower": lower}
    if lower > tol:
        return {"z": None, "alloc": None, "lower": lower}
    raise SolverError(f"oracle inconclusive: status {res.status}, lower {lower}")


def decide_robust(n, m, sigmas, time_limit=1800, tol=1e-6, max_rounds=200, rng=None):
    rng = rng or np.random.default_rng(0)
    orc = Oracle(n, m, sigmas)
    Sp = []
    for v in [np.ones(m), np.linspace(1, 0.05, m)] + [np.sort(rng.random(m))[::-1] for _ in range(2)]:
        v = np.asarray(v, float) / v[0]
        o = oracle_robust(orc, v, eps=np.inf, time_limit=time_limit)
        if o["alloc"] is not None and not any((o["alloc"] == b).all() for b in Sp):
            Sp.append(o["alloc"])
    for rounds in range(1, max_rounds + 1):
        mst = master_robust(Sp, n, m, time_limit=time_limit)
        if mst["v"] is None or mst["upper"] <= tol:
            return {"result": "refuted", "rounds": rounds, "cert_size": len(Sp),
                    "certificate": [a.tolist() for a in Sp]}
        o = oracle_robust(orc, mst["v"], mst["eps"], time_limit=time_limit)
        if o["alloc"] is None or o["lower"] > tol:
            return {"result": "counterexample", "rounds": rounds, "v": mst["v"].tolist(),
                    "eps": mst["eps"], "oracle_lower_bound": o["lower"]}
        Sp.append(o["alloc"])
    raise RuntimeError("robust CEGAR did not converge")
