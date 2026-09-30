"""Covering families for two agents: exact small-instance experiments.

Setting.  Two agents, goods 0..m-1, strict rankings s1, s2.  An allocation is a +1/-1 vector x
(x[g] = +1 if agent 1 holds g).  S is the set of subjectively SD-EF1 allocations:
agent 1 needs the running sum of x along s1 to stay >= -1, agent 2 needs the running
sum of x along s2 to stay <= +1 (prefix form of SD-EF1).

A family F of allocations *covers* a class of market valuations if every valuation in the class
makes at least one member of F market EF1 (for the pair (A1, A2): some g in A_j with v(A_i) >= v(A_j - g),
for both i != j with A_j non-empty).  Two classes are considered:

  additive  : v_0 >= v_1 >= ... >= v_{m-1} >= 0, v additive (market order = identity);
  monotone  : every inclusion-monotone v: 2^M -> R_{>=0} (then the market order plays no role).

An allocation and its agent-swap have the same market behaviour, so they are identified.

Everything here is independent of fd.py.  Non-covering is decided by linear programs:
for each choice of "which agent envies beyond one good" per family member, the failure
conditions are strict linear inequalities in v (scaled to margin 1); the family fails iff one
choice is feasible.  The LP returns a witness valuation, checked directly against the definition.

Experiments (python n2_cover.py all):
  E1  additive: two allocations cover every additive valuation on all sampled hard profiles
      (random profiles, plus a hill-climb that tries to destroy every 2-cover);
  E2  monotone: exact minimum covering family (CEGAR hitting set) on sampled hard profiles, m = 7, 8;
  E3  monotone: the reversal-path family of Theorem 1 is an exact-LP cover, every member is necessary
      (sampled hard profiles, m = 9, 10, 11);
  E4  the disjoint-defect route already fails for two agents (m = 11 profile), yet a 2-cover exists;
  E5  additive: a covering pair need not lie on the reversal path of Theorem 3.1;
  E0  validation of the linear-programming cover test against direct evaluation (python n2_cover.py E0).
Results are written to ../results/n2_cover_experiments.json.
"""
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix, vstack


# ---------------------------------------------------------------------------------------------
# allocations and subjective fairness
# ---------------------------------------------------------------------------------------------
def all_pm(m):
    a = ((np.arange(1 << m)[:, None] >> np.arange(m)) & 1).astype(np.int8)
    return 2 * a - 1


def subj_ok(X, s1, s2):
    w1 = np.cumsum(X[:, s1], axis=1)
    w2 = np.cumsum(X[:, s2], axis=1)
    return (w1.min(axis=1) >= -1) & (w2.max(axis=1) <= 1)


def defect_masks(X):
    """Bit t-1 is set iff the market prefix of length t has |#agent1 - #agent2| >= 2."""
    bad = np.abs(np.cumsum(X, axis=1)) >= 2
    return (bad * (1 << np.arange(X.shape[1]))).sum(axis=1)


def is_hard(dm, S_idx):
    """No allocation in S is market-block balanced (EF1 for every additive valuation)."""
    return not (dm[S_idx] == 0).any()


# ---------------------------------------------------------------------------------------------
# additive markets (market order = identity, v nonincreasing)
# ---------------------------------------------------------------------------------------------
def _coeffs_additive(x):
    """c_U . v > 0  iff agent 2 envies agent 1 beyond one good;  c_L . v > 0 iff the reverse."""
    cU = x.astype(float).copy()
    cL = -x.astype(float)
    a1 = np.where(x == 1)[0]
    a2 = np.where(x == -1)[0]
    if len(a1):
        cU[a1.min()] -= 1.0
    else:
        cU = None
    if len(a2):
        cL[a2.min()] -= 1.0
    else:
        cL = None
    return cU, cL


def additive_fails(xs):
    """True iff some nonincreasing v >= 0 makes every allocation in xs violate market EF1."""
    m = len(xs[0])
    opts = [[c for c in _coeffs_additive(x) if c is not None] for x in xs]
    mono = np.zeros((m - 1, m + 1))
    for i in range(m - 1):
        mono[i, i] = -1
        mono[i, i + 1] = 1
    for choice in itertools.product(*opts):
        rows = [mono]
        for c in choice:
            r = np.zeros(m + 1)
            r[:m] = -c
            r[m] = 1
            rows.append(r[None, :])
        Aub = np.vstack(rows)
        Aeq = np.zeros((1, m + 1))
        Aeq[0, :m] = 1
        res = linprog(np.r_[np.zeros(m), -1.0], A_ub=Aub, b_ub=np.zeros(len(Aub)), A_eq=Aeq, b_eq=[1.0],
                      bounds=[(0, None)] * m + [(None, 1.0)], method="highs")
        if res.status == 0 and -res.fun > 1e-9:
            return True
    return False


def ef1_additive(x, V, tol=1e-12):
    """Direct definition: x is market EF1 for each row of V (nonincreasing valuations)."""
    a1, a2 = (x == 1), (x == -1)
    v1, v2 = V[:, a1].sum(1), V[:, a2].sum(1)
    m1 = V[:, a1].max(1) if a1.any() else np.zeros(len(V))
    m2 = V[:, a2].max(1) if a2.any() else np.zeros(len(V))
    ok2 = np.ones(len(V), bool) if not a1.any() else (v2 >= v1 - m1 - tol)
    ok1 = np.ones(len(V), bool) if not a2.any() else (v1 >= v2 - m2 - tol)
    return ok1 & ok2


# ---------------------------------------------------------------------------------------------
# arbitrary monotone markets: v is a vector indexed by bundle masks
# ---------------------------------------------------------------------------------------------
class Monotone:
    def __init__(self, m):
        self.m = m
        self.NS = 1 << m
        r = 0
        mono = lil_matrix((m * (self.NS // 2), self.NS))
        for S in range(self.NS):
            for g in range(m):
                if not (S >> g) & 1:
                    mono[r, S] = 1
                    mono[r, S | (1 << g)] = -1
                    r += 1
        self.mono = csr_matrix(mono[:r])
        self.bounds = [(0, 0)] + [(0, None)] * (self.NS - 1)        # v(empty) = 0 w.l.o.g.
        X = all_pm(m)
        w = 1 << np.arange(m)
        self.X = X
        self.A1 = ((X == 1) * w).sum(axis=1)
        self.A2 = ((X == -1) * w).sum(axis=1)

    def masks(self, x):
        w = 1 << np.arange(self.m)
        return int(((x == 1) * w).sum()), int(((x == -1) * w).sum())

    def _rows(self, a1, a2, side):
        m = self.m
        if side == "U":                       # agent 2 envies beyond one good: v(A1 - g) - v(A2) >= 1 for all g in A1
            if a1 == 0:
                return None
            return [(a1 & ~(1 << g), a2) for g in range(m) if (a1 >> g) & 1]
        if a2 == 0:
            return None
        return [(a2 & ~(1 << g), a1) for g in range(m) if (a2 >> g) & 1]

    def witness(self, fam):
        """fam: list of (a1, a2) bundle masks.  A monotone valuation defeating every member, or None."""
        opts = []
        for (a1, a2) in fam:
            opts.append([R for R in (self._rows(a1, a2, "U"), self._rows(a1, a2, "L")) if R is not None])
        for choice in itertools.product(*opts):
            rows = [p for R in choice for p in R]
            A = lil_matrix((len(rows), self.NS))
            for i, (hi, lo) in enumerate(rows):
                A[i, hi] += -1.0
                A[i, lo] += 1.0
            Aub = vstack([self.mono, csr_matrix(A)])
            bub = np.r_[np.zeros(self.mono.shape[0]), -np.ones(len(rows))]
            res = linprog(np.zeros(self.NS), A_ub=Aub, b_ub=bub, bounds=self.bounds, method="highs")
            if res.status == 0:
                return res.x
        return None

    def ef1(self, vv, a1, a2, tol=1e-9):
        m = self.m
        ok2 = (a1 == 0) or any(vv[a2] >= vv[a1 & ~(1 << g)] - tol for g in range(m) if (a1 >> g) & 1)
        ok1 = (a2 == 0) or any(vv[a1] >= vv[a2 & ~(1 << g)] - tol for g in range(m) if (a2 >> g) & 1)
        return ok1 and ok2


def random_monotone_valuations(mono, rng, K):
    m, NS = mono.m, mono.NS
    bits = ((np.arange(NS)[:, None] >> np.arange(m)) & 1).astype(float)          # NS x m
    out = []
    for _ in range(K):
        kind = rng.integers(5)
        if kind == 0:                                     # additive
            w = rng.random(m) ** rng.integers(1, 4)
            vv = bits @ w
        elif kind == 1:                                   # threshold on a random subset
            Wm = (rng.random(m) < rng.uniform(0.3, 0.9)).astype(float)
            k = rng.integers(1, max(2, int(Wm.sum()) + 1))
            vv = ((bits @ Wm) >= k).astype(float)
        elif kind == 2:                                   # budget additive
            w = rng.random(m)
            B = rng.uniform(0.3, 0.9) * w.sum()
            vv = np.minimum(B, bits @ w)
        elif kind == 3:                                   # monotone closure of a sparse random table
            t = np.zeros(NS)
            idx = rng.integers(0, NS, size=rng.integers(3, 3 * m))
            t[idx] = rng.random(len(idx))
            vv = t
            for g in range(m):
                sel = np.where((np.arange(NS) >> g) & 1)[0]
                vv[sel] = np.maximum(vv[sel], vv[sel ^ (1 << g)])
        else:                                             # weighted sum of thresholds
            vv = np.zeros(NS)
            for _ in range(rng.integers(2, 5)):
                Wm = (rng.random(m) < 0.6).astype(float)
                k = rng.integers(1, max(2, int(Wm.sum()) + 1))
                vv += rng.random() * ((bits @ Wm) >= k)
        out.append(vv)
    return out


def min_monotone_cover(mono, S_idx, rng, max_iter=600, n_init=120):
    """Smallest family inside S covering all monotone valuations: hitting set over witnesses (CEGAR)."""
    classes = {}
    for i in S_idx:
        key = frozenset((int(mono.A1[i]), int(mono.A2[i])))
        classes.setdefault(key, (int(mono.A1[i]), int(mono.A2[i])))
    classes = list(classes.values())
    C = len(classes)
    rows = []

    def add(vv):
        E = [c for c in range(C) if mono.ef1(vv, *classes[c])]
        rows.append(E)
        return E

    for vv in random_monotone_valuations(mono, rng, n_init):
        add(vv)
    for it in range(1, max_iter + 1):
        if any(len(E) == 0 for E in rows):
            return None, it
        Aub = np.zeros((len(rows), C))
        for i, E in enumerate(rows):
            Aub[i, E] = 1
        res = milp(c=np.ones(C), constraints=LinearConstraint(Aub, lb=np.ones(len(rows)), ub=np.inf),
                   integrality=np.ones(C), bounds=Bounds(0, 1))
        y = np.round(res.x).astype(int)
        fam = [classes[c] for c in range(C) if y[c]]
        w = mono.witness(fam)
        if w is None:
            return len(fam), it
        E = add(w)
        assert not any(y[c] for c in E), "witness does not defeat the family"
    return "timeout", max_iter


# ---------------------------------------------------------------------------------------------
# the reversal path of Theorem 1 (independent implementation)
# ---------------------------------------------------------------------------------------------
def reversal_path(s1, s2):
    """Allocations A^0..A^T of Theorem 1 (+1/-1 vectors over the real goods)."""
    m = len(s1)
    s1, s2 = list(s1), list(s2)
    dummy = m if m % 2 else None
    if dummy is not None:
        s1.append(dummy)
        s2.append(dummy)
    mm = len(s1)
    red = [(s1[2 * i], s1[2 * i + 1]) for i in range(mm // 2)]
    blue = [(s2[2 * i], s2[2 * i + 1]) for i in range(mm // 2)]
    pos2 = {g: i for i, g in enumerate(s2)}
    rmate, bmate = {}, {}
    for a, b in red:
        rmate[a], rmate[b] = b, a
    for a, b in blue:
        bmate[a], bmate[b] = b, a
    seen, cycles = set(), []
    for g in range(mm):
        if g in seen:
            continue
        cyc, cur = [g], g
        seen.add(g)
        while True:
            nxt = rmate[cur]
            if nxt in seen:
                break
            cyc.append(nxt)
            seen.add(nxt)
            nxt2 = bmate[nxt]
            if nxt2 in seen:
                break
            cyc.append(nxt2)
            seen.add(nxt2)
            cur = nxt2
        cycles.append(cyc)
    owner = {}
    for cyc in cycles:
        for i, g in enumerate(cyc):
            owner[g] = 1 if i % 2 == 0 else 2
    fam, cur = [dict(owner)], dict(owner)
    for cyc in cycles:
        L = len(cyc) // 2
        if L == 1:
            a, b = cyc
            cur[a], cur[b] = cur[b], cur[a]
            fam.append(dict(cur))
            continue
        cand = [(min(pos2[a], pos2[b]), a, b) for a, b in blue if a in cyc]
        _, a, b = min(cand)
        c1 = a if cur[a] == 1 else b
        c_last = b if c1 == a else a
        idx = cyc.index(c1)
        seq = None
        for orient in (1, -1):
            s = [cyc[(idx + orient * k) % len(cyc)] for k in range(len(cyc))]
            if rmate[s[0]] == s[1] and s[-1] == c_last:
                seq = s
                break
        assert seq is not None
        for j in range(L):
            u, w = seq[2 * j], seq[2 * j + 1]
            cur[u], cur[w] = cur[w], cur[u]
            fam.append(dict(cur))
    return [np.array([1 if f[g] == 1 else -1 for g in range(m)], dtype=np.int8) for f in fam]


def market_classes(xs):
    seen, out = set(), []
    for x in xs:
        k = min(x.tobytes(), (-x).tobytes())
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


# ---------------------------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------------------------
def sample_hard(m, rng, want, max_trials):
    X = all_pm(m)
    dm = defect_masks(X)
    out, trials = [], 0
    while len(out) < want and trials < max_trials:
        trials += 1
        s1, s2 = rng.permutation(m), rng.permutation(m)
        S = np.where(subj_ok(X, s1, s2))[0]
        if len(S) == 0 or not is_hard(dm, S):
            continue
        out.append((s1.tolist(), s2.tolist()))
    return out, trials


def two_cover_exists(X, dm, S_idx):
    D = dm[S_idx]
    for i, j in itertools.combinations(range(len(S_idx)), 2):
        if (int(D[i]) & int(D[j])) == 0 and not additive_fails([X[S_idx[i]], X[S_idx[j]]]):
            return True
    return False


def experiment_E1(args):
    m, seed, n_random, n_climb_seconds = args
    rng = np.random.default_rng(seed)
    X = all_pm(m)
    dm = defect_masks(X)
    profiles, trials = sample_hard(m, rng, n_random, 4_000_000)
    lacking = []
    for s1, s2 in profiles:
        S = np.where(subj_ok(X, np.array(s1), np.array(s2)))[0]
        if not two_cover_exists(X, dm, S):
            lacking.append((s1, s2))
    # adversarial hill-climb: minimise the number of allocation pairs with disjoint defect masks
    def score(s1, s2):
        S = np.where(subj_ok(X, s1, s2))[0]
        if len(S) == 0 or not is_hard(dm, S):
            return None
        D = dm[S]
        return int(np.triu((D[:, None] & D[None, :]) == 0, 1).sum()), S

    t0, restarts, best = time.time(), 0, None
    while time.time() - t0 < n_climb_seconds:
        restarts += 1
        while True:
            s1, s2 = rng.permutation(m), rng.permutation(m)
            r = score(s1, s2)
            if r is not None:
                break
        cur, stall = r[0], 0
        while stall < 300 and cur > 0 and time.time() - t0 < n_climb_seconds:
            t1, t2 = s1.copy(), s2.copy()
            w = t1 if rng.random() < .5 else t2
            i, j = rng.choice(m, 2, replace=False)
            w[i], w[j] = w[j], w[i]
            r = score(t1, t2)
            if r is None:
                stall += 1
                continue
            if r[0] <= cur:
                stall = 0 if r[0] < cur else stall + 1
                s1, s2, cur = t1, t2, r[0]
            else:
                stall += 1
        S = np.where(subj_ok(X, s1, s2))[0]
        ok = two_cover_exists(X, dm, S)
        if best is None or cur < best["disjoint_pairs"]:
            best = {"disjoint_pairs": int(cur), "two_cover": bool(ok), "s1": s1.tolist(), "s2": s2.tolist(), "S": int(len(S))}
        if not ok:
            lacking.append((s1.tolist(), s2.tolist()))
    return {"m": m, "seed": seed, "random_hard_profiles": len(profiles), "random_trials": trials,
            "profiles_without_2cover": lacking, "climb_restarts": restarts, "climb_best": best}


def experiment_E2(args):
    m, seed, want = args
    rng = np.random.default_rng(seed)
    mono = Monotone(m)
    dm = defect_masks(mono.X)
    profiles, _ = sample_hard(m, rng, want, 3_000_000)
    res = []
    for s1, s2 in profiles:
        S = np.where(subj_ok(mono.X, np.array(s1), np.array(s2)))[0]
        k, it = min_monotone_cover(mono, S, rng)
        res.append({"s1": s1, "s2": s2, "S": int(len(S)), "min_cover": k, "cegar_iterations": it})
    return {"m": m, "seed": seed, "ceil_m_over_2": -(-m // 2), "profiles": res}


def experiment_E3(args):
    m, seed, want = args
    rng = np.random.default_rng(seed)
    mono = Monotone(m)
    profiles, _ = sample_hard(m, rng, want, 3_000_000)
    res = []
    for s1, s2 in profiles:
        S_ok = subj_ok(mono.X, np.array(s1), np.array(s2))
        path = reversal_path(s1, s2)
        in_S = all(bool(subj_ok(x[None, :], np.array(s1), np.array(s2))[0]) for x in path)
        cls = market_classes(path)
        fam = [mono.masks(x) for x in cls]
        cover = mono.witness(fam) is None
        necessary = all(mono.witness(fam[:i] + fam[i + 1:]) is not None for i in range(len(fam)))
        res.append({"s1": s1, "s2": s2, "S": int(S_ok.sum()), "path_members": len(path), "market_classes": len(cls),
                    "path_in_S": in_S, "path_is_cover": cover, "every_member_necessary": necessary})
    return {"m": m, "seed": seed, "profiles": res}


def experiment_E4():
    m = 11
    s1 = [0, 3, 4, 7, 2, 6, 1, 9, 8, 10, 5]
    s2 = [7, 3, 4, 6, 0, 2, 8, 9, 10, 5, 1]
    X = all_pm(m)
    dm = defect_masks(X)
    S = np.where(subj_ok(X, np.array(s1), np.array(s2)))[0]
    near = np.abs(np.cumsum(X[S], axis=1)).max(axis=1) <= 2
    N = S[near]
    D = dm[N]
    disjoint_pairs = int(np.triu((D[:, None] & D[None, :]) == 0, 1).sum())
    found = None
    Ds = dm[S]
    for i, j in itertools.combinations(range(len(S)), 2):
        if (int(Ds[i]) & int(Ds[j])) == 0 and not additive_fails([X[S[i]], X[S[j]]]):
            found = [X[S[i]].tolist(), X[S[j]].tolist()]
            break
    return {"m": m, "s1": s1, "s2": s2, "subjectively_fair": int(len(S)), "near_block_fair": int(len(N)),
            "block_balanced_fair": int((dm[S] == 0).sum()), "near_block_pairs_with_disjoint_defects": disjoint_pairs,
            "two_cover_exists": found is not None, "two_cover_owner_vectors": found}


def sampled_additive_valuations(m, rng, K):
    """K nonincreasing additive valuations: uniform sorted, geometric, sums of thresholds, two-level."""
    Vs = []
    for _ in range(K // 4):
        Vs.append(np.sort(rng.random(m))[::-1])
    for _ in range(K // 4):
        Vs.append(rng.random() ** np.arange(m) * rng.random())
    for _ in range(K // 4):
        w = rng.random(m) * (rng.random(m) < 0.4)
        Vs.append(np.cumsum(w[::-1])[::-1])
    for _ in range(K - 3 * (K // 4)):
        t = rng.integers(1, m)
        Vs.append(np.r_[np.ones(t), np.full(m - t, rng.random())] + 1e-3 * rng.random(m))
    return -np.sort(-np.array(Vs), axis=1)


def experiment_E0(args):
    """Validate the linear-programming cover test against direct evaluation of the EF1 definition."""
    m, seed = args
    rng = np.random.default_rng(seed)
    X = all_pm(m)
    dm = defect_masks(X)
    V = sampled_additive_valuations(m, rng, 40000)
    # (1) single allocations
    lp_fail = lp_cover = contradictions = 0
    for _ in range(3000):
        x = X[rng.integers(len(X))]
        lp = additive_fails([x])
        broken = (~ef1_additive(x, V)).any()
        if (not lp) and broken:
            contradictions += 1                   # LP claims cover but a sampled valuation breaks it: must never happen
        lp_fail += int(lp)
        lp_cover += int(not lp)
    # (2) covering pairs on hard profiles
    cover_pairs = cover_pairs_broken = 0
    noncover_sampled = noncover_broken = 0
    profiles = 0
    while profiles < 120:
        s1, s2 = rng.permutation(m), rng.permutation(m)
        S = np.where(subj_ok(X, s1, s2))[0]
        if len(S) == 0 or not is_hard(dm, S):
            continue
        profiles += 1
        D = dm[S]
        for i, j in itertools.combinations(range(len(S)), 2):
            xi, xj = X[S[i]], X[S[j]]
            if (int(D[i]) & int(D[j])) == 0 and not additive_fails([xi, xj]):
                cover_pairs += 1
                if ((~ef1_additive(xi, V)) & (~ef1_additive(xj, V))).any():
                    cover_pairs_broken += 1
            elif noncover_sampled < 3000 and rng.random() < 0.05:
                noncover_sampled += 1
                noncover_broken += int(((~ef1_additive(xi, V)) & (~ef1_additive(xj, V))).any())
    return {"m": m, "seed": seed, "sampled_valuations": len(V), "single_allocations": 3000, "lp_fails": lp_fail,
            "lp_covers": lp_cover, "lp_cover_but_broken_by_sample": contradictions, "hard_profiles": profiles,
            "lp_covering_pairs_tested": cover_pairs, "lp_covering_pairs_broken_by_sample": cover_pairs_broken,
            "sampled_noncovering_pairs": noncover_sampled, "sampled_noncovering_pairs_broken_by_sample": noncover_broken}


def experiment_E5(args):
    """Fraction of sampled hard profiles whose additive 2-cover can be found inside the reversal path."""
    m, seed, want = args
    rng = np.random.default_rng(seed)
    profiles, _ = sample_hard(m, rng, want, 4_000_000)
    inside = 0
    for s1, s2 in profiles:
        cls = market_classes(reversal_path(s1, s2))
        if any(not additive_fails([cls[i], cls[j]]) for i, j in itertools.combinations(range(len(cls)), 2)):
            inside += 1
    return {"m": m, "seed": seed, "profiles": len(profiles), "with_covering_pair_on_path": inside}


def run_all(out_path):
    t0 = time.time()
    out = {}
    with Pool(12) as pool:
        e1 = pool.map(experiment_E1, [(m, 100 + m, 100, 120) for m in (8, 9, 10, 11, 12, 13)])
        e2 = pool.map(experiment_E2, [(7, 1, 8), (8, 15, 12)])
        e3 = pool.map(experiment_E3, [(9, 9, 4), (10, 10, 4), (11, 11, 4)])
        e5 = pool.map(experiment_E5, [(m, 500 + m, 60) for m in (8, 9, 10, 11)])
    out["E1_additive_two_cover"] = e1
    out["E2_monotone_exact_min_cover"] = e2
    out["E3_reversal_path_is_exact_cover"] = e3
    out["E4_disjoint_defect_route_fails_for_two_agents"] = experiment_E4()
    out["E5_covering_pair_on_reversal_path"] = e5
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def summarize(out):
    print("E1 (additive, two allocations always cover?)")
    for r in out["E1_additive_two_cover"]:
        print(f"  m={r['m']}: random hard profiles {r['random_hard_profiles']}, without 2-cover: {len(r['profiles_without_2cover'])}; "
              f"climb restarts {r['climb_restarts']}, best #disjoint pairs {r['climb_best']['disjoint_pairs']} (2-cover: {r['climb_best']['two_cover']})")
    print("E2 (monotone, exact minimum cover)")
    for r in out["E2_monotone_exact_min_cover"]:
        print(f"  m={r['m']}: ceil(m/2)={r['ceil_m_over_2']}; min covers {[p['min_cover'] for p in r['profiles']]}")
    print("E3 (reversal path = exact cover, every member necessary)")
    for r in out["E3_reversal_path_is_exact_cover"]:
        print(f"  m={r['m']}: " + "; ".join(f"cls={p['market_classes']} cover={p['path_is_cover']} nec={p['every_member_necessary']} inS={p['path_in_S']}" for p in r["profiles"]))
    for r in out.get("E5_covering_pair_on_reversal_path", []):
        print(f"E5 m={r['m']}: {r['with_covering_pair_on_path']}/{r['profiles']} sampled hard profiles have a covering pair inside the reversal path")
    e4 = out["E4_disjoint_defect_route_fails_for_two_agents"]
    print("E4", {k: e4[k] for k in ("m", "subjectively_fair", "near_block_fair", "block_balanced_fair", "near_block_pairs_with_disjoint_defects", "two_cover_exists")})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        summarize(run_all("../results/n2_cover_experiments.json"))
    elif len(sys.argv) > 1 and sys.argv[1] == "E0":          # LP validation, merged into the results file
        path = "../results/n2_cover_experiments.json"
        out = json.load(open(path))
        out["E0_lp_validation"] = experiment_E0((10, 1000))
        json.dump(out, open(path, "w"), indent=1)
        print(out["E0_lp_validation"])
    elif len(sys.argv) > 1 and sys.argv[1] == "E5":          # add E5 to an existing results file
        path = "../results/n2_cover_experiments.json"
        out = json.load(open(path))
        with Pool(4) as pool:
            out["E5_covering_pair_on_reversal_path"] = pool.map(experiment_E5, [(m, 500 + m, 60) for m in (8, 9, 10, 11)])
        json.dump(out, open(path, "w"), indent=1)
        print(out["E5_covering_pair_on_reversal_path"])
    else:
        print(__doc__)
