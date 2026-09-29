"""Independent check, with fd.py rather than the checker in verification/repair,
of the repair-distance family (Theorem 14) and the one-singleton rule (Theorem 7).

Agents: 0 = dummy with the market ranking pi, 1 = sigma_1 (owner of the
prescribed matching S), 2 = sigma_2. "Fair" = market-balanced (each market
triple split one per agent) + SD-EF1 for every agent w.r.t. its own ranking.
Own-class balanced = agent 1 has exactly one good in every full sigma_1 triple.
dist(S, T) = |S minus T|.

Theorem 1 (connected repair-distance family, m = 6r): rebuilt from the general
formulas, then brute force over all market-balanced allocations for r = 2, 3, 4.
Theorem 3 (condition (C) on sigma_2): the block rule on random sigma_2
satisfying (C) with arbitrary sigma_1, plus the converse witness construction.
"""
import json
import numpy as np
import fd

N = 3


def family(r):
    q = 2 * r
    a = lambda i: 3 * (i % q)
    b = lambda i: 3 * (i % q) + 1
    z = lambda i: 3 * (i % q) + 2
    s1 = []
    for j in range(q):
        s1 += [a(j - 1), b(j - 2), z(j)]
    s2 = [a(0), a(1), b(0), b(1), b(2), b(3)] + [a(i) for i in range(2, q)] + \
         [b(i) for i in range(4, q)] + [z(i) for i in range(q)]
    S = {z(i) for i in range(q)}
    repaired_S = (S - {z(q - 1)}) | {a(q - 1)}
    repaired_X = ({a(i) for i in range(q)} - {a(2), a(q - 1)}) | {b(2), b(q - 1)}
    m = 3 * q
    return m, s1, s2, S, repaired_S, repaired_X


def owner_from(m, S1, X2):
    o = np.zeros(m, dtype=np.int8)
    for g in S1: o[g] = 1
    for g in X2: o[g] = 2
    return o


def fair(alloc, sig):
    return fd.subjective_sd_ef1(alloc, sig)


def own_balanced(alloc, s1):
    ok = np.ones(len(alloc), bool)
    for k in range(0, len(s1) - len(s1) % 3, 3):
        ok &= (alloc[:, s1[k:k + 3]] == 1).sum(axis=1) == 1
    return ok


def check_theorem1(r):
    m, s1, s2, S, rS, rX = family(r)
    sig = [list(range(m)), s1, s2]
    Mal = fd.market_sd_ef1_allocations(N, m)
    F = fair(Mal, sig)
    fairA = Mal[F]
    ob = own_balanced(fairA, s1)
    Smask = np.zeros(m, bool); Smask[list(S)] = True
    dist = (Smask[None, :] & (fairA != 1)).sum(axis=1)
    hist = {int(k): int(v) for k, v in zip(*np.unique(dist[ob], return_counts=True))}
    rep = owner_from(m, rS, rX)
    rep_ok = bool(fd.subjective_sd_ef1(rep[None, :], sig)[0]) and \
        bool(fd.market_sd_ef1_allocations is not None) and \
        all(len(set(rep[k:k + 3])) == 3 for k in range(0, m, 3))
    return {"r": r, "m": m, "market_allocations": int(len(Mal)), "fair": int(F.sum()),
            "own_balanced_fair": int(ob.sum()),
            "S_completable": bool((dist == 0).any()),
            "min_dist_own_balanced": int(dist[ob].min()), "own_balanced_dist_hist": hist,
            "min_dist_any_fair": int(dist.min()),
            "repaired_allocation_fair": rep_ok, "repair_dist": len(S - rS)}


def check_repair_large(rmax=10):
    out = {}
    for r in range(2, rmax + 1):
        m, s1, s2, S, rS, rX = family(r)
        rep = owner_from(m, rS, rX)
        ok = bool(fd.subjective_sd_ef1(rep[None, :], [list(range(m)), s1, s2])[0]) and \
            all(len(set(rep[k:k + 3])) == 3 for k in range(0, m, 3))
        out[r] = ok
    return out


# ---------------- Theorem 3 ----------------
def cond_C(s2, m):
    blk = [g // 3 for g in range(m)]
    cnt = {}
    singles = 0
    for g in s2:
        b = blk[g]
        c = cnt.get(b, 0)
        if c == 0: singles += 1
        elif c == 1: singles -= 1
        cnt[b] = c + 1
        if singles > 1:
            return False
    return True


def rule(m, s1, s2):
    pos1 = {g: k for k, g in enumerate(s1)}; pos2 = {g: k for k, g in enumerate(s2)}
    o = np.zeros(m, dtype=np.int8)
    for b in range(m // 3):
        blk = [3 * b, 3 * b + 1, 3 * b + 2]
        g1 = min(blk, key=lambda g: pos1[g]); blk.remove(g1)
        g2 = min(blk, key=lambda g: pos2[g]); blk.remove(g2)
        o[g1], o[g2], o[blk[0]] = 1, 2, 0
    return o


def gen_C(m, rng, kind):
    """kind 0: disjoint adjacent transpositions of pi;
    kind 1: permute blocks and goods within blocks, then swap at selected block boundaries."""
    if kind == 0:
        s = list(range(m)); k = 0
        while k < m - 1:
            if rng.random() < 0.4:
                s[k], s[k + 1] = s[k + 1], s[k]; k += 2
            else:
                k += 1
        return s
    blocks = [list(rng.permutation([3 * b, 3 * b + 1, 3 * b + 2])) for b in rng.permutation(m // 3)]
    s = [g for blk in blocks for g in blk]
    for bd in range(3, m, 3):
        if rng.random() < 0.4 and s[bd - 1] // 3 != s[bd] // 3:
            s[bd - 1], s[bd] = s[bd], s[bd - 1]
    return [int(x) for x in s]


def check_theorem3(trials=4000, seed=0):
    rng = np.random.default_rng(seed)
    res = {"rule_tested": 0, "rule_fail": 0, "gen_not_C": 0, "converse_tested": 0, "converse_rule_fails": 0}
    for t in range(trials):
        m = 3 * int(rng.integers(1, 11))
        s2 = gen_C(m, rng, t % 2)
        if not cond_C(s2, m):
            res["gen_not_C"] += 1; continue
        s1 = [int(x) for x in rng.permutation(m)]
        o = rule(m, s1, s2)
        ok = bool(fd.subjective_sd_ef1(o[None, :], [list(range(m)), s1, s2])[0])
        res["rule_tested"] += 1; res["rule_fail"] += (not ok)
    # converse: random sigma_2 violating (C); build sigma_1 from the witness prefix
    for t in range(trials):
        m = 3 * int(rng.integers(2, 11))
        s2 = [int(x) for x in rng.permutation(m)]
        if cond_C(s2, m): continue
        # shortest prefix with >= 2 singleton blocks
        cnt = {}; P = None
        for k, g in enumerate(s2):
            cnt[g // 3] = cnt.get(g // 3, 0) + 1
            if sum(1 for v in cnt.values() if v == 1) >= 2:
                P = s2[:k + 1]; break
        chosen = {}
        for g in P:
            chosen.setdefault(g // 3, g)
        sel = list(chosen.values())
        s1 = sel + [g for g in range(m) if g not in sel]
        o = rule(m, s1, s2)
        ok = bool(fd.subjective_sd_ef1(o[None, :], [list(range(m)), s1, s2])[0])
        res["converse_tested"] += 1; res["converse_rule_fails"] += (not ok)
    return res


if __name__ == "__main__":
    out = {"theorem1": [check_theorem1(r) for r in (2, 3, 4)],
           "theorem1_repair_r2_to_10": check_repair_large(10),
           "theorem3": check_theorem3()}
    print(json.dumps(out, indent=1))
    json.dump(out, open("../results/verify_repair_onesingleton.json", "w"), indent=1)
