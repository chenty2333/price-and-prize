"""Extra machine audits for claims of the AAMAS manuscript (independent of fd.py / cegar.py).

  A. Proposition 3.4: (i) an allocation is EF1 for the majority valuation v_W (|W| even) iff it splits W evenly
     (all allocations, all W, m <= 8); (ii) the counting bound is consistent with exact minimum covers of the
     family {v_W} by allocations for m = 4, 5, 6.
  B. Own-bundle balance fails at m = 13: recompute the block-compatible matchings S, check the three forms I-III
     of the hand proof, and check the counting claims used in the case analysis; plus a full enumeration of the
     market-block-balanced fair allocations (3,888 -> 206 -> 0).
  C. Twelve-good disjoint-defect profile: enumerate all 3^12 allocations, count the subjectively fair and
     near-block-balanced ones and their defect sets, check that no two near-block ones have disjoint defects,
     and check by linear programming that the stated pair (A, B) covers all additive market valuations.
Run:  python audit_extra.py [A|B|C|all]     (writes ../results/audit_extra.json)
"""
import itertools
import json
import sys
from collections import Counter
from math import comb

import numpy as np
from scipy.optimize import linprog


# ------------------------------------------------------------------------------------------------
# A. Proposition 3.4
# ------------------------------------------------------------------------------------------------
def audit_A():
    out = {}
    for m in range(2, 9):
        n_alloc = 1 << m
        X = ((np.arange(n_alloc)[:, None] >> np.arange(m)) & 1).astype(int)          # 1 -> agent 1 holds the good
        checked, bad = 0, 0
        for mask in range(1, 1 << m):
            W = [g for g in range(m) if (mask >> g) & 1]
            if len(W) % 2:
                continue
            h = len(W) // 2
            # direct EF1 test for v_W(S) = [ |S cap W| >= h ] : agent i is EF1 towards j iff exists g in A_j with v(A_i) >= v(A_j - g)
            for x in range(n_alloc):
                A1 = [g for g in range(m) if X[x, g]]
                A2 = [g for g in range(m) if not X[x, g]]
                cW1 = sum(1 for g in A1 if g in W)
                cW2 = len(W) - cW1
                v = lambda c: 1 if c >= h else 0
                ok2 = (not A1) or any(v(cW2) >= v(cW1 - (1 if g in W else 0)) for g in A1)     # agent 2 towards agent 1
                ok1 = (not A2) or any(v(cW1) >= v(cW2 - (1 if g in W else 0)) for g in A2)     # agent 1 towards agent 2
                checked += 1
                if (ok1 and ok2) != (cW1 == cW2):
                    bad += 1
        out[f"m={m}"] = {"instances": checked, "violations_of_'EF1 iff split evenly'": bad,
                         "bound": (2 ** (m - 1) - 1) / (comb(m, m // 2) - 1) if comb(m, m // 2) > 1 else None}
    # exact minimum number of allocations that split every even set W (|W| >= 2), by brute-force set cover
    for m in (4, 5, 6):
        n_alloc = 1 << m
        sets = [mask for mask in range(1, 1 << m) if bin(mask).count("1") % 2 == 0]
        cover_of = []
        for x in range(n_alloc):
            cs = 0
            for i, mask in enumerate(sets):
                c1 = bin(mask & x).count("1")
                if 2 * c1 == bin(mask).count("1"):
                    cs |= 1 << i
            cover_of.append(cs)
        full = (1 << len(sets)) - 1
        k = 1
        while True:
            if any(_or_all(c) == full for c in itertools.combinations(cover_of, k)):
                break
            k += 1
        out[f"m={m}"]["exact_min_family_that_splits_every_even_set"] = k
    return out


def _or_all(cs):
    r = 0
    for c in cs:
        r |= c
    return r


# ------------------------------------------------------------------------------------------------
# B. thirteen goods
# ------------------------------------------------------------------------------------------------
def audit_B():
    sigma0 = list(range(13))
    s1 = [0, 1, 3, 12, 2, 6, 9, 10, 4, 7, 5, 8, 11]
    s2 = [3, 2, 0, 1, 6, 5, 12, 4, 8, 11, 7, 9, 10]
    m = 13
    rank1 = {g: i for i, g in enumerate(s1)}
    # relabel by position in sigma_1
    lab = {g: rank1[g] for g in range(m)}
    market_blocks = [[lab[g] for g in range(k, min(k + 3, m))] for k in range(0, m, 3)]
    s2r = [lab[g] for g in s2]
    s1_triples = [list(range(k, min(k + 3, m))) for k in range(0, m, 3)]
    res = {"market_blocks_relabelled": market_blocks, "sigma2_relabelled": s2r}
    # (1) all S: at most one good per market block, exactly one per full s1 triple (and <=1 in the short one),
    #     with |S| = 4 or 5 forced; enumerate
    Ss = []
    for choice in itertools.product(*[[None] + b for b in market_blocks]):
        S = [c for c in choice if c is not None]
        if len(set(S)) != len(S):
            continue
        if any(len([g for g in t if g in S]) > 1 for t in s1_triples):
            continue
        # full triples need exactly one, and full market blocks exactly one
        if any(len([g for g in t if g in S]) != 1 for t in s1_triples[:4]):
            continue
        if any(len([g for g in b if g in S]) != 1 for b in market_blocks[:4]):
            continue
        Ss.append(tuple(sorted(S)))
    res["number_of_block_compatible_S"] = len(Ss)
    form = Counter()
    for S in Ss:
        Sset = set(S)
        if 3 in Sset and 12 in Sset and 8 in Sset and Sset & {0, 1} and Sset & {9, 11} and len(S) == 5:
            form["I"] += 1
        elif {2, 4} <= Sset and Sset & {6, 7} and Sset & {9, 11} and len(S) == 4:
            form["II"] += 1
        elif 5 in Sset and 10 in Sset and Sset & {0, 1} and Sset & {6, 7} and len(S) == 4:
            form["III"] += 1
        else:
            form["other"] += 1
    res["forms"] = dict(form)
    # (2) direct claims for each form
    claims = {"I": {"first11_of_s2_contain_all_5_S": True, "max_X_in_first11": 0},
              "II": {"first2_of_s2_in_S": True}, "III": {}}
    def prefix(k):
        return set(s2r[:k])
    ok = True
    for S in Ss:
        Sset = set(S)
        if len(S) == 5:   # form I
            ok &= Sset <= prefix(11)
        elif 2 in Sset and 4 in Sset:   # form II
            ok &= set(s2r[:2]) <= Sset
    res["claims_I_and_II_hold_for_all_S"] = bool(ok)
    # (3) full enumeration of market-block-balanced fair allocations
    blocks = market_blocks
    per_block = [list(itertools.permutations(range(3), len(b))) for b in blocks]
    n_bal = n_fair = n_own = 0
    pos1 = {g: i for i, g in enumerate(range(m))}      # sigma_1 is the identity after relabelling
    for choice in itertools.product(*per_block):
        owner = [None] * m
        for b, assign in zip(blocks, choice):
            for g, a in zip(b, assign):
                owner[g] = a
        n_bal += 1
        def fair(seq, i):
            cnt = [0, 0, 0]
            for g in seq:
                cnt[owner[g]] += 1
                if any(cnt[j] > cnt[i] + 1 for j in range(3) if j != i):
                    return False
            return True
        if fair(list(range(m)), 1) and fair(s2r, 2) and fair(sigma0_relabelled(lab), 0):
            n_fair += 1
            S = [g for g in range(m) if owner[g] == 1]
            if all(sum(1 for g in t if g in S) == 1 for t in s1_triples[:4]) and sum(1 for g in s1_triples[4] if g in S) <= 1:
                n_own += 1
    res["market_block_balanced"] = n_bal
    res["balanced_and_subjectively_SD-EF1"] = n_fair
    res["and_own_bundle_balanced"] = n_own
    return res


def sigma0_relabelled(lab):
    return [lab[g] for g in range(13)]


# ------------------------------------------------------------------------------------------------
# C. twelve-good profile, three agents
# ------------------------------------------------------------------------------------------------
def audit_C():
    m = 12
    s0 = [0, 7, 2, 5, 1, 11, 8, 3, 6, 9, 4, 10]
    s1 = [0, 11, 2, 7, 1, 5, 10, 9, 3, 8, 6, 4]
    s2 = [5, 0, 2, 11, 7, 1, 4, 3, 9, 8, 6, 10]
    sig = [s0, s1, s2]
    N = 3 ** m
    idx = np.arange(N)
    O = np.zeros((N, m), dtype=np.int8)
    t = idx.copy()
    for g in range(m):
        O[:, g] = t % 3
        t //= 3
    ok = np.ones(N, dtype=bool)
    for i in range(3):
        seq = O[:, sig[i]]
        cnt = np.stack([np.cumsum(seq == j, axis=1) for j in range(3)], axis=2)      # N x m x 3
        for j in range(3):
            if j != i:
                ok &= (cnt[:, :, j] - cnt[:, :, i]).max(axis=1) <= 1
    S = np.where(ok)[0]
    OS = O[S]                                                           # market order = identity 0..11
    cnt = np.stack([np.cumsum(OS == j, axis=1) for j in range(3)], axis=2)   # prefix counts, market order
    diff = cnt.max(axis=2) - cnt.min(axis=2)                            # after t goods = index t-1
    near = np.ones(len(S), dtype=bool)
    for t_ in range(1, m + 1):
        if t_ % 3 != 0:
            near &= diff[:, t_ - 1] <= 1
    near &= True
    defects = []
    for k in np.where(near)[0]:
        dset = frozenset(t_ for t_ in range(3, m, 3) if diff[k, t_ - 1] == 2)
        defects.append(dset)
    hist = Counter(tuple(sorted(d)) for d in defects)
    disjoint = sum(1 for a, b in itertools.combinations(defects, 2) if not (a & b))
    res = {"allocations": int(N), "subjectively_SD-EF1": int(len(S)), "near_block_balanced": int(near.sum()),
           "defect_set_histogram": {str(k): v for k, v in sorted(hist.items())},
           "disjoint_pairs_among_near_block": int(disjoint)}
    # the stated cover A, B (owner vectors)
    A = np.array([1, 0, 2, 1, 2, 2, 0, 0, 2, 1, 0, 1])
    B = np.array([1, 1, 0, 2, 0, 2, 1, 2, 0, 2, 1, 0])
    def in_S(o):
        for i in range(3):
            cn = [0, 0, 0]
            for g in sig[i]:
                cn[o[g]] += 1
                if any(cn[j] > cn[i] + 1 for j in range(3) if j != i):
                    return False
        return True
    res["A_in_S"] = bool(in_S(A))
    res["B_in_S"] = bool(in_S(B))
    res["B_near_block_balanced"] = bool(all(((np.bincount(B[:t_], minlength=3).max() - np.bincount(B[:t_], minlength=3).min()) <= 1)
                                            for t_ in range(1, m + 1) if t_ % 3))
    # additive cover check by LP: violation rows c_ij(t) = a_j(t) - a_i(t) - [a_j(t) > 0], v = sum_t Delta_t 1_{[0,t)}, Delta >= 0
    def rows(o):
        cn = np.zeros((m, 3), dtype=int)
        for t_ in range(1, m + 1):
            cn[t_ - 1] = np.bincount(o[:t_], minlength=3)
        R = []
        for i in range(3):
            for j in range(3):
                if i != j and (o == j).any():
                    R.append(np.array([cn[t_ - 1, j] - cn[t_ - 1, i] - (1 if cn[t_ - 1, j] > 0 else 0) for t_ in range(1, m + 1)], dtype=float))
        return R
    RA, RB = rows(A), rows(B)
    fails = False
    for ra in RA:
        for rb in RB:
            # exists Delta >= 0, sum Delta = 1, ra.Delta >= eps, rb.Delta >= eps with eps > 0 ?
            c = np.r_[np.zeros(m), -1.0]
            Aub = np.array([np.r_[-ra, 1.0], np.r_[-rb, 1.0]])
            r = linprog(c, A_ub=Aub, b_ub=[0, 0], A_eq=[np.r_[np.ones(m), 0.0]], b_eq=[1.0],
                        bounds=[(0, None)] * m + [(None, 1.0)], method="highs")
            if r.status == 0 and -r.fun > 1e-9:
                fails = True
    res["stated_pair_covers_all_additive_valuations"] = (not fails)
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}
    if which in ("A", "all"):
        out["A_proposition_3_4"] = audit_A()
    if which in ("B", "all"):
        out["B_thirteen_goods"] = audit_B()
    if which in ("C", "all"):
        out["C_twelve_good_defects"] = audit_C()
    print(json.dumps(out, indent=1, default=str))
    json.dump(out, open("../results/audit_extra.json", "w"), indent=1, default=str)
