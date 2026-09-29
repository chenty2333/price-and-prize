"""Independent check (our own code) of the monotone-market extension of the
n = 2 theorem, and of the ten-good pair-selection obstruction.

Part 1. For two agents, our cycle-flip reversal path (n2_constructive: pair
graph of consecutive sigma_1 / sigma_2 pairs, flip cycles starting at the
sigma_2-earliest blue pair) is computed from the rankings alone. For random
MONOTONE (non-additive) market valuations v we check that
  (a) every state of the path is subjectively SD-EF1 (fd), and
  (b) some state is market EF1 in the monotone sense:
      for i != j with A_j nonempty, v(A_i) >= v(A_j minus {g}) for some g in A_j,
  (c) no consecutive pair of states jumps from "agent 2 fails" to "agent 1 fails"
      or back (the monotone no-jump lemma).
Part 2. The ten-good obstruction: counts of SD-EF1 / near-block allocations,
defect-set histogram, the claims about selection rules, and the covering
pair (re-checked with the LP-only search of verify_certificates.refutes)."""
import itertools
import json
from collections import Counter

import numpy as np

import fd
from n2_constructive import pair_graph, cycles
from verify_certificates import refutes


# ---------------- Part 1 ----------------
def reversal_path(m, s1, s2):
    M, red, blue, pos2 = pair_graph(m, s1, s2)
    cyc = cycles(M, red, blue)
    owner = {}
    for c in cyc:
        for k, g in enumerate(c):
            owner[g] = k % 2
    states = [dict(owner)]
    for c in cyc:
        L = len(c)
        if L == 2:
            seq = [(c[0], c[1])]
        else:
            blues = [(c[k], c[(k + 1) % L]) for k in range(1, L, 2)]
            first = min(blues, key=lambda e: min(pos2[e[0]], pos2[e[1]]))
            c1 = first[0] if owner[first[0]] == 0 else first[1]
            order, u, use_red = [c1], c1, True
            while len(order) < L:
                u = red[u] if use_red else blue[u]
                use_red = not use_red
                order.append(u)
            seq = [(order[k], order[k + 1]) for k in range(0, L, 2)]
        for g, h in seq:
            owner[g], owner[h] = owner[h], owner[g]
            states.append(dict(owner))
    return [np.array([st[g] for g in range(m)], dtype=np.int8) for st in states]


def ef1_fail(alloc, v, i, j):
    """True iff agent i fails market EF1 towards j (monotone v on frozensets)."""
    Ai = frozenset(np.nonzero(alloc == i)[0].tolist())
    Aj = frozenset(np.nonzero(alloc == j)[0].tolist())
    if not Aj:
        return False
    return all(v(Ai) < v(Aj - {g}) for g in Aj)


def monotone_family(m, rng, kind):
    if kind == "table":  # random monotone table via max-closure (m <= 10)
        r = {S: rng.random() for S in range(1 << m)}
        val = {}
        for S in range(1 << m):  # increasing order of popcount not needed: max over submasks
            best = r[S]; T = S
            while T:
                T = (T - 1) & S
                best = max(best, r[T])
            val[S] = best
        return lambda A: val[sum(1 << g for g in A)]
    if kind == "xos":
        W = rng.random((3, m)) ** 2
        return lambda A: max(float(W[k, list(A)].sum()) if A else 0.0 for k in range(3))
    if kind == "budget":
        w = rng.random(m); B = w.sum() * rng.uniform(0.2, 0.6)
        return lambda A: min(B, float(w[list(A)].sum()) if A else 0.0)
    if kind == "coverage":
        sets = [set(rng.choice(12, rng.integers(1, 5), replace=False).tolist()) for _ in range(m)]
        return lambda A: float(len(set().union(*[sets[g] for g in A]))) if A else 0.0
    if kind == "concave":
        w = rng.random(m)
        return lambda A: float(np.sqrt(w[list(A)].sum())) if A else 0.0
    if kind == "supermodular":
        w = rng.random(m)
        return lambda A: float(w[list(A)].sum() ** 2) if A else 0.0
    raise ValueError(kind)


def part1(rng):
    stats = Counter()
    for t in range(3000):
        small = t % 3 == 0
        m = int(rng.integers(2, 11)) if small else int(rng.integers(2, 25))
        s1 = [int(x) for x in rng.permutation(m)]; s2 = [int(x) for x in rng.permutation(m)]
        path = reversal_path(m, s1, s2)
        sd = fd.subjective_sd_ef1(np.array(path), [s1, s2])
        stats["states"] += len(path); stats["states_not_sd"] += int((~sd).sum())
        kinds = ["table", "xos", "budget", "coverage", "concave", "supermodular"] if small else \
                ["xos", "budget", "coverage", "concave", "supermodular"]
        for kind in kinds:
            v = monotone_family(m, rng, kind)
            U = [ef1_fail(a, v, 1, 0) for a in path]   # agent 2 (label 1) fails
            L = [ef1_fail(a, v, 0, 1) for a in path]   # agent 1 (label 0) fails
            ok = [not (u or l) for u, l in zip(U, L)]
            stats["instances"] += 1
            stats["no_ef1_state"] += (not any(ok))
            stats["jumps"] += sum((U[k] and L[k + 1]) or (L[k] and U[k + 1]) for k in range(len(path) - 1))
            stats["both_fail_same_state"] += sum(u and l for u, l in zip(U, L))
    return dict(stats)


# ---------------- Part 2 ----------------
def part2():
    P = [[7, 4, 2, 5, 0, 8, 9, 6, 3, 1],
         [4, 2, 0, 3, 7, 5, 9, 1, 8, 6],
         [4, 0, 3, 2, 7, 5, 9, 1, 8, 6]]
    m = 10
    alloc = fd.all_allocations(3, m)
    S = alloc[fd.subjective_sd_ef1(alloc, P)]
    near, D = [], []
    for a in S:
        c = [0, 0, 0]; d = set(); good = True
        for t, i in enumerate(a, 1):
            c[i] += 1
            if t % 3 and max(c) - min(c) > 1:
                good = False; break
            if t % 3 == 0 and len(set(c)) > 1:
                d.add(t)
        if good:
            near.append(a); D.append(frozenset(d))
    hist = Counter(tuple(sorted(d)) for d in D)
    market_balanced = int(fd.subjective_sd_ef1(fd.market_sd_ef1_allocations(3, m), P).sum())
    lex = min(hist)
    lex_partners = sum(1 for d in D if d.isdisjoint(lex))
    pairs = [(i, j) for i in range(len(near)) for j in range(i + 1, len(near)) if D[i].isdisjoint(D[j])]
    pair_types = Counter(tuple(sorted([tuple(sorted(D[i])), tuple(sorted(D[j]))])) for i, j in pairs)
    A = [0, 1, 2, 2, 1, 1, 0, 0, 2, 0]; B = [1, 0, 1, 2, 0, 2, 0, 2, 0, 1]
    return {"sd_ef1": int(len(S)), "near_block": len(near), "market_balanced_in_S": market_balanced,
            "defect_hist": {str(k): v for k, v in sorted(hist.items())},
            "lex_first": list(lex), "lex_first_partners": lex_partners,
            "disjoint_pair_types": {str(k): v for k, v in pair_types.items()},
            "given_pair_sd_ef1": bool(fd.subjective_sd_ef1(np.array([A, B], dtype=np.int8), P).all()),
            "given_pair_LP_cover": refutes([A, B], 3, m)[0]}


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    out = {"part1_monotone_two_agents": part1(rng), "part2_ten_good_obstruction": part2()}
    print(json.dumps(out, indent=1))
    json.dump(out, open("../results/verify_monotone_ours.json", "w"), indent=1)
