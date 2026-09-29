"""Constructive algorithm for OQ1 with n = 2 (proof in Section 3 of the paper), and
an independent checker run on many instances.

Algorithm
1. Pair consecutive items of sigma1 (red edges) and of sigma2 (blue edges);
   append a dummy item at the end of both rankings if m is odd. The union is a
   disjoint union of even cycles (Kyropoulou-Suksompong-Voudouris, TCS 2020,
   Lemma 4.1); a proper 2-colouring (V1, V2) splits every red and blue pair.
2. Give agent 1 the bundle with larger market value. Both assignments of
   (V1, V2) are subjectively SD-EF1 for both agents.
3. Flip the cycles one at a time. Within a cycle, start at its sigma2-earliest
   blue pair {c_2L, c_1} with c_1 currently held by agent 1, then swap the red
   pairs (c_1,c_2), (c_3,c_4), ... in order. Every intermediate state stays
   subjectively SD-EF1 (checked here by assertion). After all flips the bundles
   are exchanged, so the market difference D = v(A1) - v(A2) changes sign;
   by the window lemma, some state on the way is EF1 w.r.t. v. Return the first one.
"""
import sys
import time

import numpy as np

import fd


def pair_graph(m, s1, s2):
    M = m + (m % 2)
    a = list(s1) + ([m] if m % 2 else [])
    b = list(s2) + ([m] if m % 2 else [])
    red, blue = {}, {}
    for t, E in ((a, red), (b, blue)):
        for k in range(0, M, 2):
            E[t[k]] = t[k + 1]; E[t[k + 1]] = t[k]
    pos2 = {g: k for k, g in enumerate(b)}
    return M, red, blue, pos2


def cycles(M, red, blue):
    seen, out = set(), []
    for s in range(M):
        if s in seen:
            continue
        cyc, u, use_red = [s], s, True
        seen.add(s)
        while True:
            w = red[u] if use_red else blue[u]
            use_red = not use_red
            if w == s:
                break
            cyc.append(w); seen.add(w); u = w
        out.append(cyc)  # alternating red/blue, starting with a red edge at s
    return out


def check_S(owner, m, s1, s2):
    x = np.array([[owner[g] for g in range(m)]], dtype=np.int8)
    return bool(fd.sd_ef1_pair(x, 0, 1, s1)[0] and fd.sd_ef1_pair(x, 1, 0, s2)[0])


def market_ef1(owner, m, v):
    x = np.array([[owner[g] for g in range(m)]], dtype=np.int8)
    return bool(fd.market_ef1(x, v, 2, tol=1e-12)[0])


def solve(m, s1, s2, v, check=True):
    M, red, blue, pos2 = pair_graph(m, s1, s2)
    cyc = cycles(M, red, blue)
    owner = {}
    for c in cyc:
        for k, g in enumerate(c):
            owner[g] = k % 2  # agent 0 / agent 1 alternate along the cycle
    val = [sum(v[g] for g in range(m) if owner[g] == a) for a in (0, 1)]
    if val[0] < val[1]:
        owner = {g: 1 - o for g, o in owner.items()}
    steps = 0
    if check:
        assert check_S(owner, m, s1, s2)
    if market_ef1(owner, m, v):
        return owner, steps
    for c in cyc:
        L = len(c)
        if L == 2:  # red pair == blue pair
            seq = [(c[0], c[1])]
        else:
            # blue edges of the cycle: (c[k], c[k+1]) for odd k, and (c[-1], c[0])
            blues = [(c[k], c[(k + 1) % L]) for k in range(1, L, 2)]
            first = min(blues, key=lambda e: min(pos2[e[0]], pos2[e[1]]))
            c1 = first[0] if owner[first[0]] == 0 else first[1]
            # walk the cycle starting at c1 along its red edge
            order, u, use_red = [c1], c1, True
            while len(order) < L:
                u = red[u] if use_red else blue[u]
                use_red = not use_red
                order.append(u)
            seq = [(order[k], order[k + 1]) for k in range(0, L, 2)]
        for g, h in seq:
            owner[g], owner[h] = owner[h], owner[g]
            steps += 1
            if check:
                assert check_S(owner, m, s1, s2), "left S"
            if market_ef1(owner, m, v):
                return owner, steps
    raise AssertionError("no EF1 state found; contradicts the proof")


def random_v(m, rng):
    kind = rng.integers(4)
    if kind == 0:
        v = rng.random(m)
    elif kind == 1:
        k = rng.integers(1, m + 1); v = np.r_[np.ones(k), np.zeros(m - k)]
    elif kind == 2:
        v = rng.random() ** np.arange(m)
    else:
        v = rng.exponential(size=m) ** 3
    return np.sort(v)[::-1]


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    rng = np.random.default_rng(2026)
    t0, n_inst, max_steps = time.time(), 0, 0
    per_m = {}
    while time.time() - t0 < budget:
        m = int(rng.integers(2, 41))
        s1, s2 = list(rng.permutation(m)), list(rng.permutation(m))
        for _ in range(5):
            v = random_v(m, rng)
            owner, steps = solve(m, s1, s2, v)
            x = np.array([[owner[g] for g in range(m)]], dtype=np.int8)
            assert fd.subjective_sd_ef1(x, [s1, s2])[0] and fd.market_ef1(x, v, 2)[0]
            n_inst += 1; max_steps = max(max_steps, steps)
            per_m[m] = per_m.get(m, 0) + 1
    print(f"verified {n_inst} (profile, v) instances, m in [2,40], all intermediate states in S; "
          f"max swaps before hitting EF1 = {max_steps}; min per-m count = {min(per_m.values())}")
