"""Randomised check of the appendix theorem: n >= 2 agents, m = q n + r goods in market blocks of n consecutive
goods; if every agent except possibly one (agent f) has a block-contiguous ranking, the following rule gives a
market-block-balanced, subjectively SD-EF1 allocation:
  the short block T (r goods) is given bijectively to a set R of r agents other than f; H = the other agents
  except f; in every full block f picks its favourite remaining good, then the agents of H, then those of R.
Also checks the reverse-transposition style corner cases (r = 0, r = n-1, n = 2).  Writes ../results/verify_block_contiguous.json.
"""
import json
import random


def rank_contiguous(m, n, rng):
    blocks = [list(range(k, min(k + n, m))) for k in range(0, m, n)]
    rng.shuffle(blocks)
    out = []
    for b in blocks:
        b = b[:]
        rng.shuffle(b)
        out += b
    return out


def run_instance(n, m, rng):
    f = rng.randrange(n)
    sigma = []
    for i in range(n):
        sigma.append(rng.sample(range(m), m) if i == f else rank_contiguous(m, n, rng))
    q, r = divmod(m, n)
    others = [i for i in range(n) if i != f]
    rng.shuffle(others)
    R = others[:r]
    H = others[r:]
    owner = [None] * m
    pos = [{g: k for k, g in enumerate(s)} for s in sigma]
    for b in range(q):
        block = list(range(b * n, b * n + n))
        for i in [f] + H + R:
            g = min((g for g in block if owner[g] is None), key=lambda g: pos[i][g])
            owner[g] = i
    if r:
        T = list(range(q * n, m))
        rng.shuffle(T)
        for g, i in zip(T, R):
            owner[g] = i
    # market-block balance
    for k in range(0, m, n):
        blk = [owner[g] for g in range(k, min(k + n, m))]
        if len(set(blk)) != len(blk):
            return "not block balanced"
        if k + n <= m and len(set(blk)) != n:
            return "full block not rainbow"
    # subjective SD-EF1
    for i in range(n):
        cnt = [0] * n
        for g in sigma[i]:
            cnt[owner[g]] += 1
            if any(cnt[j] > cnt[i] + 1 for j in range(n) if j != i):
                return f"agent {i} not SD-EF1"
    return None


if __name__ == "__main__":
    rng = random.Random(20260930)
    fails, total = [], 0
    by_n = {}
    for _ in range(30000):
        n = rng.randrange(2, 7)
        m = rng.randrange(n, 45)
        res = run_instance(n, m, rng)
        total += 1
        by_n[n] = by_n.get(n, 0) + 1
        if res:
            fails.append((n, m, res))
    out = {"instances": total, "by_n": by_n, "failures": len(fails), "first_failures": fails[:5]}
    print(out)
    json.dump(out, open("../results/verify_block_contiguous.json", "w"), indent=1)
