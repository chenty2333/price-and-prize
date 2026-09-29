"""Exhaustive search for counterexamples to OQ1 over all ranking profiles.

usage: python exhaustive.py n m [--workers W]

Step 1 (necessary condition): a profile (sigma_1..sigma_n) can be a
counterexample only if no market-SD-EF1 allocation (EF1 for every market
valuation) is subjectively SD-EF1. For every ranking and agent role we
precompute a bitmask over the market-SD-EF1 allocations, so step 1 is a
bitwise AND. Agents are interchangeable, so only sorted index tuples are
enumerated.

Step 2: for each surviving profile, compute the full set S of subjectively
SD-EF1 allocations, deduplicate by S, and run the cutting-plane MILP of
fd.find_breaking_v. eps > 0 would be a counterexample.

This is an independent code path from oq1_probe.py (bitmask over all 2^m
partitions for n = 2); the two must agree for n = 2.
"""
import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import fd


def role_masks(n, m, perms, Mal):
    """bits[r][p] = packed bitmask of market-SD-EF1 allocations in which agent r
    with ranking perms[p] is SD-EF1 towards every other agent."""
    out = []
    for r in range(n):
        rows = []
        for p in perms:
            ok = np.ones(len(Mal), dtype=bool)
            for j in range(n):
                if j != r:
                    ok &= fd.sd_ef1_pair(Mal, r, j, list(p))
            rows.append(np.packbits(ok))
        out.append(np.array(rows))
    return out


_BITS = None


def _init(bits):
    global _BITS
    _BITS = bits


def _worker(args):
    n, i = args
    bits = _BITS
    P = bits[0].shape[0]
    hits = []
    if n == 2:
        z = ~((bits[0][i][None, :] & bits[1][i:]).any(axis=1))
        hits = [(i, i + k) for k in np.nonzero(z)[0]]
    else:  # n == 3, indices i <= j <= k
        for j in range(i, P):
            ab = bits[0][i] & bits[1][j]
            if not ab.any():
                hits.extend((i, j, k) for k in range(j, P))
                continue
            z = ~((ab[None, :] & bits[2][j:]).any(axis=1))
            hits.extend((i, j, j + k) for k in np.nonzero(z)[0])
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("m", type=int)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    n, m = args.n, args.m
    t0 = time.time()
    perms = list(itertools.permutations(range(m)))
    Mal = fd.market_sd_ef1_allocations(n, m)
    bits = role_masks(n, m, perms, Mal)
    with Pool(args.workers, initializer=_init, initargs=(bits,)) as pool:
        chunks = pool.map(_worker, [(n, i) for i in range(len(perms))], chunksize=8)
    hard = [h for c in chunks for h in c]
    print(f"n={n} m={m}: |market-SD-EF1 allocations|={len(Mal)}, "
          f"hard profiles (sorted tuples)={len(hard)}  [{time.time()-t0:.1f}s]", flush=True)

    alloc = fd.all_allocations(n, m)
    distinct = {}
    for h in hard:
        S = fd.subjective_sd_ef1(alloc, [list(perms[p]) for p in h])
        key = np.packbits(S).tobytes()
        distinct.setdefault(key, h)
    print(f"distinct S among hard profiles: {len(distinct)}  [{time.time()-t0:.1f}s]", flush=True)

    best, examples, sizes = 0.0, [], []
    for key, h in distinct.items():
        S = fd.subjective_sd_ef1(alloc, [list(perms[p]) for p in h])
        sizes.append(int(S.sum()))
        eps, v, _ = fd.find_breaking_v(alloc[S], n, m)
        best = max(best, eps)
        if eps > 1e-9:
            examples.append({"sigmas": [list(perms[p]) for p in h], "v": v.tolist(), "eps": eps})
    out = {"n": n, "m": m, "hard_profiles": len(hard), "distinct_S": len(distinct),
           "S_size_min": min(sizes) if sizes else None, "best_eps": best,
           "counterexamples": examples, "seconds": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in out.items() if k != "counterexamples"}),
          "counterexamples:", len(examples))
    os.makedirs("../results", exist_ok=True)
    with open(f"../results/exhaustive_n{n}_m{m}.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
