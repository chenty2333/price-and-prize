"""Exact finite allocation predicates; numeric MILP discovery lives separately."""
from itertools import product, permutations
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
from numba import njit

@njit(cache=True)
def sd_one(owner, ranking, i, n):
    counts = np.zeros(n, dtype=np.int16)
    for g in ranking:
        j = owner[g]
        counts[j] += 1
        if j != i and counts[j] > counts[i] + 1:
            return False
    return True

@njit(cache=True)
def sd_all(owner, profile):
    n = len(profile)
    for i in range(n):
        if not sd_one(owner, profile[i], i, n):
            return False
    return True

@njit(cache=True)
def filter_sd(owners, profile):
    keep = np.empty(len(owners), dtype=np.int64)
    count = 0
    for k in range(len(owners)):
        if sd_all(owners[k], profile):
            keep[count] = k
            count += 1
    return keep[:count]

@njit(cache=True)
def single_mask(owners, ranking, i, n):
    result = np.empty(len(owners), dtype=np.bool_)
    for k in range(len(owners)):
        result[k] = sd_one(owners[k], ranking, i, n)
    return result

@njit(cache=True)
def market_block(owner, n):
    seen = np.zeros(n, dtype=np.bool_)
    for g in range(len(owner)):
        if g % n == 0:
            seen[:] = False
        if seen[owner[g]]:
            return False
        seen[owner[g]] = True
    return True

@njit(cache=True)
def balanced_universe(n, m):
    # Every n^m owner vector is visited; only full-prefix-feasible ones stored.
    total = n ** m
    capacity = 1
    # First pass counts; second fills. Pure integer, exact.
    lo, hi = m // n, (m + n - 1) // n
    owner = np.zeros(m, dtype=np.uint8)
    counts = np.zeros(n, dtype=np.int16)
    counts[0] = m
    k = 0
    for t in range(total):
        good = True
        for i in range(n):
            if counts[i] < lo or counts[i] > hi:
                good = False
                break
        if good:
            k += 1
        for g in range(m):
            old = owner[g]
            counts[old] -= 1
            new = old + 1
            if new == n:
                new = 0
            owner[g] = new
            counts[new] += 1
            if new != 0:
                break
    out = np.empty((k, m), dtype=np.uint8)
    owner[:] = 0
    counts[:] = 0
    counts[0] = m
    k = 0
    for t in range(total):
        good = True
        for i in range(n):
            if counts[i] < lo or counts[i] > hi:
                good = False
                break
        if good:
            out[k] = owner
            k += 1
        for g in range(m):
            old = owner[g]
            counts[old] -= 1
            new = old + 1
            if new == n:
                new = 0
            owner[g] = new
            counts[new] += 1
            if new != 0:
                break
    return out

@lru_cache(None)
def universe(n, m):
    assert n ** m <= 10_000_000, (n, m)
    path = Path(__file__).parent / f'balanced_{n}_{m}.npy'
    if path.exists():
        return np.load(path)
    a = balanced_universe(n, m)
    np.save(path, a)
    return a

@lru_cache(None)
def blocks(n, m):
    parts = [list(permutations(range(n), min(n, m-start)))
             for start in range(0, m, n)]
    return np.array([sum(p, ()) for p in product(*parts)], dtype=np.uint8)

def sd_python(owner, profile):
    n = len(profile)
    for i, ranking in enumerate(profile):
        for t in range(1, len(owner)+1):
            counts = [sum(owner[g] == j for g in ranking[:t]) for j in range(n)]
            if any(counts[j] > counts[i]+1 for j in range(n)):
                return False
    return True

def market_python(owner, values, n):
    bundles = [[g for g, who in enumerate(owner) if who == i] for i in range(n)]
    sums = [sum((values[g] for g in b), 0) for b in bundles]
    return all(not bundles[j] or sums[i] >= sums[j] - max(values[g] for g in bundles[j])
               for i in range(n) for j in range(n) if i != j)

def two_level_rr(profile, t):
    """Top t goods form the first category. No floating-point arithmetic."""
    n, m = len(profile), len(profile[0])
    owner = [-1]*m
    for goods, order in [(set(range(t)), list(range(n))),
                         (set(range(t, m)), list(reversed(range(n))))]:
        while goods:
            for i in order:
                if not goods:
                    break
                g = next(g for g in profile[i] if g in goods)
                owner[g] = i
                goods.remove(g)
    return owner

@njit(cache=True)
def market_margins(owners, values, n):
    ans = np.empty(len(owners), dtype=np.float64)
    for k in range(len(owners)):
        sums = np.zeros(n, dtype=np.float64)
        largest = np.zeros(n, dtype=np.float64)
        for g in range(owners.shape[1]):
            who = owners[k, g]
            sums[who] += values[g]
            largest[who] = max(largest[who], values[g])
        worst = -1e100
        for i in range(n):
            for j in range(n):
                if i != j:
                    worst = max(worst, sums[j]-largest[j]-sums[i])
        ans[k] = worst
    return ans

@njit(cache=True)
def violations(owners, n):
    # D[k, pair, g] @ v > 0 is one market-EF1 violation.
    m = owners.shape[1]
    D = np.zeros((len(owners), n*(n-1), m), dtype=np.int16)
    for k in range(len(owners)):
        heads = np.full(n, -1, dtype=np.int16)
        for g in range(m):
            if heads[owners[k,g]] == -1:
                heads[owners[k,g]] = g
        p = 0
        for i in range(n):
            for j in range(n):
                if i == j: continue
                for g in range(m):
                    if owners[k,g] == i: D[k,p,g] = -1
                    elif owners[k,g] == j and g != heads[j]: D[k,p,g] = 1
                p += 1
    return D

if __name__ == '__main__':
    for n,m in [(2,7),(3,10),(3,12),(3,13),(3,14),(4,9),(4,10),(4,11),(5,9),(5,10)]:
        u, b = universe(n,m), blocks(n,m)
        print(n,m, 'all',n**m, 'balanced',len(u), 'block',len(b),flush=True)
    p=np.array([[0,2,1,4,3,6,5],[0,4,1,2,5,6,3]])
    inds=filter_sd(universe(2,7),p)
    print('Theorem 1 regression:', 'subjective',len(inds), 'market-SD',len(filter_sd(blocks(2,7),p)),flush=True)
    for a in product(range(2),repeat=7):
        assert sd_python(a,p.tolist()) == sd_all(np.array(a,dtype=np.uint8),p)
    print('Independent prefix checker agrees on all 128 allocations.',flush=True)
