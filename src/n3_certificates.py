"""Which structured families of allocations already refute every market v?

For a finite family F of subjectively SD-EF1 allocations, run CEGAR with an
enumeration oracle over F: F "covers" the profile if no market valuation
makes every member of F violate market EF1.

Families (greedy picking: each agent takes its favourite remaining item):
  RR_fixed : round robin with one fixed agent order, all n! orders
  R_rounds : every sequence of per-round agent orders (n!^(#rounds))
Both families lie in S (a round-based greedy sequence is SD-EF1).

Tested on the hard n = 3 profiles (results/n3_search.json) and on hard n = 2
profiles (m = 7, 8), and also reports whether each CEGAR certificate from
cegar.decide consists of R_rounds members.
"""
import itertools
import json

import numpy as np

import cegar
import fd


def greedy(m, sigmas, round_orders):
    n = len(sigmas)
    left = set(range(m)); a = np.full(m, -1, dtype=np.int8)
    ptr = [0] * n
    for order in round_orders:
        for i in order:
            if not left:
                break
            while sigmas[i][ptr[i]] not in left:
                ptr[i] += 1
            g = sigmas[i][ptr[i]]; left.remove(g); a[g] = i
    return a


def family(m, sigmas, kind):
    n = len(sigmas)
    rounds = -(-m // n)
    perms = list(itertools.permutations(range(n)))
    if kind == "RR_fixed":
        seqs = [[p] * rounds for p in perms]
    else:
        seqs = itertools.product(perms, repeat=rounds)
    out = {tuple(greedy(m, sigmas, s)) for s in seqs}
    return np.array(sorted(out), dtype=np.int8)


def max_violation(F, v, n):
    v = np.asarray(v)
    val = np.stack([(F == k) @ v for k in range(n)], axis=1)
    top = np.zeros_like(val)
    for k in range(n):
        has = F == k
        top[:, k] = np.where(has.any(axis=1), v[has.argmax(axis=1)], 0.0)
    worst = np.full(len(F), -np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                worst = np.maximum(worst, val[:, j] - top[:, j] - val[:, i])
    return worst


def covers(F, n, m, tol=1e-6):
    Sp = [F[0]]
    for _ in range(500):
        eps, v = cegar.master(Sp, n, m)
        if eps <= tol:
            return True, len(Sp)
        mv = max_violation(F, v, n)
        k = int(mv.argmin())
        if mv[k] > tol:
            return False, len(Sp)
        Sp.append(F[k])
    raise RuntimeError


def main():
    out = []
    hard = json.load(open("../results/n3_search.json"))
    cases = [(3, int(mk), ex["sigmas"]) for mk, rec in hard.items() for ex in rec["hard_examples"]]
    # n = 2 hard profiles (m = 7, 8), a few of each
    rng = np.random.default_rng(0)
    for m in (7, 8):
        perms = list(itertools.permutations(range(m)))
        Mal = fd.market_sd_ef1_allocations(2, m)
        found = 0
        while found < 8:
            s = [list(perms[rng.integers(len(perms))]) for _ in range(2)]
            if not fd.subjective_sd_ef1(Mal, s).any():
                cases.append((2, m, s)); found += 1
    for n, m, s in cases:
        rec = {"n": n, "m": m}
        for kind in ("RR_fixed", "R_rounds"):
            F = family(m, s, kind)
            assert fd.subjective_sd_ef1(F, s).all()
            rec[kind] = covers(F, n, m)[0]
            rec[kind + "_size"] = len(F)
        r = cegar.decide(n, m, s)
        Rset = {tuple(a) for a in family(m, s, "R_rounds")}
        rec["cert_size"] = r["cert_size"]
        rec["cert_in_R_rounds"] = sum(tuple(a) in Rset for a in r["certificate"])
        out.append(rec)
        print(rec, flush=True)
    json.dump(out, open("../results/n3_certificates.json", "w"), indent=1)
    for n in (2, 3):
        rs = [r for r in out if r["n"] == n]
        print(f"n={n}: RR_fixed covers {sum(r['RR_fixed'] for r in rs)}/{len(rs)}, "
              f"R_rounds covers {sum(r['R_rounds'] for r in rs)}/{len(rs)}")


if __name__ == "__main__":
    main()
