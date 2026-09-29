"""n = 3: structured ("glued") candidate counterexamples to OQ1.

Families (market order = item index order in all of them):
  lex(r,d)   items = strings in [r]^d, market = plain lexicographic order.
             Agent i reads the digits in its own significance order q_i and
             relabels each digit by p_{i,l} in S_r (a generalised
             Newman-Nikolov recursive construction; with r = 3 and a common
             digit order the last-digit colouring balances everything, so
             per-agent digit orders or r in {2, 4} are needed).
  concat     two hard n = 3 gadgets (results/n3_search.json, m = 10..12)
             placed one after the other in market order; each agent ranks
             the two gadgets in its own block order.
  thm1       the Theorem 1 gadget of Barman et al. (n = 2, 7 goods) for
             agents 1, 2, plus a third agent and k extra goods inserted at
             random market positions and random ranking positions.

Each candidate is screened with cegar.market_balanced_in_S (fast ILP; a
counterexample needs this to be False) and, if hard, decided exactly with
cegar.decide under a wall-clock cap. Workers save results incrementally.

usage: python n3_gluing.py screen            # hardness rates per family
       python n3_gluing.py search BUDGET     # screen + decide, BUDGET seconds per worker
"""
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import cegar

N = 3
THM1 = ([0, 2, 1, 4, 3, 6, 5], [0, 4, 1, 2, 5, 6, 3])  # Barman et al., Theorem 1


def lex_profile(r, d, rng):
    items = list(itertools.product(range(r), repeat=d))
    idx = {a: k for k, a in enumerate(items)}
    sig = []
    for _ in range(N):
        q = list(rng.permutation(d))
        p = [list(rng.permutation(r)) for _ in range(d)]
        key = lambda a: tuple(p[l][a[l]] for l in q)
        sig.append([idx[a] for a in sorted(items, key=key)])
    return r ** d, sig


def concat_profile(gadgets, rng):
    g1, g2 = gadgets[rng.integers(len(gadgets))], gadgets[rng.integers(len(gadgets))]
    m1, m2 = g1[0], g2[0]
    sig = []
    for i in range(N):
        a = list(g1[1][i]); b = [m1 + x for x in g2[1][i]]
        mode = rng.integers(3)
        if mode == 0:
            sig.append(a + b)
        elif mode == 1:
            sig.append(b + a)
        else:  # riffle the two block rankings
            out, ia, ib = [], 0, 0
            while ia < len(a) or ib < len(b):
                if ib >= len(b) or (ia < len(a) and rng.random() < 0.5):
                    out.append(a[ia]); ia += 1
                else:
                    out.append(b[ib]); ib += 1
            sig.append(out)
    return m1 + m2, sig


def thm1_profile(rng):
    k = int(rng.integers(3, 9))
    m = 7 + k
    market = list(rng.permutation(m))  # market rank of each good; core goods keep their order
    core_pos = sorted(rng.choice(m, 7, replace=False))
    extra_pos = [x for x in range(m) if x not in core_pos]
    core = {c: core_pos[c] for c in range(7)}  # core good c -> market index
    sig = []
    for base in THM1:
        s = [core[c] for c in base]
        for e in rng.permutation(extra_pos):
            s.insert(int(rng.integers(len(s) + 1)), int(e))
        sig.append(s)
    sig.append([int(x) for x in rng.permutation(m)])
    del market
    return m, sig


def gen(family, rng, gadgets):
    if family.startswith("lex"):
        r, d = map(int, family[3:].split("_"))
        return lex_profile(r, d, rng)
    if family == "concat":
        return concat_profile(gadgets, rng)
    return thm1_profile(rng)


FAMILIES = ["lex2_4", "lex2_5", "lex4_2", "lex3_3", "lex5_2", "concat", "thm1"]


def load_gadgets():
    hard = json.load(open("../results/n3_search.json"))
    return [(int(mk), ex["sigmas"]) for mk, rec in hard.items() for ex in rec["hard_examples"]]


def screen_worker(args):
    family, seed, count = args
    rng = np.random.default_rng(seed)
    gadgets = load_gadgets()
    hard = 0
    for _ in range(count):
        m, sig = gen(family, rng, gadgets)
        hard += not cegar.market_balanced_in_S(N, m, sig)
    return family, hard, count


def search_worker(args):
    family, seed, budget = args
    rng = np.random.default_rng(seed)
    gadgets = load_gadgets()
    t0 = time.time()
    rec = {"family": family, "seed": seed, "screened": 0, "hard": 0, "decided": 0,
           "refuted": 0, "counterexamples": [], "undecided": [], "cert_sizes": []}
    path = f"../results/n3_gluing_{family}_{seed}.json"
    while time.time() - t0 < budget:
        m, sig = gen(family, rng, gadgets)
        rec["screened"] += 1
        if cegar.market_balanced_in_S(N, m, sig):
            continue
        rec["hard"] += 1
        try:
            r = cegar.decide(N, m, sig, rng=rng, max_rounds=60)
        except (cegar.SolverError, RuntimeError) as e:
            rec["undecided"].append({"m": m, "sigmas": sig, "error": str(e)[:200]})
            continue
        rec["decided"] += 1
        if r["result"] == "refuted":
            rec["refuted"] += 1; rec["cert_sizes"].append(r["cert_size"])
        else:
            rec["counterexamples"].append({"m": m, "sigmas": sig, "v": r["v"], "eps": r["eps"]})
        with open(path, "w") as f:
            json.dump(rec, f)
    with open(path, "w") as f:
        json.dump(rec, f)
    return rec


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "screen":
        jobs = [(f, 100 + k, 25) for f in FAMILIES for k in range(2)]
        with Pool(min(len(jobs), os.cpu_count())) as p:
            res = p.map(screen_worker, jobs)
        agg = {}
        for f, h, c in res:
            a = agg.setdefault(f, [0, 0]); a[0] += h; a[1] += c
        for f, (h, c) in agg.items():
            print(f"{f}: hard {h}/{c}")
        json.dump(agg, open("../results/n3_gluing_screen.json", "w"), indent=1)
    else:
        budget = float(sys.argv[2])
        fams = sys.argv[3].split(",") if len(sys.argv) > 3 else FAMILIES
        per = max(1, os.cpu_count() // len(fams))
        jobs = [(f, 1000 + k, budget) for f in fams for k in range(per)]
        with Pool(min(len(jobs), os.cpu_count())) as p:
            res = p.map(search_worker, jobs)
        agg = {}
        for r in res:
            a = agg.setdefault(r["family"], {"screened": 0, "hard": 0, "decided": 0, "refuted": 0,
                                             "counterexamples": 0, "undecided": 0})
            for k in ("screened", "hard", "decided", "refuted"):
                a[k] += r[k]
            a["counterexamples"] += len(r["counterexamples"]); a["undecided"] += len(r["undecided"])
        for f, a in agg.items():
            print(f, a, flush=True)
        json.dump(agg, open("../results/n3_gluing_summary.json", "w"), indent=1)
