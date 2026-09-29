"""n = 3: hardness-driven local search inside the glued families of n3_gluing.py.

The profile is kept inside its family by mutating the family's structural
parameters, not the raw rankings:
  lex(r,d): each agent's digit-significance order q_i and per-digit
            relabelings p_{i,l};
  thm1    : positions of the extra goods in agents 1-2's rankings (the
            Theorem 1 core order is preserved) and agent 3's ranking;
  concatK : K hard gadgets in market order; each agent's interleaving pattern
            of its K gadget rankings (mutation swaps two slots);
  deep    : start from concat2, then anneal the raw rankings (leaves the family).
Objective: cegar.market_imbalance (0 = easy; >= 1 = passes the necessary
condition for a counterexample), maximised with tie acceptance. Every newly
reached hard profile is decided exactly with cegar.decide.

usage: python n3_gluing_ls.py BUDGET_SECONDS [family,...]
"""
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import cegar
from n3_gluing import THM1, load_gadgets

N = 3


class Lex:
    def __init__(self, r, d, rng):
        self.r, self.d, self.rng = r, d, rng
        self.items = list(itertools.product(range(r), repeat=d))
        self.idx = {a: k for k, a in enumerate(self.items)}
        self.params = [(list(rng.permutation(d)), [list(rng.permutation(r)) for _ in range(d)])
                       for _ in range(N)]

    def profile(self):
        sig = []
        for q, p in self.params:
            key = lambda a: tuple(p[l][a[l]] for l in q)
            sig.append([self.idx[a] for a in sorted(self.items, key=key)])
        return self.r ** self.d, sig

    def mutate(self):
        new = Lex.__new__(Lex); new.__dict__.update(self.__dict__)
        new.params = [(list(q), [list(x) for x in p]) for q, p in self.params]
        i = self.rng.integers(N); q, p = new.params[i]
        if self.rng.random() < 0.3:
            a, b = self.rng.choice(self.d, 2, replace=False); q[a], q[b] = q[b], q[a]
        else:
            l = self.rng.integers(self.d); a, b = self.rng.choice(self.r, 2, replace=False)
            p[l][a], p[l][b] = p[l][b], p[l][a]
        return new


class Thm1:
    def __init__(self, rng):
        self.rng = rng
        k = int(rng.integers(3, 9)); self.m = 7 + k
        core_pos = sorted(rng.choice(self.m, 7, replace=False))
        self.core = [int(x) for x in core_pos]
        self.extra = [x for x in range(self.m) if x not in core_pos]
        self.rank12 = []
        for base in THM1:
            s = [self.core[c] for c in base]
            for e in rng.permutation(self.extra):
                s.insert(int(rng.integers(len(s) + 1)), int(e))
            self.rank12.append(s)
        self.rank3 = [int(x) for x in rng.permutation(self.m)]

    def profile(self):
        return self.m, [self.rank12[0], self.rank12[1], self.rank3]

    def mutate(self):
        new = Thm1.__new__(Thm1); new.__dict__.update(self.__dict__)
        new.rank12 = [list(s) for s in self.rank12]; new.rank3 = list(self.rank3)
        if self.rng.random() < 0.5:
            s = new.rank12[self.rng.integers(2)]
            e = self.extra[self.rng.integers(len(self.extra))]
            s.remove(e); s.insert(int(self.rng.integers(len(s) + 1)), e)
        else:
            a, b = self.rng.choice(self.m, 2, replace=False)
            new.rank3[a], new.rank3[b] = new.rank3[b], new.rank3[a]
        return new


class Concat:
    """k hard gadgets placed consecutively in market order; each agent merges
    its k gadget rankings according to its own interleaving pattern."""

    def __init__(self, rng, gadgets, k=2):
        self.rng, self.k = rng, k
        self.gs = [gadgets[rng.integers(len(gadgets))] for _ in range(k)]
        self.offsets = np.cumsum([0] + [g[0] for g in self.gs])[:-1].tolist()
        self.m = sum(g[0] for g in self.gs)
        self.patterns = []
        for _ in range(N):
            if rng.random() < 0.5:  # blocks in some order
                order = rng.permutation(k)
                self.patterns.append([int(b) for b in order for _ in range(self.gs[b][0])])
            else:
                pat = [b for b in range(k) for _ in range(self.gs[b][0])]
                self.patterns.append([int(x) for x in rng.permutation(pat)])

    def profile(self):
        sig = []
        for i in range(N):
            lists = [[self.offsets[b] + x for x in self.gs[b][1][i]] for b in range(self.k)]
            ptr = [0] * self.k
            out = []
            for b in self.patterns[i]:
                out.append(lists[b][ptr[b]]); ptr[b] += 1
            sig.append(out)
        return self.m, sig

    def mutate(self):
        new = Concat.__new__(Concat); new.__dict__.update(self.__dict__)
        new.patterns = [list(p) for p in self.patterns]
        p = new.patterns[self.rng.integers(N)]
        a, b = self.rng.choice(len(p), 2, replace=False)
        p[a], p[b] = p[b], p[a]  # swap two interleaving slots (keeps block sizes)
        return new


class Deep:
    """Start from a glued profile, then anneal the raw rankings (leaving the
    family) to push market_imbalance as high as possible."""

    def __init__(self, rng, gadgets):
        self.rng = rng
        c = Concat(rng, gadgets, k=2)
        self.m, self.sig = c.profile()

    def profile(self):
        return self.m, self.sig

    def mutate(self):
        new = Deep.__new__(Deep); new.__dict__.update(self.__dict__)
        new.sig = [list(s) for s in self.sig]
        s = new.sig[self.rng.integers(N)]
        a, b = self.rng.choice(self.m, 2, replace=False)
        s[a], s[b] = s[b], s[a]
        return new


def make(family, rng, gadgets):
    if family.startswith("lex"):
        r, d = map(int, family[3:].split("_"))
        return Lex(r, d, rng)
    if family == "thm1":
        return Thm1(rng)
    if family == "deep":
        return Deep(rng, gadgets)
    return Concat(rng, gadgets, k=int(family[-1]))


def worker(args):
    family, seed, budget = args
    rng = np.random.default_rng(seed)
    gadgets = load_gadgets()
    t0 = time.time()
    rec = {"family": family, "seed": seed, "steps": 0, "restarts": 0, "hard_distinct": 0,
           "max_imbalance": 0, "decided": 0, "refuted": 0, "counterexamples": [], "undecided": [],
           "cert_sizes": [], "decide_sec": []}
    path = f"../results/n3_gluing_ls_{family}_{seed}.json"
    seen = set()

    def save():
        with open(path, "w") as f:
            json.dump(rec, f)

    while time.time() - t0 < budget:
        rec["restarts"] += 1
        cur = make(family, rng, gadgets)
        m, sig = cur.profile()
        val = cegar.market_imbalance(N, m, sig)
        stall = 0
        while stall < 150 and time.time() - t0 < budget:
            cand = cur.mutate(); m, sig = cand.profile()
            rec["steps"] += 1
            v2 = cegar.market_imbalance(N, m, sig)
            if v2 > val or (v2 == val and rng.random() < 0.5):
                stall = 0 if v2 > val else stall + 1
                cur, val = cand, v2
            else:
                stall += 1
            rec["max_imbalance"] = max(rec["max_imbalance"], val)
            key = json.dumps(sig)
            if val >= 1 and key not in seen:
                seen.add(key); rec["hard_distinct"] += 1
                t = time.time()
                try:
                    r = cegar.decide(N, m, sig, rng=rng, max_rounds=60)
                except (cegar.SolverError, RuntimeError) as e:
                    rec["undecided"].append({"m": m, "sigmas": sig, "error": str(e)[:200]}); save()
                    continue
                rec["decide_sec"].append(round(time.time() - t, 1)); rec["decided"] += 1
                if r["result"] == "refuted":
                    rec["refuted"] += 1; rec["cert_sizes"].append(r["cert_size"])
                else:
                    rec["counterexamples"].append({"m": m, "sigmas": sig, "v": r["v"], "eps": r["eps"]})
                save()
        save()
    save()
    return rec


if __name__ == "__main__":
    budget = float(sys.argv[1])
    fams = sys.argv[2].split(",") if len(sys.argv) > 2 else \
        ["lex3_3", "lex2_4", "lex2_5", "lex4_2", "lex5_2", "thm1", "concat2"]
    per = max(1, os.cpu_count() // len(fams))
    jobs = [(f, 5000 + k, budget) for f in fams for k in range(per)]
    with Pool(min(len(jobs), os.cpu_count())) as p:
        res = p.map(worker, jobs)
    agg = {}
    for r in res:
        a = agg.setdefault(r["family"], {"steps": 0, "hard_distinct": 0, "max_imbalance": 0,
                                         "decided": 0, "refuted": 0, "counterexamples": 0,
                                         "undecided": 0, "max_cert": 0})
        for k in ("steps", "hard_distinct", "decided", "refuted"):
            a[k] += r[k]
        a["max_imbalance"] = max(a["max_imbalance"], r["max_imbalance"])
        a["counterexamples"] += len(r["counterexamples"]); a["undecided"] += len(r["undecided"])
        a["max_cert"] = max([a["max_cert"]] + r["cert_sizes"])
    for f, a in agg.items():
        print(f, a, flush=True)
    json.dump(agg, open("../results/n3_gluing_ls_summary.json", "w"), indent=1)
