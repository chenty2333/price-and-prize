"""n = 3: counterexample search for OQ1 at m = 13..18 without enumerating S.

Per worker (fixed m), repeat until the time budget is spent:
  1. hardness phase: hill-climb (sigma_1, sigma_2, sigma_3) until
     h = #(market-SD-EF1 allocations that are subjectively SD-EF1) = 0
     (necessary for a counterexample; market-SD-EF1 allocations are
     enumerated, 6^(m/3) of them, so this is cheap);
  2. decide the profile exactly with cegar.decide (CEGAR, no enumeration of S);
  3. evolutionary phase: mutate the profile (random transpositions), keep h = 0,
     accept if the refutation got harder (more CEGAR rounds, then larger
     certificate), and decide each mutant exactly.
Every profile examined is decided exactly; guidance only affects which
profiles get examined. Counterexamples are re-verified by an independent
oracle solve with presolve disabled.

usage: python n3_cegar_search.py --ms 13,14,15,16,17,18 --per_m 2 --budget 2400
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import cegar
import fd

N = 3


def h_of(Mal, s):
    return int(fd.subjective_sd_ef1(Mal, s).sum())


def mutate(s, rng, m, k=1):
    t = [x[:] for x in s]
    for _ in range(k):
        r = rng.integers(N)
        a, b = rng.choice(m, 2, replace=False)
        t[r][a], t[r][b] = t[r][b], t[r][a]
    return t


def find_hard(Mal, m, rng, steps=4000):
    s = [list(map(int, rng.permutation(m))) for _ in range(N)]
    h = h_of(Mal, s)
    for _ in range(steps):
        if h == 0:
            return s
        t = mutate(s, rng, m)
        h2 = h_of(Mal, t)
        if h2 <= h or rng.random() < 0.01:
            s, h = t, h2
    return None


def worker(args):
    m, seed, budget = args
    rng = np.random.default_rng(seed)
    Mal = fd.market_sd_ef1_allocations(N, m)
    t0 = time.time()
    rec = {"m": m, "seed": seed, "hard_found": 0, "decided": 0, "counterexamples": [],
           "max_rounds": 0, "max_cert": 0, "hardest": None, "cert_hist": {}, "t_decide": []}

    out_path = f"../results/n3_cegar_search_w{m}_{seed}.json"
    last_dump = [time.time()]
    rec["undecided"] = []

    def dump(force=False):
        if force or time.time() - last_dump[0] > 60:
            with open(out_path, "w") as f:
                json.dump(rec, f)
            last_dump[0] = time.time()

    def decide(s):
        t = time.time()
        try:
            r = cegar.decide(N, m, s, rng=rng)
        except cegar.SolverError as e:
            rec["undecided"].append({"sigmas": s, "error": str(e)})
            dump(force=True)
            return {"result": "undecided", "rounds": -1, "cert_size": -1}
        rec["t_decide"].append(time.time() - t)
        rec["decided"] += 1
        rec["cert_hist"][r["cert_size"]] = rec["cert_hist"].get(r["cert_size"], 0) + 1
        if r["result"] == "counterexample":
            z, _ = cegar.Oracle(N, m, s).solve(np.array(r["v"]))
            rec["counterexamples"].append({"sigmas": s, "v": r["v"], "eps": r["eps"],
                                           "recheck_min_violation": z})
            dump(force=True)
        dump()
        return r

    while time.time() - t0 < budget:
        s = find_hard(Mal, m, rng)
        if s is None:
            continue
        rec["hard_found"] += 1
        r = decide(s)
        score = (r["rounds"], r["cert_size"])
        stall = 0
        while stall < 25 and time.time() - t0 < budget:
            t = mutate(s, rng, m, k=int(rng.integers(1, 3)))
            if h_of(Mal, t) != 0:
                stall += 1
                continue
            r2 = decide(t)
            sc2 = (r2["rounds"], r2["cert_size"])
            if sc2 >= score:
                if sc2 > score:
                    stall = 0
                s, score = t, sc2
            else:
                stall += 1
            if score > (rec["max_rounds"], rec["max_cert"]):
                rec["max_rounds"], rec["max_cert"] = score
                rec["hardest"] = {"sigmas": s, "rounds": score[0], "cert": score[1]}
    rec["t_decide_median"] = float(np.median(rec["t_decide"])) if rec["t_decide"] else None
    del rec["t_decide"]
    dump(force=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="13,14,15,16,17,18")
    ap.add_argument("--per_m", type=int, default=2)
    ap.add_argument("--budget", type=float, default=2400)
    args = ap.parse_args()
    ms = [int(x) for x in args.ms.split(",")]
    jobs = [(m, 31337 * m + k, args.budget) for m in ms for k in range(args.per_m)]
    t0 = time.time()
    with Pool(min(len(jobs), os.cpu_count())) as pool:
        res = pool.map(worker, jobs)
    summary = {}
    for r in res:
        a = summary.setdefault(r["m"], {"hard_found": 0, "decided": 0, "counterexamples": 0,
                                        "undecided": 0, "max_rounds": 0, "max_cert": 0,
                                        "t_decide_median": []})
        a["hard_found"] += r["hard_found"]; a["decided"] += r["decided"]
        a["undecided"] += len(r["undecided"])
        a["counterexamples"] += len(r["counterexamples"])
        a["max_rounds"] = max(a["max_rounds"], r["max_rounds"])
        a["max_cert"] = max(a["max_cert"], r["max_cert"])
        a["t_decide_median"].append(r["t_decide_median"])
    for m in sorted(summary):
        print(m, summary[m], flush=True)
    json.dump({"summary": {str(k): v for k, v in summary.items()}, "workers": res},
              open("../results/n3_cegar_search.json", "w"), indent=1)
    print(f"done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
