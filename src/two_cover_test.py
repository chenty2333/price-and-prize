"""Test the candidate lemma that every hard n = 3 profile has a TWO-allocation
cover (two subjectively SD-EF1 allocations such that no market valuation makes
both violate market EF1). Pool of allocations per profile: the CEGAR
certificate plus oracle optima at many valuations; every pair in the pool is
checked with the LP-only search from verify_certificates.refutes."""
import glob, json, sys, time
from multiprocessing import Pool
import numpy as np
import cegar
from verify_certificates import refutes


def corpus():
    P = []
    for mk, rec in json.load(open("../results/n3_search.json")).items():
        P += [(int(mk), ex["sigmas"], "n3_search") for ex in rec["hard_examples"]]
    for f in glob.glob("../results/n3_cegar_search_w*.json"):
        r = json.load(open(f))
        if r.get("hardest"): P.append((r["m"], r["hardest"]["sigmas"], "cegar_hardest"))
    for r in json.load(open("../results/n3_redecided.json")):
        P.append((r["m"], r["sigmas"], "redecided"))
    for u in json.load(open("../results/n3_gluing_undecided.json")):
        P.append((u["m"], u["sigmas"], "glued_" + u["family"]))
    return P


def work(args):
    k, (m, s, src) = args
    rng = np.random.default_rng(k)
    t0 = time.time()
    try:
        r = cegar.decide_robust(3, m, s, time_limit=600, rng=rng) if m > 21 else cegar.decide(3, m, s, rng=rng)
        pool = [np.array(a, dtype=np.int8) for a in r["certificate"]]
        orc = cegar.Oracle(3, m, s)
        vs = [np.sort(rng.random(m))[::-1] for _ in range(15)] + \
             [np.r_[np.ones(t), np.full(m - t, 0.01)] for t in range(2, m, 2)] + \
             [rng.random() ** np.arange(m) for _ in range(5)]
        for v in vs:
            v = v / v[0]
            try:
                o = cegar.oracle_robust(orc, v, eps=np.inf, time_limit=120)
            except cegar.SolverError:
                continue
            if o["alloc"] is not None and not any((o["alloc"] == b).all() for b in pool):
                pool.append(o["alloc"])
        found = None
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                if refutes([pool[i].tolist(), pool[j].tolist()], 3, m)[0]:
                    found = (i, j); break
            if found: break
        return {"k": k, "m": m, "src": src, "cert": r["cert_size"], "pool": len(pool),
                "two_cover": found is not None, "sec": round(time.time() - t0)}
    except Exception as e:
        return {"k": k, "m": m, "src": src, "error": str(e)[:120]}


if __name__ == "__main__":
    P = corpus()
    out = []
    with Pool(14) as p:
        for r in p.imap_unordered(work, list(enumerate(P))):
            out.append(r); print(r, flush=True)
            json.dump(out, open("../results/two_cover_test.json", "w"), indent=1)
    ok = [r for r in out if "two_cover" in r]
    print(f"profiles {len(out)}, tested {len(ok)}, two-cover found {sum(r['two_cover'] for r in ok)}")
