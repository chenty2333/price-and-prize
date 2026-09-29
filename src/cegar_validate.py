"""Validate cegar.decide against enumeration.

1. OQ1 (EF1 market) on the hard n = 3 profiles found by n3_search.py
   (m = 10..12): decide must say 'refuted', agreeing with fd.find_breaking_v.
2. Positive control: SD-EF1 subjective + EFX market, which Barman et al.
   show can be impossible; values are floored at 0.2 * v[0] so that EFX does
   not degenerate to EF and both outcomes occur. On random small profiles (n = 2, 3), every
   'counterexample' is re-verified by enumerating S; every 'refuted' is
   re-verified by solving the master over the full enumerated S.
"""
import json
import time

import numpy as np

import cegar
import fd


def efx_violated_all(alloc_S, v, n):
    """True iff every allocation violates market EFX under v (enumeration)."""
    v = np.asarray(v)
    for a in alloc_S:
        bad = False
        for i in range(n):
            for j in range(n):
                if i == j or not (a == j).any():
                    continue
                idx = np.nonzero(a == j)[0]
                if v[a == j].sum() - v[idx[-1]] - v[a == i].sum() > 1e-9:
                    bad = True
        if not bad:
            return False
    return True


def main():
    out = {"oq1_hard": [], "efx_control": []}
    hard = json.load(open("../results/n3_search.json"))
    for mk, rec in hard.items():
        m = int(mk)
        for ex in rec["hard_examples"][:4]:
            t0 = time.time()
            r = cegar.decide(3, m, ex["sigmas"])
            out["oq1_hard"].append({"m": m, "result": r["result"], "rounds": r["rounds"],
                                    "cert": r["cert_size"], "sec": round(time.time() - t0, 1)})
            assert r["result"] == "refuted", r
            print("OQ1 hard", out["oq1_hard"][-1], flush=True)
    rng = np.random.default_rng(3)
    counts = {"counterexample": 0, "refuted": 0}
    for n, m in [(2, 5), (2, 6), (2, 7), (3, 5), (3, 6), (3, 7)]:
        alloc = fd.all_allocations(n, m)
        for _ in range(12):
            sig = [list(rng.permutation(m)) for _ in range(n)]
            r = cegar.decide(n, m, sig, market="EFX", vmin=0.2)
            S = alloc[fd.subjective_sd_ef1(alloc, sig)]
            if r["result"] == "counterexample":
                assert efx_violated_all(S, r["v"], n), "false counterexample"
            else:
                eps, _ = cegar.master(list(S), n, m, market="EFX", vmin=0.2)
                assert eps <= 1e-6, f"missed counterexample (full-S master eps={eps})"
            counts[r["result"]] += 1
            out["efx_control"].append({"n": n, "m": m, "result": r["result"], "rounds": r["rounds"]})
        print(f"EFX control n={n} m={m}: {counts}", flush=True)
    # 3. mixed-outcome control: EFX with high value floors (rare counterexamples)
    rng = np.random.default_rng(9)
    mixed = {}
    for vmin in (0.8, 0.9, 0.95):
        for n, m in [(2, 6), (2, 7), (3, 6), (3, 7)]:
            alloc = fd.all_allocations(n, m)
            for _ in range(10):
                sig = [list(rng.permutation(m)) for _ in range(n)]
                r = cegar.decide(n, m, sig, market="EFX", vmin=vmin)
                S = alloc[fd.subjective_sd_ef1(alloc, sig)]
                if r["result"] == "counterexample":
                    assert efx_violated_all(S, r["v"], n), "false counterexample"
                else:
                    eps, _ = cegar.master(list(S), n, m, market="EFX", vmin=vmin)
                    assert eps <= 1e-6, f"missed counterexample (full-S master eps={eps})"
                key = f"vmin={vmin}:{r['result']}"
                mixed[key] = mixed.get(key, 0) + 1
    out["efx_mixed_control"] = mixed
    print("EFX mixed control:", mixed, flush=True)
    json.dump(out, open("../results/cegar_validate.json", "w"), indent=1)
    print("all validations passed", counts)


if __name__ == "__main__":
    main()
