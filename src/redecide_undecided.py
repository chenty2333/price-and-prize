"""Re-decide profiles that n3_cegar_search.py left undecided because HiGHS
reported an internal error. Different rng seeds change the initial cut set and
hence the MILPs HiGHS sees; a profile counts as decided once any seed succeeds."""
import glob
import json

import numpy as np

import cegar

out = []
for f in sorted(glob.glob("../results/n3_cegar_search_w*.json")):
    rec = json.load(open(f))
    for u in rec.get("undecided", []):
        s, m = u["sigmas"], rec["m"]
        res = None
        for seed in range(1, 21):
            try:
                res = cegar.decide(3, m, s, rng=np.random.default_rng(seed))
                res["seed"] = seed
                break
            except cegar.SolverError:
                continue
        row = {"file": f, "m": m, "sigmas": s,
               "result": None if res is None else res["result"],
               "seed": None if res is None else res["seed"],
               "rounds": None if res is None else res["rounds"]}
        if res is not None and res["result"] == "counterexample":
            row["v"] = res["v"]
        out.append(row)
        print({k: row[k] for k in ("m", "result", "seed", "rounds")}, flush=True)
json.dump(out, open("../results/n3_redecided.json", "w"), indent=1)
