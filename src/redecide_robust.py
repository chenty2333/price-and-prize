"""Re-decide the glued profiles left undecided (solver time limits) with
cegar.decide_robust, in parallel; each result is saved as soon as it is known.
Refutation certificates are re-checked with the LP-only search."""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import cegar
from verify_certificates import refutes

TL = float(sys.argv[1]) if len(sys.argv) > 1 else 600
SRC = sys.argv[2] if len(sys.argv) > 2 else "../results/n3_gluing_undecided.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else "../results/redecide_robust"


def work(args):
    k, item = args
    t0 = time.time()
    out = {"k": k, "family": item["family"], "m": item["m"]}
    try:
        r = cegar.decide_robust(3, item["m"], item["sigmas"], time_limit=TL,
                                rng=np.random.default_rng(k))
        out.update({"result": r["result"], "rounds": r["rounds"]})
        if r["result"] == "refuted":
            out["cert_size"] = r["cert_size"]
            out["lp_verified"] = refutes(r["certificate"], 3, item["m"])[0]
        else:
            out.update({"v": r["v"], "eps": r["eps"], "sigmas": item["sigmas"]})
    except Exception as e:
        out.update({"result": "undecided", "error": str(e)[:200]})
    out["sec"] = round(time.time() - t0)
    with open(f"{OUT}/{item.get('k0', k)}.json", "w") as f:
        json.dump(out, f)
    return out


if __name__ == "__main__":
    items = json.load(open(SRC))
    os.makedirs(OUT, exist_ok=True)
    with Pool(min(16, len(items))) as p:
        for o in p.imap_unordered(work, list(enumerate(items))):
            print({k: o.get(k) for k in ("k", "family", "m", "result", "rounds", "cert_size", "lp_verified", "sec")}, flush=True)
