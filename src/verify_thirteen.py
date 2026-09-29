"""Independent check (fd.py) of the thirteen-good counterexample to
own-bundle balance and its padded fifteen-good version."""
import json
import numpy as np
import fd

s1 = [0,1,3,12,2,6,9,10,4,7,5,8,11]
s2 = [3,2,0,1,6,5,12,4,8,11,7,9,10]

def analyse(m, s1, s2):
    sig = [list(range(m)), s1, s2]
    Mal = fd.market_sd_ef1_allocations(3, m)
    F = Mal[fd.subjective_sd_ef1(Mal, sig)]
    full = [s1[k:k+3] for k in range(0, m - m % 3, 3)]
    short = s1[m - m % 3:] if m % 3 else []
    cnt = np.stack([(F[:, T] == 1).sum(axis=1) for T in full], axis=1)
    own = (cnt == 1).all(axis=1)
    if short:
        own &= (F[:, short] == 1).sum(axis=1) <= 1
    budget = np.maximum(cnt - 1, 0).sum(axis=1)
    return {"m": m, "market_balanced": int(len(Mal)), "fair": int(len(F)),
            "fair_and_own_balanced": int(own.sum()),
            "budget_hist": {int(k): int(v) for k, v in zip(*np.unique(budget, return_counts=True))}}

out = {"m13": analyse(13, s1, s2)}
w = np.zeros(13, dtype=np.int8)
for g in [1,4,6,10]: w[g] = 1
for g in [2,5,8,11,12]: w[g] = 2
out["witness_fair"] = bool(fd.subjective_sd_ef1(w[None, :], [list(range(13)), s1, s2])[0]) and \
    all(len(set(w[k:k+3])) == len(w[k:k+3]) for k in range(0, 13, 3))
out["m15_padded"] = analyse(15, s1 + [13, 14], s2 + [13, 14])
print(json.dumps(out, indent=1))
json.dump(out, open("../results/verify_thirteen.json", "w"), indent=1)
