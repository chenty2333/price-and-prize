"""Independent check (fd.py) of the claimed 12-good counterexample to general
disjoint-defect existence, and of its two-allocation cover (LP-only search)."""
import itertools, json
from collections import Counter
import numpy as np
import fd
from verify_certificates import refutes

P = [[0,7,2,5,1,11,8,3,6,9,4,10],
     [0,11,2,7,1,5,10,9,3,8,6,4],
     [5,0,2,11,7,1,4,3,9,8,6,10]]
m = 12
alloc = fd.all_allocations(3, m)
S = alloc[fd.subjective_sd_ef1(alloc, P)]
near, D = [], []
for a in S:
    c = [0, 0, 0]; d = set(); ok = True
    for t, i in enumerate(a, 1):
        c[i] += 1
        if t % 3 and max(c) - min(c) > 1: ok = False; break
        if t % 3 == 0 and len(set(c)) > 1: d.add(t)
    if ok: near.append(a); D.append(frozenset(d))
hist = Counter(tuple(sorted(d)) for d in D)
pairs = sum(D[i].isdisjoint(D[j]) for i in range(len(D)) for j in range(i + 1, len(D)))
A = [1,0,2,1,2,2,0,0,2,1,0,1]; B = [1,1,0,2,0,2,1,2,0,2,1,0]
out = {"sd_ef1": int(len(S)), "near_block": len(near),
       "market_balanced_in_S": int(fd.subjective_sd_ef1(fd.market_sd_ef1_allocations(3, m), P).sum()),
       "defect_hist": {str(k): v for k, v in sorted(hist.items())}, "disjoint_pairs": pairs,
       "cover_pair_sd_ef1": bool(fd.subjective_sd_ef1(np.array([A, B], dtype=np.int8), P).all()),
       "cover_pair_LP": refutes([A, B], 3, m)[0]}
print(json.dumps(out, indent=1))
json.dump(out, open("../results/verify_defect_counterexample.json", "w"), indent=1)
