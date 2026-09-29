"""Check the O(log m)-query algorithm (sign bisection + one repair query) on our
own reversal path (verify_monotone_ours.reversal_path) with random monotone v:
the returned allocation must be market EF1 (checked by brute force over all
deletions) and the number of value queries must be <= 3 + 2*ceil(log2 ceil(m/2))."""
import math
from collections import Counter
import numpy as np
import fd
from verify_monotone_ours import reversal_path, monotone_family, ef1_fail

rng = np.random.default_rng(99)
stats = Counter()
for t in range(4000):
    m = int(rng.integers(1, 11)) if t % 4 == 0 else int(rng.integers(1, 40))
    s1 = [int(x) for x in rng.permutation(m)]; s2 = [int(x) for x in rng.permutation(m)]
    path = reversal_path(m, s1, s2)
    kinds = ["table", "xos", "budget", "coverage", "concave", "supermodular"] if m <= 10 else \
            ["xos", "budget", "coverage", "concave", "supermodular"]
    v0 = monotone_family(m, rng, kinds[t % len(kinds)])
    q = [0]; cache = {}
    def v(S):  # each distinct set is one oracle query; repeated sets reuse the answer
        S = frozenset(S)
        if S not in cache:
            q[0] += 1; cache[S] = v0(S)
        return cache[S]
    bundles = lambda a: (frozenset(np.nonzero(a == 0)[0].tolist()), frozenset(np.nonzero(a == 1)[0].tolist()))
    def sign(k):
        B1, B2 = bundles(path[k]); d = v(B1) - v(B2)
        return 0 if d == 0 else (1 if d > 0 else -1)
    T = len(path) - 1
    s0 = sign(0)
    if s0 == 0 or T == 0:
        ans = path[0]
    else:
        lo, hi = 0, T  # sign(T) = -s0 by reversal (not queried)
        found = None
        while hi - lo > 1:
            mid = (lo + hi) // 2; sm = sign(mid)
            if sm == 0: found = path[mid]; break
            if sm == s0: lo = mid
            else: hi = mid
        if found is None:
            a, b = path[lo], path[hi]
            # orient: first bundle higher at earlier state (sign at lo is s0)
            hiB = 0 if s0 > 0 else 1
            X_plus_g = frozenset(np.nonzero(a == hiB)[0].tolist())
            Y_plus_h = frozenset(np.nonzero(a != hiB)[0].tolist())
            g = next(iter(X_plus_g - frozenset(np.nonzero(b == hiB)[0].tolist())), None)
            assert g is not None
            X = X_plus_g - {g}
            found = a if v(X) <= v(Y_plus_h) else b
        ans = found
    ok = not ef1_fail(ans, v0 if False else (lambda S: v0(frozenset(S))), 0, 1) and \
         not ef1_fail(ans, (lambda S: v0(frozenset(S))), 1, 0)
    sd = bool(fd.subjective_sd_ef1(ans[None, :], [s1, s2])[0])
    bound = 3 + 2 * math.ceil(math.log2(max(1, math.ceil(m / 2))))
    stats["instances"] += 1; stats["not_ef1"] += (not ok); stats["not_sd"] += (not sd)
    stats["over_bound"] += (q[0] > bound); stats["max_queries"] = max(stats["max_queries"], q[0])
print(dict(stats))
