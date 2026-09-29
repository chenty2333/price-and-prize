import sys; sys.argv=['x','7']
exec(open(__import__('os').path.join(__import__('os').path.dirname(__file__),'oq1_probe.py')).read().split("cands = {}")[0])
exec("def _():\n    pass")
src=open(__import__('os').path.join(__import__('os').path.dirname(__file__),'oq1_probe.py')).read()
exec(src[src.index("def solve"):src.index("best = 0.0")])
# partitions NOT market-SD-EF1: each alone should be breakable (eps>0)
bad=[int(A) for A in range(1,FULL) if not mk[A]][:200]
e=[-solve([A]).fun for A in bad]
print("single non-market-SD-EF1 partitions: min eps", min(e), "count", len(e))
good=[int(A) for A in range(1,FULL) if mk[A]]
e2=[-solve([A]).fun for A in good]
print("single market-SD-EF1 partitions: max eps", max(e2), "(should be 0)")
# known-feasible multi-partition case: {A_bad1, A_bad2} sharing a breaking v
print("pair test eps", -solve(bad[:2]).fun)
