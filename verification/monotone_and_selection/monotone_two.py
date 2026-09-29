"""Two-agent ordinal SD-EF1 paths and identical-market EF1.
No optimization package is used. Items and agents are 0-indexed.
For mixed items, negatives is a COMMON set of subjective/market chores;
rankings order ABSOLUTE subjective item magnitudes, largest first.
"""
from typing import Callable, Sequence


def sd_path(rankings: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    """A path from a doubly balanced coloring to its reversal.
    At most ceil(m/2) swaps/moves after removing a common bottom dummy.
    Every output is SD-EF1 for the two rankings.
    """
    if len(rankings) != 2:
        raise ValueError('Exactly two rankings are required')
    m = len(rankings[0])
    if any(sorted(r) != list(range(m)) for r in rankings):
        raise ValueError('Rankings must both be permutations of range(m)')
    p = [list(r) for r in rankings]
    if m % 2:
        for r in p: r.append(m)
    N = len(p[0])
    mates = []
    for r in p:
        mate = [-1]*N
        for k in range(0,N,2):
            x,y = r[k:k+2]
            mate[x],mate[y] = y,x
        mates.append(mate)
    red,blue = mates
    owner = [-1]*N
    components = []
    for seed in range(N):
        if owner[seed] != -1: continue
        owner[seed] = 0
        stack,component = [seed],[]
        while stack:
            x = stack.pop(); component.append(x)
            for y in (red[x],blue[x]):
                if owner[y] == -1:
                    owner[y] = 1-owner[x]; stack.append(y)
                elif owner[y] == owner[x]:
                    raise AssertionError('Union of pairings was not bipartite')
        components.append(component)
    rank2 = {g:k for k,g in enumerate(p[1])}
    path = [tuple(owner[:m])]
    for component in components:
        first = min(component, key=rank2.__getitem__)
        start = first if owner[first] == 0 else blue[first]
        x = start
        while True:
            y = red[x]
            assert owner[x] == 0 and owner[y] == 1
            owner[x],owner[y] = 1,0
            path.append(tuple(owner[:m]))
            x = blue[y]
            if x == start: break
    assert path[-1] == tuple(1-i for i in path[0])
    return path


def mixed_path(rankings: Sequence[Sequence[int]], negatives: set[int]) -> list[tuple[int,...]]:
    """Toggle the ownership of common chores in the goods-coordinate path."""
    m = len(rankings[0])
    if not negatives <= set(range(m)):
        raise ValueError('Unknown chore')
    return [tuple(i ^ (g in negatives) for g,i in enumerate(a))
            for a in sd_path(rankings)]


def ef1(a: Sequence[int], value: Callable[[frozenset[int]], int],
        negatives: set[int] | None = None) -> bool:
    """Independent definition-based shared-market check.
    Positive items may be removed from the envied bundle;
    negative items may be removed from the envier's bundle.
    """
    C = set() if negatives is None else set(negatives)
    bundles = [frozenset(g for g,i in enumerate(a) if i == k) for k in range(2)]
    vals = [value(S) for S in bundles]
    for i in range(2):
        j = 1-i
        if vals[i] >= vals[j]: continue
        if any(vals[i] >= value(bundles[j]-{g}) for g in bundles[j]-C): continue
        if any(value(bundles[i]-{g}) >= vals[j] for g in bundles[i]&C): continue
        return False
    return True


def allocate(rankings: Sequence[Sequence[int]],
             value: Callable[[frozenset[int]], int],
             negatives: set[int] | None = None) -> tuple[int,...]:
    """Polynomial query algorithm; requires market monotonicity in common signs.
    For all chores, pass value=-cost and all items in negatives.
    """
    C = set() if negatives is None else set(negatives)
    for a in mixed_path(rankings,C):
        if ef1(a,value,C): return a
    raise ValueError('No path witness: check common signs and market monotonicity')
