"""Round-two constructive results. Standard library only.

solve_up_to_12: existence computer-assisted for n=3, m<=12.
search_own_balanced: exact search of a stronger family for arbitrary m;
    existence for m>12 is NOT proved and the running time is exponential.
solve_modules: proved gluing reduction to the certified <=12 theorem.
"""
from __future__ import annotations
from itertools import combinations, permutations, product
from typing import Sequence


def validate(profile):
    if len(profile) != 3:
        raise ValueError('Requires three agents')
    m = len(profile[0])
    if any(len(s) != m or set(s) != set(range(m)) for s in profile):
        raise ValueError('Rankings must be permutations of 0..m-1')
    if list(profile[0]) != list(range(m)):
        raise ValueError('Agent 0 must follow the market')
    return m


def check_allocation(profile, owner):
    """Independent direct checker, with explicit prefix witnesses."""
    m = validate(profile)
    if len(owner) != m or any(type(x) is not int or x not in (0, 1, 2) for x in owner):
        return {'kind': 'invalid_owner'}
    for b in range(0, m, 3):
        if len(set(owner[b:b+3])) != len(owner[b:b+3]):
            return {'kind': 'market', 'block': b//3}
    for i, ranking in enumerate(profile):
        counts = [0, 0, 0]
        for t, g in enumerate(ranking, 1):
            counts[owner[g]] += 1
            for j in range(3):
                if counts[j] > counts[i]+1:
                    return {'kind': 'sd_ef1', 'agent': i, 'towards': j,
                            'prefix_length': t, 'prefix': list(ranking[:t]),
                            'counts': counts.copy()}
    return None


def search_own_balanced(profile, stats=None):
    """Find a fair allocation with agent 1 balanced in its own ranking blocks.

    Complete exact search of this restricted family on the padded instance.
    Returns None if that family is empty. Such a return does NOT disprove B.
    Worst-case exponential, not a polynomial-time algorithm for general m.
    """
    m = validate(profile)
    M = 3*((m+2)//3)
    if not M:
        return []
    q = M//3
    pp = [list(s)+list(range(m, M)) for s in profile]
    rank1 = [0]*M
    rank2 = [0]*M
    for t, g in enumerate(pp[1]):
        rank1[g] = t
    for t, g in enumerate(pp[2]):
        rank2[g] = t
    if stats is None:
        stats = {}
    stats.setdefault('matchings', 0)
    stats.setdefault('orientations', 0)
    prefix1 = []
    prefix2 = []
    for sigma, dest in ((pp[1], prefix1), (pp[2], prefix2)):
        mask = 0
        for g in sigma:
            mask |= 1 << g
            dest.append(mask)

    # MRV recursion enumerates every perfect matching between market triples
    # and sigma_1 triples. The matching's goods are exactly agent 1's bundle.
    choices = [sorted(range(3*b, 3*b+3), key=lambda g: -rank2[g]) for b in range(q)]
    def matchings(left, used, S):
        if not left:
            yield S
            return
        available = [(b, [g for g in choices[b] if not (used >> (rank1[g]//3)) & 1]) for b in left]
        b, opts = min(available, key=lambda pair: (len(pair[1]), pair[0]))
        if not opts:
            return
        nxt = [v for v in left if v != b]
        for g in opts:
            yield from matchings(nxt, used | (1 << (rank1[g]//3)), S | (1 << g))

    for S in matchings(list(range(q)), 0, 0):
        stats['matchings'] += 1
        pairs = [[g for g in range(3*b, 3*b+3) if not (S>>g)&1] for b in range(q)]
        base = 0
        toggles = []
        for a, b in pairs:
            base |= 1 << min((a, b), key=lambda g: rank2[g])
            toggles.append((1 << a) | (1 << b))
        bounds1 = [(P, t-2*(P&S).bit_count()-1, (P&S).bit_count()+1)
                   for t, P in enumerate(prefix1, 1)]
        bounds2 = [(P, max((P&S).bit_count()-1, (t-(P&S).bit_count())//2))
                   for t, P in enumerate(prefix2, 1)]
        # Enumerate every binary orientation, trying few deviations from
        # agent 2's locally favorite residual goods first.
        for h in range(q+1):
            for changed in combinations(range(q), h):
                X = base
                for b in changed:
                    X ^= toggles[b]
                stats['orientations'] += 1
                if any(not lo <= (P&X).bit_count() <= hi for P, lo, hi in bounds1):
                    continue
                if any((P&X).bit_count() < lo for P, lo in bounds2):
                    continue
                owner = [1 if (S>>g)&1 else 2 if (X>>g)&1 else 0 for g in range(m)]
                violation = check_allocation(profile, owner)
                if violation is not None:
                    raise AssertionError(violation)
                return owner
    return None


def solve_up_to_12(profile, stats=None):
    m = validate(profile)
    if m > 12:
        raise ValueError('Certified existence only for m<=12')
    answer = search_own_balanced(profile, stats)
    if answer is None:
        raise AssertionError('Contradicts the finite coverage verification')
    return answer


def solve_modules(profile):
    """Linear-time constructive reduction for common modules of <=12 goods.

    Worst-ranked virtual goods pad to a multiple of three. A module is an
    interval of complete market blocks, of size 3,6,9,or12, contiguous in
    each real ranking. Modules can occur in different orders for each agent.
    Return None if no such segmentation exists; otherwise return allocation.
    """
    m = validate(profile)
    M = 3*((m+2)//3)
    pp = [list(s)+list(range(m, M)) for s in profile]
    rank = [[0]*M for _ in range(3)]
    for i in (1, 2):
        for t, g in enumerate(pp[i]):
            rank[i][g] = t
    pred = {0: None}
    for end in range(3, M+1, 3):
        for width in (3, 6, 9, 12):
            start = end-width
            if start not in pred:
                continue
            if all(max(rank[i][g] for g in range(start,end))-min(rank[i][g] for g in range(start,end)) == width-1 for i in (1,2)):
                pred[end] = start
                break
    if M not in pred:
        return None
    segments=[]
    end=M
    while end:
        start=pred[end]
        segments.append((start,end));end=start
    owner=[-1]*M
    for start,end in reversed(segments):
        local=[list(range(end-start))]
        for i in (1,2):
            # Read just this contiguous module, not all M positions.
            first=min(rank[i][g] for g in range(start,end))
            local.append([g-start for g in pp[i][first:first+end-start]])
        ans=solve_up_to_12(local)
        owner[start:end]=ans
    owner=owner[:m]
    error=check_allocation(profile,owner)
    if error is not None:
        raise AssertionError(error)
    return owner


def all_market_allocations(m):
    choices=[list(permutations(range(3),min(3,m-b))) for b in range(0,m,3)]
    for picks in product(*choices):
        yield [a for block in picks for a in block]
