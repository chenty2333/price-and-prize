"""Round-five constructions and exact search; Python standard library only.

Public allocations use owner[g] in {0,1,2}.  Rankings are best-first permutations.
Shift budgets are measured on the padded instance with a multiple of three goods.
A failed restricted search is NOT a counterexample to unrestricted Conjecture B.
The direct-scan combinatorial bound is O(M * 6**(M//3)). This implementation
uses M-bit Python integers: a conservative multiword bound is
O(M * ceil(M/w) * 6**(M//3)) word operations. Certified M <= 15 fits one word.
"""
from __future__ import annotations
from itertools import combinations
from typing import MutableMapping, Sequence
from independent_checker import check

def validate(profile):
    """Linear-time validation of indexed strict rankings and market-ranked agent 0."""
    if len(profile) != 3: raise ValueError("Exactly three rankings are required")
    m = len(profile[0])
    for sigma in profile:
        if len(sigma) != m: raise ValueError("Ranking lengths differ")
        seen = bytearray(m)
        for g in sigma:
            if type(g) is not int or not 0 <= g < m or seen[g]:
                raise ValueError("Rankings must be permutations of 0..m-1")
            seen[g] = 1
    if any(g != t for t,g in enumerate(profile[0])):
        raise ValueError("Agent 0 must follow the market")
    return m



def counterexample13() -> tuple[list[list[int]], list[int]]:
    """Minimal counterexample to own-class balance, with a fair unrestricted witness."""
    profile = [list(range(13)),
        [0,1,3,12,2,6,9,10,4,7,5,8,11],
        [3,2,0,1,6,5,12,4,8,11,7,9,10]]
    owner = [0,1,2,0,1,2,1,0,2,0,1,2,2]
    return profile, owner


def shifted_modules(copies: int) -> tuple[list[list[int]], list[int]]:
    """15h-good family with minimum forward-shift budget exactly h.

    The returned allocation attains that bound.  The lower bound follows by
    common-module factorization and the proved thirteen-good obstruction.
    """
    if type(copies) is not int or copies < 1:
        raise ValueError('copies must be a positive integer')
    base, owner = counterexample13()
    base = [s + [13, 14] for s in base]
    owner += [0, 1]
    m = 15*copies
    result = [list(range(m))]
    result += [[g+15*j for j in range(copies) for g in base[i]] for i in (1, 2)]
    return result, owner*copies


def triple_counts(profile: Sequence[Sequence[int]], owner: Sequence[int]) -> list[int]:
    """Agent 1's counts in consecutive triples of its ranking (last may be short)."""
    return [sum(owner[g] == 1 for g in profile[1][b:b+3])
            for b in range(0, len(owner), 3)]


def shift_budget(profile: Sequence[Sequence[int]], owner: Sequence[int]) -> int:
    """Surplus units on full triples. This is not replacement distance from a bundle."""
    if len(owner) % 3:
        raise ValueError('Shift budget is defined here only after padding to full triples')
    return sum(max(c-1, 0) for c in triple_counts(profile, owner))


def forward_transfers(counts: Sequence[int]) -> list[tuple[int, int]]:
    """Decompose a nonnegative height profile into earlier-surplus/later-deficit units.

    This describes the triple-count vector, not a sequence of physical fair swaps.
    A pair (a,b), a<b, transfers one unit of the baseline count from triple b to a.
    """
    if any(type(c) is not int or not 0 <= c <= 3 for c in counts):
        raise ValueError('Triple counts must be integers from zero to three')
    if sum(counts) != len(counts):
        raise ValueError('The total must be one good per triple')
    available: list[int] = []
    moves: list[tuple[int, int]] = []
    for j, c in enumerate(counts):
        if c >= 1:
            available.extend([j]*(c-1))
        else:
            if not available:
                raise ValueError('Negative block-end height')
            moves.append((available.pop(), j))
    if available:
        raise AssertionError('Unmatched surplus despite equal total counts')
    return moves


def _padded(profile):
    m = validate(profile)
    M = 3*((m+2)//3)
    return m, [list(s)+list(range(m, M)) for s in profile]


def _small_modules(pp):
    """Find a full segmentation into common modules of at most 15 goods."""
    M = len(pp[0])
    ranks = [[0]*M for _ in range(3)]
    for i in (1, 2):
        for j, g in enumerate(pp[i]):
            ranks[i][g] = j
    pred = {0: None}
    for end in range(3, M+1, 3):
        for width in (3, 6, 9, 12, 15):
            start = end-width
            if start not in pred:
                continue
            if all(max(ranks[i][g] for g in range(start,end)) -
                   min(ranks[i][g] for g in range(start,end)) == width-1
                   for i in (1, 2)):
                pred[end] = start
                break
    if M not in pred:
        return None
    modules = []
    end = M
    while end:
        start = pred[end]
        local = [list(range(end-start))]
        for i in (1, 2):
            first = min(ranks[i][g] for g in range(start,end))
            local.append([g-start for g in pp[i][first:first+end-start]])
        modules.append((start, end, local))
        end = start
    return list(reversed(modules))


def _search_component(pp, budget: int, stats: MutableMapping) -> list[int] | None:
    """Complete exhaustive search on full triples, in increasing shift budget."""
    M = len(pp[0])
    if M == 0:
        return []
    q = M//3
    rank1 = [0]*M
    rank2 = [0]*M
    for t, g in enumerate(pp[1]): rank1[g] = t
    for t, g in enumerate(pp[2]): rank2[g] = t
    pref = []
    for sigma in pp[1:]:
        chain = []; mask = 0
        for g in sigma:
            mask |= 1 << g; chain.append(mask)
        pref.append(chain)
    choices = [sorted(range(3*b, 3*b+3), key=lambda g: -rank2[g]) for b in range(q)]

    def own_matchings(left, used, S):
        if not left:
            yield S; return
        options = [(b, [g for g in choices[b] if not (used >> (rank1[g]//3)) & 1]) for b in left]
        b, opts = min(options, key=lambda pair: (len(pair[1]), pair[0]))
        if not opts:
            return
        nxt = [v for v in left if v != b]
        for g in opts:
            yield from own_matchings(nxt, used | (1 << (rank1[g]//3)), S | (1 << g))

    def shifted_transversals(target):
        counts = [0]*q
        def visit(b, S, surplus):
            if surplus > target: return
            if b == q:
                if surplus != target: return
                height = 0
                for c in counts:
                    height += c-1
                    if height < 0: return
                yield S
                return
            for g in choices[b]:
                t = rank1[g]//3
                extra = int(counts[t] >= 1)
                counts[t] += 1
                yield from visit(b+1, S | (1 << g), surplus+extra)
                counts[t] -= 1
        yield from visit(0, 0, 0)

    def complete(S):
        stats['S_candidates'] = stats.get('S_candidates', 0)+1
        pairs = [[g for g in range(3*b,3*b+3) if not (S>>g)&1] for b in range(q)]
        base = 0; toggles = []
        for a, b in pairs:
            base |= 1 << min((a,b), key=lambda g: rank2[g])
            toggles.append((1<<a) | (1<<b))
        bounds1 = [(P, t-2*(P&S).bit_count()-1, (P&S).bit_count()+1)
                   for t, P in enumerate(pref[0], 1)]
        bounds2 = [(P, max((P&S).bit_count()-1, (t-(P&S).bit_count())//2))
                   for t, P in enumerate(pref[1], 1)]
        for h in range(q+1):
            for changed in combinations(range(q), h):
                X = base
                for b in changed: X ^= toggles[b]
                stats['orientations'] = stats.get('orientations', 0)+1
                if any(not lo <= (P&X).bit_count() <= hi for P,lo,hi in bounds1):
                    continue
                if any((P&X).bit_count() < lo for P,lo in bounds2):
                    continue
                owner = [1 if (S>>g)&1 else 2 if (X>>g)&1 else 0 for g in range(M)]
                error = check(pp, owner)
                if error is not None: raise AssertionError(error)
                return owner
        return None

    for target in range(min(budget, q)+1):
        iterator = own_matchings(list(range(q)),0,0) if target == 0 else shifted_transversals(target)
        for S in iterator:
            owner = complete(S)
            if owner is not None:
                return owner
    return None


def search_shift_balanced(profile, budget: int = 1, stats=None, use_modules: bool = True):
    """Exact search with at most `budget` forward surplus units on the padded instance.

    Returns None only after exhausting that restricted family. For recognized common
    modules, local minimum budgets add exactly; this is an exact reduction, not a
    heuristic. Every individual partial assignment need not be fair.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError('budget must be a nonnegative integer')
    m, pp = _padded(profile)
    M = len(pp[0])
    if stats is None: stats = {}
    modules = _small_modules(pp) if use_modules and M > 15 else None
    if modules is not None and len(modules) > 1:
        owner = [-1]*M; total = 0; details = []
        for start,end,local in modules:
            answer = _search_component(local, budget-total, stats)
            if answer is None:
                stats['infeasibility_reason'] = 'exact sum of local minimum budgets exceeds budget'
                stats['solved_modules'] = details
                return None
            used = shift_budget(local, answer)
            total += used; owner[start:end] = answer
            details.append({'start':start,'end':end,'minimum_budget':used})
        stats['solved_modules'] = details
    else:
        owner = _search_component(pp, budget, stats)
        if owner is None: return None
    error = check(pp, owner)
    if error is not None: raise AssertionError(error)
    used = shift_budget(pp, owner)
    if used > budget: raise AssertionError(('budget exceeded', used, budget))
    forward_transfers(triple_counts(pp, owner))
    stats['padded_owner'] = owner.copy()
    stats['padded_m'] = M
    stats['shift_budget'] = used
    return owner[:m]


def solve_up_to_15(profile, stats=None):
    """Certified finite-domain constructor once coverage15/verification.json is PASS.

    The intended theorem is unrestricted B with at most one forward surplus unit,
    not own-class balance. The latter has a thirteen-good counterexample.
    """
    m = validate(profile)
    if m > 15: raise ValueError('This finite-domain wrapper requires m <= 15')
    answer = search_shift_balanced(profile, 0 if m <= 12 else 1, stats)
    if answer is None: raise AssertionError('Contradicts the corresponding completed finite verification')
    return answer


def solve_common_modules(profile, stats=None):
    """Compose <=15-good common modules; return None if none is recognized.

    When the finite coverage certificate passes, construction is guaranteed on
    recognized inputs. Runtime O(m) with the fixed module-size bound.
    """
    m, pp = _padded(profile)
    if stats is None: stats = {}
    modules = _small_modules(pp)
    if modules is None: return None
    owner = [-1]*len(pp[0]); total = 0
    for start,end,local in modules:
        answer = solve_up_to_15(local)
        total += shift_budget(local, answer)
        owner[start:end] = answer
    if check(pp,owner) is not None: raise AssertionError(check(pp,owner))
    stats.update(padded_owner=owner.copy(), padded_m=len(owner), shift_budget=total,
                 modules=[(a,b) for a,b,_ in modules])
    return owner[:m]
