"""Exact fixed-matching results and the SAT-family repair. Standard library only.

No all-m existence assertion for unrestricted Conjecture B is made here.
solve_reset_matching is exact ONLY when S is last in every sigma_1 triple.
Its complexity is O(m * 2**c), where c counts residual alternating cycles.
The general fixed-S brute-force iterator costs O(m * 2**(m/3)).
"""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
from itertools import product
from typing import Iterable, Iterator, Mapping, Sequence


def validate_matching(profile: Sequence[Sequence[int]], S: Iterable[int],
                      reset: bool = False) -> tuple[int, set[int]]:
    if len(profile) != 3:
        raise ValueError('Exactly three rankings are required')
    m = len(profile[0])
    if m % 3:
        raise ValueError('Pad first: this interface requires m divisible by three')
    if list(profile[0]) != list(range(m)):
        raise ValueError('Agent 0 must follow the market')
    for sigma in profile:
        seen = [False]*m
        if len(sigma) != m:
            raise ValueError('Ranking lengths differ')
        for g in sigma:
            if type(g) is not int or g < 0 or g >= m or seen[g]:
                raise ValueError('Every ranking must be a permutation of 0..m-1')
            seen[g] = True
    original = list(S)
    S = set(original)
    if len(original) != len(S) or len(S) != m//3 or not S <= set(range(m)):
        raise ValueError('S must contain exactly m/3 distinct goods')
    for sigma in (profile[0], profile[1]):
        if any(sum(g in S for g in sigma[k:k+3]) != 1 for k in range(0, m, 3)):
            raise ValueError('S is not a matching between the two block systems')
    if reset and any(profile[1][k+2] not in S for k in range(0, m, 3)):
        raise ValueError('The reset theorem requires S last in every sigma_1 triple')
    return m, S


def _agent2_witness(sigma2: Sequence[int], owner: Sequence[int]):
    count = [0, 0, 0]
    for t, g in enumerate(sigma2, 1):
        count[owner[g]] += 1
        if count[0] > count[2]+1 or count[1] > count[2]+1:
            return {'prefix_length': t, 'counts': count.copy()}
    return None


def residual_cycles(profile: Sequence[Sequence[int]], S: Iterable[int]):
    """Return the two bipartition sides of every residual alternating cycle.

    Parallel matching edges are retained. A two-vertex/two-edge cycle is valid.
    """
    m, S = validate_matching(profile, S, reset=True)
    adj = {g: [] for g in range(m) if g not in S}
    for sigma in (profile[0], profile[1]):
        for k in range(0, m, 3):
            a, b = [g for g in sigma[k:k+3] if g not in S]
            adj[a].append(b)
            adj[b].append(a)
    color = {}
    components = []
    for root in range(m):
        if root in S or root in color:
            continue
        sides = [[], []]
        color[root] = 0
        stack = [root]
        while stack:
            u = stack.pop()
            sides[color[u]].append(u)
            if len(adj[u]) != 2:
                raise AssertionError('Residual degree is not two')
            for v in adj[u]:
                if v not in color:
                    color[v] = 1-color[u]
                    stack.append(v)
                elif color[v] == color[u]:
                    raise AssertionError('Union of two matchings has an odd cycle')
        if len(sides[0]) != len(sides[1]):
            raise AssertionError('Unbalanced cycle bipartition')
        components.append((tuple(sides[0]), tuple(sides[1])))
    return components


def cycle_owner(m: int, S: Iterable[int], components, choices: Sequence[int]):
    if len(choices) != len(components) or any(c not in (0, 1) for c in choices):
        raise ValueError('One binary choice is required per component')
    owner = [0]*m
    for g in S:
        owner[g] = 1
    for sides, choice in zip(components, choices):
        for g in sides[choice]:
            owner[g] = 2
    return owner


def reset_candidates(profile: Sequence[Sequence[int]], S: Iterable[int]):
    """All and only fixed-S market allocations that satisfy agent 1."""
    m, S = validate_matching(profile, S, reset=True)
    components = residual_cycles(profile, S)
    for choices in product((0, 1), repeat=len(components)):
        yield choices, cycle_owner(m, S, components, choices)


def solve_reset_matching(profile: Sequence[Sequence[int]], S: Iterable[int]):
    """Return a fair completion or None; exact, worst-case exponential."""
    for choices, owner in reset_candidates(profile, S):
        if _agent2_witness(profile[2], owner) is None:
            return owner
    return None


def reset_certificate(profile: Sequence[Sequence[int]], S: Iterable[int]):
    """Enumerate a witness prefix for EVERY rejected cycle orientation.

    This is an exact but potentially exponentially long infeasibility certificate.
    A nonempty solutions list means that the matching is feasible.
    """
    solutions, rejected = [], []
    for choices, owner in reset_candidates(profile, S):
        witness = _agent2_witness(profile[2], owner)
        if witness is None:
            solutions.append(owner)
        else:
            rejected.append({'choices': list(choices), **witness})
    return {'components': residual_cycles(profile, S),
            'solutions': solutions, 'rejected': rejected}


def signed_prefix_rows(profile: Sequence[Sequence[int]], S: Iterable[int]):
    """Exact reduced inequalities d(P).epsilon >= b(P), epsilon in {-1,+1}^c.

    epsilon=+1 means the component's first side is given to agent 2.
    b uses the *rounded* original constraints, not a weaker LP relaxation.
    """
    m, S = validate_matching(profile, S, reset=True)
    components = residual_cycles(profile, S)
    locate = {}
    for c, (a, b) in enumerate(components):
        locate.update((g, (c, 1)) for g in a)
        locate.update((g, (c, -1)) for g in b)
    d = [0]*len(components)
    rows = []
    s = r = 0
    for t, g in enumerate(profile[2], 1):
        if g in S:
            s += 1
        else:
            r += 1
            c, sign = locate[g]
            d[c] += sign
        rows.append({'t': t, 's': s, 'r': r, 'd': d.copy(),
                     'lower': max(2*s-r-2, -(r % 2))})
    return rows


def all_fixed_candidates(profile: Sequence[Sequence[int]], S: Iterable[int]):
    """All market-balanced fixed-S allocations, without using the cycle lemma."""
    m, S = validate_matching(profile, S)
    pairs = [[g for g in range(k, k+3) if g not in S] for k in range(0, m, 3)]
    for choices in product((0, 1), repeat=len(pairs)):
        owner = [0]*m
        for g in S:
            owner[g] = 1
        for pair, choice in zip(pairs, choices):
            owner[pair[choice]] = 2
        yield owner


def fractional_check(profile: Sequence[Sequence[int]], S: Iterable[int],
                     weight: Sequence[Fraction]):
    """Exact rational checking of the stated fixed-matching LP relaxation."""
    m, S = validate_matching(profile, S)
    if len(weight) != m:
        raise ValueError('One weight per good is required')
    w = [Fraction(v) for v in weight]
    if any(w[g] != 0 for g in S) or any(v < 0 or v > 1 for v in w):
        return {'kind': 'bounds'}
    for k in range(0, m, 3):
        if sum(w[k:k+3]) != 1:
            return {'kind': 'pair', 'block': k//3}
    for i in (1, 2):
        s = 0
        x = Fraction(0)
        for t, g in enumerate(profile[i], 1):
            s += g in S
            x += w[g]
            ok = (t-2*s-1 <= x <= s+1) if i == 1 else (
                x >= max(s-1, (t-s)//2))
            if not ok:
                return {'kind': 'prefix', 'agent': i, 't': t,
                        's': s, 'x': str(x)}
    return None


def repair_one_unit(profile: Sequence[Sequence[int]], owner: Sequence[int]):
    """Proved conditional repair; returns None unless its hypotheses hold.

    S is a suffix in sigma_2, agent 1 is already fair, and X is fixed.
    On R=goods\\S the X-minus-dummy walk must never go below -2.
    At its first -2, the dummy good b must share both a market and sigma_1
    block with its S good a, and b must precede a in sigma_1. Swap a,b.

    The input is allowed to violate agent 2's final fairness constraints.
    Runtime O(m). This is NOT asserted to repair every matching/allocation.
    """
    m = len(owner)
    S0 = {g for g, i in enumerate(owner) if i == 1}
    m, S = validate_matching(profile, S0)
    if len(owner) != m or any(i not in (0, 1, 2) for i in owner):
        raise ValueError('Invalid owner array')
    if any(sorted(owner[k:k+3]) != [0, 1, 2] for k in range(0, m, 3)):
        raise ValueError('The allocation must be market-balanced')
    q = m//3
    if any(g in S for g in profile[2][:2*q]) or set(profile[2][2*q:]) != S:
        return None
    count = [0, 0, 0]
    for g in profile[1]:
        count[owner[g]] += 1
        if count[0] > count[1]+1 or count[2] > count[1]+1:
            return None
    d = 0
    first = None
    for g in profile[2][:2*q]:
        d += 1 if owner[g] == 2 else -1
        if d < -2:
            return None
        if d == -2 and first is None:
            first = g
    if first is None:
        return list(owner)
    b = first
    block = 3*(b//3)
    a = next(g for g in range(block, block+3) if owner[g] == 1)
    rank1 = [0]*m
    for t, g in enumerate(profile[1]):
        rank1[g] = t
    if rank1[a]//3 != rank1[b]//3 or rank1[b] >= rank1[a]:
        return None
    result = list(owner)
    result[a], result[b] = result[b], result[a]
    return result


def encode_3cnf(clauses: Sequence[Sequence[int]]):
    """Linear-size reduction from 3-CNF SAT to fixed-matching completion.

    Repeated literals are allowed. Signed integers identify literals.
    With h clauses the output has m=12*h+6 goods.
    Returns (profile, fixed_S, variable/anchor cycle metadata).
    """
    occurrences = defaultdict(list)
    for j, clause in enumerate(clauses):
        if len(clause) != 3 or any(type(l) is not int or l == 0 for l in clause):
            raise ValueError('Clauses must have exactly three nonzero integer literals')
        for k, literal in enumerate(clause):
            occurrences[abs(literal)].append((j, k))
    groups = [('variable', v, len(occurrences[v])) for v in occurrences]
    groups.append(('anchor', 0, len(clauses)+2))
    sigma1_blocks, market_blocks, cycles = [], [], []
    number = 0
    for kind, variable, d in groups:
        a, b, s = [], [], []
        for k in range(d):
            a.append(number)
            b.append(number+1)
            s.append(number+2)
            number += 3
        for k in range(d):
            sigma1_blocks.append((a[k], b[k], s[k]))
            market_blocks.append((b[k], a[(k+1) % d], s[k]))
        cycles.append({'kind': kind, 'variable': variable, 'a': a, 'b': b, 's': s})
    literal_good, complement_good = {}, {}
    for cycle in cycles[:-1]:
        v = cycle['variable']
        for k, (j, h) in enumerate(occurrences[v]):
            a, b = cycle['a'][k], cycle['b'][k]
            literal_good[j,h], complement_good[j,h] = (
                (a,b) if clauses[j][h] > 0 else (b,a))
    anchor = cycles[-1]
    sigma2 = anchor['a'][:2]
    for j in range(len(clauses)):
        sigma2 += [literal_good[j,k] for k in range(3)]
        sigma2 += [anchor['b'][j]]
        sigma2 += [complement_good[j,k] for k in range(3)]
        sigma2 += [anchor['a'][j+2]]
    sigma2 += anchor['b'][len(clauses):]
    S = [g for cycle in cycles for g in cycle['s']]
    sigma2 += S
    market = [g for block in market_blocks for g in block]
    rename = {g: k for k, g in enumerate(market)}
    convert = lambda sequence: [rename[g] for g in sequence]
    profile = [list(range(number)),
               convert([g for block in sigma1_blocks for g in block]), convert(sigma2)]
    for cycle in cycles:
        for key in ('a', 'b', 's'):
            cycle[key] = convert(cycle[key])
    return profile, convert(S), cycles


def formula_value(clauses: Sequence[Sequence[int]], assignment: Mapping[int, bool]):
    return all(any(bool(assignment[abs(l)]) == (l > 0) for l in clause)
               for clause in clauses)


def formula_owner(profile, S, cycles, assignment: Mapping[int, bool],
                  anchor: bool = True):
    owner = [0]*len(profile[0])
    for g in S:
        owner[g] = 1
    for cycle in cycles:
        sign = anchor if cycle['kind'] == 'anchor' else assignment[cycle['variable']]
        for g in cycle['a' if sign else 'b']:
            owner[g] = 2
    return owner


def solve_encoded_family(clauses: Sequence[Sequence[int]]):
    """Fair allocation for EVERY encoded formula, satisfiable or not, in O(m).

    This is a theorem only for the profiles generated by encode_3cnf.
    The output changes at most one good of the originally fixed S.
    """
    profile, S, cycles = encode_3cnf(clauses)
    assignment = {c['variable']: False for c in cycles if c['kind'] == 'variable'}
    owner = formula_owner(profile, S, cycles, assignment, anchor=True)
    repaired = repair_one_unit(profile, owner)
    if repaired is None:
        raise AssertionError('The proved encoded-family repair hypotheses failed')
    return profile, repaired


def repair_parallel_edge(profile: Sequence[Sequence[int]], owner: Sequence[int]):
    """General one-edge repair lemma, without the sigma_2 suffix hypothesis.

    Preserves X, market balance, own-class balance, and agent 1's fairness.
    Requires agent 2 already fair towards S and at most two behind D.
    Searches all eligible parallel-edge exchanges meeting the lemma's interval
    slack conditions, in O(m) time. Returns None if the premises fail or no such
    exchange exists. No universal repair guarantee is asserted.
    """
    S0 = {g for g, i in enumerate(owner) if i == 1}
    m, S = validate_matching(profile, S0)
    if len(owner) != m or any(i not in (0, 1, 2) for i in owner):
        raise ValueError('Invalid owner array')
    if any(sorted(owner[k:k+3]) != [0, 1, 2] for k in range(0, m, 3)):
        raise ValueError('The allocation must be market-balanced')
    rank1,rank2=[0]*m,[0]*m
    c=[0,0,0]
    for t,g in enumerate(profile[1],1):
        rank1[g]=t-1
        c[owner[g]]+=1
        if c[0]>c[1]+1 or c[2]>c[1]+1:
            return None
    c=[0,0,0];slack=[0]*(m+1);bad=[]
    for t,g in enumerate(profile[2],1):
        rank2[g]=t
        c[owner[g]]+=1
        if c[1]>c[2]+1 or c[0]>c[2]+2:
            return None
        slack[t]=c[2]-c[1]
        if c[0]>c[2]+1:
            bad.append(t)
    if not bad:
        return list(owner)
    first,last=bad[0],bad[-1]
    # Every admissible repair interval contains first, so minima on intervals
    # containing this fixed pivot can all be queried after two linear scans.
    left=[0]*(m+1);right=[0]*(m+1)
    left[first]=right[first]=slack[first]
    for t in range(first-1,-1,-1):
        left[t]=min(slack[t],left[t+1])
    for t in range(first+1,m+1):
        right[t]=min(slack[t],right[t-1])
    for k in range(0,m,3):
        a=next(g for g in range(k,k+3) if owner[g]==1)
        b=next(g for g in range(k,k+3) if owner[g]==0)
        if rank1[a]//3!=rank1[b]//3 or rank1[b]>=rank1[a]:
            continue
        lo,hi=rank2[b],rank2[a]-1
        if not (lo<=first and last<=hi):
            continue
        if min(left[lo],right[hi])<0:
            continue
        answer=list(owner)
        answer[a],answer[b]=answer[b],answer[a]
        return answer
    return None


def try_cycle_repair(profile: Sequence[Sequence[int]]):
    """Linear-time conditional construction using only the rankings.

    Take S to be sigma_2's bottom m/3 goods. It must be a reset matching.
    Orient its residual cycles arbitrarily, except that agent 2 receives its
    top good. Then apply the proved one-unit repair. Guaranteed to succeed on
    EVERY profile produced by encode_3cnf, without knowing its generating
    formula. On other profiles it may return None; that is not infeasibility.
    """
    if len(profile)!=3 or len(profile[0])%3:
        raise ValueError('Requires three rankings and m divisible by three')
    m=len(profile[0]);q=m//3
    if not m:
        validate_matching(profile,[],reset=True)
        return []
    S=list(profile[2][m-q:])
    try:
        components=residual_cycles(profile,S)
    except ValueError:
        return None
    first=profile[2][0]
    choices=[1 if first in sides[1] else 0 for sides in components]
    owner=cycle_owner(m,S,components,choices)
    return repair_one_unit(profile,owner)
