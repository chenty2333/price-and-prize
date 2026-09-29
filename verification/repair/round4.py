"""New round-four constructions. No unrestricted all-m existence claim.

solve_one_singleton: O(m) allocation on its explicitly recognized ranking class.
first_priority_adversary: O(m) witness to failure of the priority rule outside
    that class (full market triples).
connected_family: O(m) explicit matching-distance separation, m=6r.
repeat_profile: common-module composition; not valid for interleaved modules.
"""
from __future__ import annotations
from typing import Sequence


def _validate(profile):
    if len(profile) != 3:
        raise ValueError('Exactly three agents are required.')
    m = len(profile[0])
    if list(profile[0]) != list(range(m)):
        raise ValueError('Agent 0 must follow the market.')
    for s in profile:
        if len(s) != m:
            raise ValueError('Ranking lengths differ.')
        seen = [False]*m
        for g in s:
            if type(g) is not int or not 0 <= g < m or seen[g]:
                raise ValueError('Rankings must be permutations of 0..m-1.')
            seen[g] = True
    return m


def singleton_violation(sigma: Sequence[int]):
    """First prefix meeting at least two market blocks in exactly one good.

    The interface requires a permutation on complete triples. Return None if
    no such prefix exists. This is a recognition test, not a fairness checker.
    """
    M = len(sigma)
    if M % 3:
        raise ValueError('A multiple of three goods is required.')
    seen = [False]*M
    for g in sigma:
        if type(g) is not int or not 0 <= g < M or seen[g]:
            raise ValueError('A permutation of 0..M-1 is required.')
        seen[g] = True
    counts = [0]*(M//3)
    singletons = 0
    for t, g in enumerate(sigma, 1):
        b = g//3
        if counts[b] == 0:
            singletons += 1
        elif counts[b] == 1:
            singletons -= 1
        counts[b] += 1
        if singletons >= 2:
            return {'length': t, 'singleton_blocks':
                    [b for b, c in enumerate(counts) if c == 1]}
    return None


def first_agent_first(profile):
    """Priority construction WITHOUT an existence guarantee outside its class.

    Pad with universally worst virtual goods. In each block agent 1 chooses
    first by sigma1, agent 2 chooses next by sigma2, and the dummy takes the rest.
    """
    m = _validate(profile)
    M = 3*((m+2)//3)
    p = [list(s)+list(range(m, M)) for s in profile]
    rank = [[0]*M for _ in range(3)]
    for i in (1, 2):
        for t, g in enumerate(p[i]):
            rank[i][g] = t
    owner = [0]*M
    for b in range(0, M, 3):
        a = min(range(b, b+3), key=lambda g: rank[1][g])
        x = min((g for g in range(b, b+3) if g != a),
                key=lambda g: rank[2][g])
        owner[a], owner[x] = 1, 2
    return owner[:m]


def solve_one_singleton(profile):
    """O(m) theorem: return a fair allocation on the recognized class.

    None means the sufficient ranking hypothesis fails, NOT infeasibility.
    The returned allocation need not retain agent 1's own-class balance.
    """
    m = _validate(profile)
    M = 3*((m+2)//3)
    sigma2 = list(profile[2])+list(range(m, M))
    # Input is already validated; avoid sorting in the linear-time algorithm.
    counts = [0]*(M//3)
    active = 0
    for g in sigma2:
        b = g//3
        active += (counts[b] == 0) - (counts[b] == 1)
        counts[b] += 1
        if active > 1:
            return None
    return first_agent_first(profile)


def first_priority_adversary(sigma2):
    """Construct a sigma1 on which first-agent-first fails.

    This applies exactly when the one-singleton-prefix condition fails, on
    complete triples. Gives a concrete obstruction to this RULE, not to B.
    """
    sigma2 = list(sigma2)
    bad = singleton_violation(sigma2)
    if bad is None:
        return None
    m = len(sigma2)
    prefix = set(sigma2[:bad['length']])
    chosen = []
    for b in range(0, m, 3):
        available = [g for g in range(b, b+3) if g in prefix]
        chosen.append(available[0] if available else b)
    preferred = set(chosen)
    sigma1 = chosen+[g for g in range(m) if g not in preferred]
    return [list(range(m)), sigma1, sigma2], bad


def connected_family(r: int):
    """Connected simple block graph; exact own-balanced repair distance r.

    m=6r, r>=2.  The prescribed reset matching S is a sigma2 suffix and is
    infeasible. 'own_owner' is fair, own-balanced, and replaces r S goods.
    'free_owner' is fair, replaces one S good, and is NOT own-balanced.
    """
    if type(r) is not int or r < 2:
        raise ValueError('r must be an integer >=2.')
    q = 2*r
    a = lambda j: 3*(j % q)
    b = lambda j: 3*(j % q)+1
    z = lambda j: 3*(j % q)+2
    sigma1 = [g for j in range(q) for g in (a(j-1), b(j-2), z(j))]
    sigma2 = [a(0), a(1), b(0), b(1), b(2), b(3)]
    sigma2 += [a(j) for j in range(2, q)]
    sigma2 += [b(j) for j in range(4, q)]
    sigma2 += [z(j) for j in range(q)]
    S = set(z(j) for j in range(q))
    own_owner = [c for j in range(q)
                 for c in ((2, 1, 0) if j % 2 == 0 else (2, 0, 1))]
    newS = (S-{z(q-1)}) | {a(q-1)}
    X = ({a(j) for j in range(q)}-{a(2), a(q-1)}) | {b(2), b(q-1)}
    free_owner = [1 if g in newS else 2 if g in X else 0 for g in range(3*q)]
    return {'profile': [list(range(3*q)), sigma1, sigma2],
            'S': sorted(S), 'own_owner': own_owner, 'free_owner': free_owner,
            'own_distance': r, 'free_distance': 1}


def relabel_blocks(profile, owners, sets, rng):
    """Random isomorphism of market block systems; rankings remain strict.

    This produces structural relabelings, not independent unstructured profiles.
    Complete market triples are required. owners and sets are lists of objects.
    """
    m = _validate(profile)
    if m % 3:
        raise ValueError('Complete triples are required.')
    blocks = list(range(m//3)); rng.shuffle(blocks)
    image = [0]*m
    for new, old in enumerate(blocks):
        goods = list(range(3*old, 3*old+3)); rng.shuffle(goods)
        for pos, good in enumerate(goods):
            image[good] = 3*new+pos
    result = [list(range(m))]+[[image[g] for g in s] for s in profile[1:]]
    transformed = []
    for owner in owners:
        out = [-1]*m
        for good in range(m):
            out[image[good]] = owner[good]
        transformed.append(out)
    return result, transformed, [[image[g] for g in S] for S in sets]


def repeat_profile(profile, owner, S, copies):
    """Copy into common modules in the same order in both subjective rankings."""
    m = _validate(profile)
    if m % 3 or type(copies) is not int or copies < 1:
        raise ValueError('Complete triples and a positive copy count are required.')
    result = [list(range(copies*m))]
    for i in (1, 2):
        result.append([g+c*m for c in range(copies) for g in profile[i]])
    return result, list(owner)*copies, [g+c*m for c in range(copies) for g in S]
