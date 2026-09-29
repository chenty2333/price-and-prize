"""Direct, proof-independent checking of every original prefix inequality."""
from typing import Sequence

def validate(profile: Sequence[Sequence[int]]) -> int:
    if len(profile) != 3:
        raise ValueError('Exactly three agents are required.')
    m = len(profile[0])
    for sigma in profile:
        if len(sigma) != m or sorted(sigma) != list(range(m)):
            raise ValueError('Each ranking must be a permutation of 0..m-1.')
    if list(profile[0]) != list(range(m)):
        raise ValueError('Agent 0 must follow the market.')
    return m

def check(profile, owner, agents=(0, 1, 2)):
    m = len(profile[0])
    if len(owner) != m or any(type(x) is not int or not 0 <= x < 3 for x in owner):
        return {'kind': 'owner'}
    for b in range(0, m, 3):
        if len(set(owner[b:b+3])) != len(owner[b:b+3]):
            return {'kind': 'market', 'block': b//3}
    for i in agents:
        counts = [0, 0, 0]
        for t, good in enumerate(profile[i], 1):
            counts[owner[good]] += 1
            for j in range(3):
                if counts[j] > counts[i] + 1:
                    return {'kind': 'prefix', 'agent': i, 'opponent': j,
                            'length': t, 'counts': counts.copy()}
    return None

def own_balanced(profile, owner):
    m = len(owner)
    for b in range(0, m, 3):
        count = sum(owner[g] == 1 for g in profile[1][b:b+3])
        if (b+3 <= m and count != 1) or (b+3 > m and count > 1):
            return False
    return True
