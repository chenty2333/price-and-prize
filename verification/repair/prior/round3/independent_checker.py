"""Direct checker: no cycle reduction, prefix-bound reduction, or SAT logic."""
from typing import Sequence


def check_allocation(profile: Sequence[Sequence[int]], owner: Sequence[int]):
    """Return None on success, otherwise a concrete violation."""
    if len(profile) != 3:
        return {'kind': 'profile_length'}
    m = len(profile[0])
    if any(len(row) != m or sorted(row) != list(range(m)) for row in profile):
        return {'kind': 'permutation'}
    if list(profile[0]) != list(range(m)):
        return {'kind': 'market_ranking'}
    if len(owner) != m or any(type(i) is not int or i not in (0, 1, 2) for i in owner):
        return {'kind': 'owner'}
    for k in range(0, m, 3):
        if len(set(owner[k:k+3])) != len(owner[k:k+3]):
            return {'kind': 'market', 'block': k//3}
    for i, ranking in enumerate(profile):
        counts = [0, 0, 0]
        for t, g in enumerate(ranking, 1):
            counts[owner[g]] += 1
            for j in range(3):
                if counts[j] > counts[i] + 1:
                    return {'kind': 'sd_ef1', 'agent': i, 'towards': j,
                            'prefix_length': t, 'prefix': list(ranking[:t]),
                            'counts': counts.copy()}
    return None


def own_balance(profile: Sequence[Sequence[int]], owner: Sequence[int]) -> bool:
    """Exactly one agent-1 good in each full sigma_1 triple, <=1 in the short one."""
    sigma = profile[1]
    for k in range(0, len(sigma), 3):
        block = sigma[k:k+3]
        count = sum(owner[g] == 1 for g in block)
        if (len(block) == 3 and count != 1) or (len(block) < 3 and count > 1):
            return False
    return True
