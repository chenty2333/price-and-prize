"""Standard-library exhaustive refutation and independent positive cover check."""
from itertools import product
from collections import Counter

P = [
 [0,7,2,5,1,11,8,3,6,9,4,10],
 [0,11,2,7,1,5,10,9,3,8,6,4],
 [5,0,2,11,7,1,4,3,9,8,6,10],
]
A = (1,0,2,1,2,2,0,0,2,1,0,1)
B = (1,1,0,2,0,2,1,2,0,2,1,0)

def sd(a):
    for i, ranking in enumerate(P):
        c = [0,0,0]
        for g in ranking:
            c[a[g]] += 1
            if max(c) > c[i]+1:
                return False
    return True

def defects(a):
    c, D = [0,0,0], set()
    for t, i in enumerate(a, 1):
        c[i] += 1
        if t % 3 and max(c)-min(c) > 1:
            return None
        if t % 3 == 0 and len(set(c)) > 1:
            D.add(t)
    return frozenset(D)

def positive_rows(a):
    cs = [[a[:t].count(i) for i in range(3)] for t in range(1,13)]
    R = {tuple(c[j]-c[i]-int(c[j]>0) for c in cs)
         for i in range(3) for j in range(3) if i != j}
    return [r for r in R if max(r) > 0]

if __name__ == '__main__':
    sd_count, near = 0, []
    for a in product(range(3), repeat=12):
        if not sd(a):
            continue
        sd_count += 1
        D = defects(a)
        if D is not None:
            near.append((a,D))
    H = Counter(tuple(sorted(D)) for a,D in near)
    assert sd_count == 1080
    assert H == {(3,6):12, (3,9):6, (6,9):36, (3,6,9):4}
    pairs = sum(D.isdisjoint(E) for k,(a,D) in enumerate(near)
                for b,E in near[k+1:])
    minimum = min(map(lambda x:len(x[1]), near))
    minima = [a for a,D in near if len(D)==minimum]
    assert pairs == 0 and minimum == 2 and len(minima) == 54
    assert sd(A) and sd(B)
    RA, RB = positive_rows(A), positive_rows(B)
    HA = tuple(max(r[t] for r in RA) for t in range(12))
    HB = tuple(max(r[t] for r in RB) for t in range(12))
    assert all(x+y <= 0 for x,y in zip(HA,HB))
    print('All allocations enumerated:', 3**12)
    print('Subjectively SD-EF1:', sd_count)
    print('Also near-block-balanced:', len(near))
    print('Defect histogram:', dict(sorted(H.items())))
    print('Minimum defects / minimum witnesses:', minimum, len(minima))
    print('Unordered disjoint-defect pairs:', pairs)
    print('Cover A positive-row envelope:', HA)
    print('Cover B positive-row envelope:', HB)
    print('Sum:', tuple(x+y for x,y in zip(HA,HB)))
    print('Exact unrestricted two-allocation cover: True')
