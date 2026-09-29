from itertools import product
from collections import Counter

P = [[7,4,2,5,0,8,9,6,3,1],
     [4,2,0,3,7,5,9,1,8,6],
     [4,0,3,2,7,5,9,1,8,6]]

def score(a):
    total = 0
    for i, ranking in enumerate(P):
        c = [0,0,0]
        for g in ranking:
            c[a[g]] += 1
            slack = 1+c[i]-max(c[j] for j in range(3) if j != i)
            if slack < 0:
                return None
            total += slack
    return total

def defects(a):
    c, D = [0,0,0], set()
    for t, i in enumerate(a, 1):
        c[i] += 1
        if t % 3 and max(c)-min(c) > 1:
            return None
        if t % 3 == 0 and len(set(c)) > 1:
            D.add(t)
    return frozenset(D)

def rows(a):
    cs = [[a[:t].count(i) for i in range(3)] for t in range(1,11)]
    return [[c[j]-c[i]-int(c[j]>0) for c in cs]
            for i in range(3) for j in range(3) if i != j]

sd, near = 0, []
for a in product(range(3), repeat=10):
    s = score(a)
    if s is not None:
        sd += 1
        D = defects(a)
        if D is not None:
            near.append((a,D,s))
H = Counter(tuple(sorted(D)) for a,D,s in near)
assert H == {(6,):9, (3,6):3, (3,9):2, (6,9):12, (3,6,9):8}
lex = min(H)
assert all(not D.isdisjoint(lex) for a,D,s in near)
best = [x for x in near if x[2] == max(s for a,D,s in near)]
assert len(best) == 1 and best[0][1] == {6,9} and best[0][2] == 36
assert all(not D.isdisjoint(best[0][1]) for a,D,s in near)
minima = [x for x in near if len(x[1]) == 1]
assert all(D == {6} for a,D,s in minima)
A = next(a for a,D,s in near if D == {6})
B = next(a for a,D,s in near if D == {3,9})
assert all(x+y <= 0 for r in rows(A) for s in rows(B) for x,y in zip(r,s))
print('Enumerated / SD-EF1 / near-block:', 3**10, sd, len(near))
print('Defect histogram:', dict(sorted(H.items())))
print('Lex-first defects; number of partners:', lex, 0)
print('Unique maximum slack; defects; partners:', best[0][2], sorted(best[0][1]), 0)
print('Minimum-defect witnesses; pairs among them:', len(minima), 0)
print('Covering pair:', A, B)
print('Exact cross-row inequalities checked:', 360)
