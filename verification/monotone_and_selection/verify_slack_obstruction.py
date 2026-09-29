"""Standalone exact counterexample to maximum-total-subjective-slack selection.
Enumerates all 3**10 allocations with Python standard-library integers.
"""
from itertools import product
from collections import Counter

P = [[8,2,0,7,5,6,9,1,3,4],
     [0,2,5,3,9,8,1,4,6,7],
     [8,5,3,2,0,4,6,9,7,1]]
C = (0,1,1,2,2,2,0,0,2,1)

def subjective_score(a):
    score=0
    for i,r in enumerate(P):
        counts=[0,0,0]
        for g in r:
            counts[a[g]]+=1
            slack=1+counts[i]-max(counts[j] for j in range(3) if j!=i)
            if slack<0:return None
            score+=slack
    return score

def defects(a):
    counts=[0,0,0];D=set()
    for t,i in enumerate(a,1):
        counts[i]+=1
        if t%3:
            if max(counts)-min(counts)>1:return None
        elif len(set(counts))>1:D.add(t)
    return frozenset(D)

def rows(a):
    result=[]
    for i in range(3):
        for j in range(3):
            if i==j:continue
            counts=[0,0,0];row=[]
            for k in a:
                counts[k]+=1
                row.append(counts[j]-counts[i]-int(counts[j]>0))
            result.append(row)
    return result

if __name__=='__main__':
    allsd=0;near=[]
    for a in product(range(3),repeat=10):
        score=subjective_score(a)
        if score is None:continue
        allsd+=1;D=defects(a)
        if D is not None:near.append((a,D,score))
    maximum=max(s for a,d,s in near)
    maximizers=[a for a,d,s in near if s==maximum]
    assert maximum==43 and maximizers==[C]
    assert not any(not d for a,d,s in near)
    assert not any(d.isdisjoint(defects(C)) for a,d,s in near)
    A,DA,_=next(x for x in near if x[1]=={3})
    B,DB,_=next(x for x in near if x[1]=={6})
    assert DA.isdisjoint(DB)
    assert all(x+y<=0 for r in rows(A) for s in rows(B) for x,y in zip(r,s))
    print('Allocations enumerated:',3**10)
    print('Subjective SD-EF1 allocations:',allsd)
    print('Subjective SD-EF1 and near-block:',len(near))
    print('Market-block witnesses:',sum(not d for a,d,s in near))
    print('Maximum total subjective slack:',maximum)
    print('Maximizers:',len(maximizers))
    print('Unique maximizer:',list(C))
    print('Its defects:',sorted(defects(C)))
    print('Its near-block disjoint partners:',sum(d.isdisjoint(defects(C)) for a,d,s in near))
    print('Different covering pair:',list(A),list(B))
    print('Defects of covering pair:',sorted(DA),sorted(DB))
    print('Exact cross-row inequalities verified:',36*10)
    print('Defect histogram:',dict(sorted(Counter(tuple(sorted(d)) for a,d,s in near).items())))
