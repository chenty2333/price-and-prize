"""Independent exhaustive audit of a ten-good pair-selection obstruction.
Refutes lex-earliest-defects, maximum subjective slack, and selecting both
members among minimum-defect witnesses. Does not refute disjoint pairs.
"""
from itertools import product
from collections import Counter

P = [[7,4,2,5,0,8,9,6,3,1],
     [4,2,0,3,7,5,9,1,8,6],
     [4,0,3,2,7,5,9,1,8,6]]

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
    histogram=dict(sorted(Counter(tuple(sorted(d)) for a,d,s in near).items()))
    assert histogram=={(6,):9,(3,6):3,(3,9):2,(6,9):12,(3,6,9):8}
    lex=min(histogram)
    lex_selected=[x for x in near if tuple(sorted(x[1]))==lex]
    assert lex==(3,6) and len(lex_selected)==3
    assert all(not any(d.isdisjoint(x[1]) for a,d,s in near) for x in lex_selected)
    maximum=max(s for a,d,s in near)
    maximizers=[x for x in near if x[2]==maximum]
    assert len(maximizers)==1 and maximum==36 and maximizers[0][1]=={6,9}
    assert not any(d.isdisjoint(maximizers[0][1]) for a,d,s in near)
    minima=[x for x in near if len(x[1])==1]
    assert len(minima)==9 and all(d=={6} for a,d,s in minima)
    assert not any(x[1].isdisjoint(y[1]) for x in minima for y in minima)
    A,DA,_=next(x for x in near if x[1]=={6})
    B,DB,_=next(x for x in near if x[1]=={3,9})
    assert all(x+y<=0 for r in rows(A) for s in rows(B) for x,y in zip(r,s))
    assert not any(set(P[0][:t])==set(P[1][:t])==set(P[2][:t]) for t in range(1,10))
    print('Allocations enumerated:',3**10)
    print('Subjective SD-EF1 allocations:',allsd)
    print('Subjective SD-EF1 and near-block:',len(near))
    print('Defect histogram:',histogram)
    print('Market-block witnesses:',sum(not d for a,d,s in near))
    print('Lex-earliest defect tuple:',lex)
    print('Lex-earliest witnesses / extendable:',len(lex_selected),0)
    print('Maximum subjective slack / maximizers:',maximum,len(maximizers))
    print('Unique maximizer:',list(maximizers[0][0]))
    print('Maximizer defect set / disjoint partners:',sorted(maximizers[0][1]),0)
    print('Minimum-defect witnesses / pairs among them:',len(minima),0)
    print('Covering pair:',list(A),list(B))
    print('Covering defect sets:',sorted(DA),sorted(DB))
    print('Exact cross-row inequalities verified:',36*10)
