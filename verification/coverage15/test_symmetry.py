"""Independent checks of endpoint group actions, stabilizer counts and scope."""
from __future__ import annotations
from collections import Counter
from itertools import permutations, product
from math import factorial
import json, pathlib, random, time
from verify_python import candidates_for


def key(blocks): return tuple(sorted(tuple(sorted(B)) for B in blocks))

def orbit_explicit(blocks,m,h):
    result=set()
    for head in permutations(range(h)):
        for tail in permutations(range(m-3,m)):
            def image(g):
                if g<h:return head[g]
                if g>=m-3:return tail[g-(m-3)]
                return g
            result.add(key([[image(g) for g in B] for B in blocks]))
    return result


def signature_and_weight(blocks,m,h):
    symbols=[];denom=1
    for B in blocks:
        a=sum(g<h for g in B);b=sum(g>=m-3 for g in B)
        fixed=tuple(sorted(g for g in B if h<=g<m-3))
        symbols.append((fixed,a,b));denom*=factorial(a)*factorial(b)
    for (fixed,a,b),number in Counter(symbols).items():
        if not fixed:denom*=factorial(number)
    numerator=factorial(h)*6
    assert numerator%denom==0
    return tuple(sorted(symbols)),numerator//denom


def fair1(owner):
    counts=[0,0,0]
    for c in owner:
        counts[c]+=1
        if counts[0]>counts[1]+1 or counts[2]>counts[1]+1:return False
    return True


def permitted(owner,family,first_full):
    m=len(owner)
    if first_full and len(set(owner[:3]))!=3:return False
    counts=[owner[t:t+3].count(1) for t in range(0,m,3)]
    if family=='own' and any(c!=1 for c in counts):return False
    if family=='shift1':
        if sum(max(c-1,0) for c in counts)>1:return False
        height=0
        for c in counts:
            height+=c-1
            if height<0:return False
    return fair1(owner)


def main():
    root=pathlib.Path(__file__).resolve().parent;start=time.perf_counter();rng=random.Random(20260930)
    n=0;suborbit_checks=0
    for m in (6,9,12,15,18):
        for trial in range(1000):
            goods=list(range(m));rng.shuffle(goods);blocks=key([goods[b:b+3] for b in range(0,m,3)])
            G=orbit_explicit(blocks,m,3);sg,wg=signature_and_weight(blocks,m,3)
            assert wg==len(G)
            Hsignatures={}
            for B in G:
                assert signature_and_weight(B,m,3)==(sg,wg)
                s,w=signature_and_weight(B,m,2);Hsignatures[s]=w
            assert sum(Hsignatures.values())==wg
            assert len(Hsignatures)<=3
            H=orbit_explicit(blocks,m,2);sh,wh=signature_and_weight(blocks,m,2)
            assert len(H)==wh
            assert all(signature_and_weight(B,m,2)==(sh,wh) for B in H)
            n+=1;suborbit_checks+=len(Hsignatures)
    invariance_checks=0
    for m in (6,9):
        for trial in range(20):
            goods=list(range(m));rng.shuffle(goods);blocks=[goods[b:b+3] for b in range(0,m,3)]
            for colors in product(tuple(permutations(range(3))),repeat=m//3):
                owner=[-1]*m
                for B,C in zip(blocks,colors):
                    for g,c in zip(B,C):owner[g]=c
                for h,firstfull in ((2,False),(3,True)):
                    for family in ('own','shift1'):
                        before=permitted(owner,family,firstfull)
                        for head in permutations(range(h)):
                            for tail in permutations(range(m-3,m)):
                                out=owner.copy()
                                for g in range(h):out[head[g]]=owner[g]
                                for g in range(m-3,m):out[tail[g-(m-3)]]=owner[g]
                                assert permitted(out,family,firstfull)==before
                                invariance_checks+=1
    # Negative control: S_3 on the first triple is NOT valid without first-full.
    owner=[1,0,0,1,2,2];transformed=[0,0,1,1,2,2]
    assert permitted(owner,'own',False) and not permitted(transformed,'own',False)
    report={'status':'PASS','seed':20260930,'random_partitions':n,'m_values':[6,9,12,15,18],
            'explicit_suborbit_checks':suborbit_checks,'candidate_membership_invariance_checks':invariance_checks,
            'negative_control':{'owner':owner,'swap_labels':[0,2],'transformed_owner':transformed,
                                'first_triple_full_is_necessary_for_this_symmetry':True},
            'seconds':time.perf_counter()-start}
    (root/'symmetry_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
