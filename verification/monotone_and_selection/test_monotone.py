"""Independent direct-definition audits of monotone_two.sd_path.
Valuations are integer subset tables (small tests) or monotone hypergraph
valuations (large tests). No numerical tolerance and no MILP.
"""
from itertools import permutations, product
from collections import Counter
import json, random, time
from monotone_two import sd_path, mixed_path, allocate


def subjective(a, rankings, C=frozenset()):
    # A negative item is 'beneficially held' by the opposite owner.
    for i,ranking in enumerate(rankings):
        own=other=0
        for g in ranking:
            bowner = 1-a[g] if g in C else a[g]
            own += bowner == i
            other += bowner != i
            if own < other-1: return False
    return True


def direct_market(a,value,C=frozenset()):
    bundles=[frozenset(g for g,o in enumerate(a) if o==i) for i in (0,1)]
    values=[value(s) for s in bundles]
    for i in (0,1):
        j=1-i
        if values[i]>=values[j]: continue
        ok=False
        for g in bundles[j]:
            if g not in C and values[i]>=value(bundles[j]-{g}): ok=True
        for g in bundles[i]:
            if g in C and value(bundles[i]-{g})>=values[j]: ok=True
        if not ok: return False
    return True


def label(a,value,C=frozenset()):
    if direct_market(a,value,C): return 0
    s0=frozenset(g for g,o in enumerate(a) if o==0)
    s1=frozenset(range(len(a)))-s0
    return 1 if value(s0)>value(s1) else -1


def check_path(p,path,C=frozenset()):
    m=len(p[0])
    assert len(path)==(m+1)//2+1
    assert path[-1]==tuple(1-i for i in path[0])
    assert all(subjective(a,p,C) for a in path)
    for a,b in zip(path,path[1:]):
        changed=[g for g in range(m) if a[g]!=b[g]]
        assert 1<=len(changed)<=2
        if len(changed)==2:
            x,y=changed
            assert (a[x] ^ (x in C)) != (a[y] ^ (y in C))


def monotone_table(rng,m):
    # Arbitrary integer seed values, then the least coordinatewise monotone majorant.
    v=[rng.randrange(1001) for _ in range(1<<m)];v[0]=0
    for g in range(m):
        for s in range(1<<m):
            if s&(1<<g): v[s]=max(v[s],v[s^(1<<g)])
    assert all(v[s]<=v[s|(1<<g)] for s in range(1<<m) for g in range(m))
    return v


def vlookup(table,C=frozenset()):
    cmask=sum(1<<g for g in C)
    return lambda S: table[sum(1<<g for g in S)^cmask]-table[cmask]


def run():
    rng=random.Random(20260929)
    out=Counter()
    # Exhaust all ranking pairs modulo common item relabelling, through eight goods.
    for m in range(1,9):
        for p2 in permutations(range(m)):
            p=[list(range(m)),list(p2)]
            path=sd_path(p);check_path(p,path)
            out['exhaustive_ranking_profiles']+=1
            out['exhaustive_path_allocations']+=len(path)
    print('Exhaustive ranking/path audit:',dict(out),flush=True)
    # All 168 monotone Boolean functions on four goods, and all 24 profiles modulo relabelling.
    m=4
    for bits in range(1<<(1<<m)):
        tab=[(bits>>s)&1 for s in range(1<<m)]
        if not all(tab[s]<=tab[s|(1<<g)] for s in range(1<<m) for g in range(m)):continue
        out['boolean_monotone_functions']+=1
        for p2 in permutations(range(m)):
            p=[list(range(m)),list(p2)];path=sd_path(p)
            val=vlookup(tab)
            assert any(direct_market(a,val) for a in path)
            out['boolean_profile_value_instances']+=1
    print('Monotone Boolean audit:', {k:v for k,v in out.items() if 'boolean' in k},flush=True)
    # Three distinct batches: goods, chores, common-sign mixed, 3000 instances each.
    for mode in ('goods','chores','mixed'):
        for k in range(3000):
            m=rng.randrange(2,11)
            p=[rng.sample(range(m),m) for _ in (0,1)]
            if mode=='goods':C=frozenset()
            elif mode=='chores':C=frozenset(range(m))
            else:C=frozenset(rng.sample(range(m),rng.randrange(1,m)))
            tab=monotone_table(rng,m);value=vlookup(tab,C)
            path=mixed_path(p,set(C));check_path(p,path,C)
            labels=[label(a,value,C) for a in path]
            assert 0 in labels and all(x*y!=-1 for x,y in zip(labels,labels[1:]))
            witness=allocate(p,value,set(C))
            assert subjective(witness,p,C) and direct_market(witness,value,C)
            # Brute-force every allocation independently, as a separate sanity check.
            count=sum(subjective(a,p,C) and direct_market(a,value,C)
                      for a in product(range(2),repeat=m))
            assert count>0
            out[mode+'_random_table_instances']+=1
            out[mode+'_path_states']+=len(path)
            out[mode+'_allocations_bruteforced']+=1<<m
        print(mode,'integer subset-table checks passed:',out[mode+'_random_table_instances'],flush=True)
    # Larger oracle instances, with complementarities and common signs.
    for k in range(3000):
        m=rng.randrange(11,61);p=[rng.sample(range(m),m) for _ in (0,1)]
        mode=k%3
        C=frozenset() if mode==0 else frozenset(range(m)) if mode==1 else frozenset(rng.sample(range(m),rng.randrange(1,m)))
        cmask=sum(1<<g for g in C)
        edges=[(sum(1<<g for g in rng.sample(range(m),rng.randrange(1,min(m,8)+1))),rng.randrange(1,101)) for _ in range(30)]
        baseline=sum(w for e,w in edges if cmask&e==e)
        def val(S):
            s=sum(1<<g for g in S)^cmask
            return sum(w for e,w in edges if s&e==e)-baseline
        path=mixed_path(p,set(C));check_path(p,path,C)
        labels=[label(a,val,C) for a in path]
        assert 0 in labels and all(x*y!=-1 for x,y in zip(labels,labels[1:]))
        witness=allocate(p,val,set(C))
        assert subjective(witness,p,C) and direct_market(witness,val,C)
        out['large_oracle_instances']+=1
        out['large_path_states']+=len(path)
    print('All exact tests passed.');print(json.dumps(dict(out),indent=2),flush=True)
    json.dump(dict(out),open('test_monotone_counts.json','w'),indent=2)

if __name__=='__main__':run()
