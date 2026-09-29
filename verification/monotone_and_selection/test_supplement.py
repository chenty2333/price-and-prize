"""Additional exact original-coordinate checks and nontrivial crossings."""
from itertools import product
from collections import Counter
import random,json
from monotone_two import mixed_path,allocate
from test_monotone import subjective,direct_market,label

rng=random.Random(31603160)
Cts=Counter()
for k in range(3000):
    m=rng.randrange(4,41)
    P=[rng.sample(range(m),m) for _ in (0,1)]
    mode=k%3
    C=frozenset() if mode==0 else frozenset(range(m)) if mode==1 else frozenset(rng.sample(range(m),rng.randrange(1,m)))
    path=mixed_path(P,set(C));T=frozenset(g for g,o in enumerate(path[0]) if (o^(g in C))==0)
    # A nonlinear, sign-monotone value whose initial and final states MUST
    # fail market EF1 in opposite directions (|T| >= 2).
    def value(S):return len((S^C)&T)**2-len(C&T)**2
    labs=[label(a,value,C) for a in path]
    assert labs[0]==1 and labs[-1]==-1 and 0 in labs
    assert all(x*y!=-1 for x,y in zip(labs,labs[1:]))
    w=allocate(P,value,set(C))
    assert subjective(w,P,C) and direct_market(w,value,C)
    Cts['nonlinear_forced_crossing_instances']+=1
    for a in path:
        assert subjective(a,P,C)
        # Check actual signed additive subjective utilities, not transformed counts.
        for scheme in (0,1,2):
            for i in (0,1):
                weight={g:(m-r if scheme==0 else 2**(m-r) if scheme==1 else 1)
                        *(-1 if g in C else 1) for r,g in enumerate(P[i])}
                own={g for g,o in enumerate(a) if o==i};rival=set(range(m))-own
                u=lambda S:sum(weight[g] for g in S)
                assert (u(own)>=u(rival)
                        or any(u(own)>=u(rival-{g}) for g in rival-C)
                        or any(u(own-{g})>=u(rival) for g in own&C))
                Cts['original_signed_subjective_agent_checks']+=1
        Cts['forced_crossing_path_states']+=1
# Common-polarity alignment is necessary for a general theorem.
P=[[0,1],[0,1]];C=frozenset({1})
records=[]
for a in product(range(2),repeat=2):
    records.append({'owner':a,'subjective_signed_SD_EF1':subjective(a,P,C),
                    'all_goods_market_EF1':direct_market(a,lambda S:len(S))})
assert sum(r['subjective_signed_SD_EF1'] for r in records)==2
assert not any(r['subjective_signed_SD_EF1'] and r['all_goods_market_EF1'] for r in records)
print(json.dumps(dict(Cts),indent=2));print('Sign-misalignment counterexample:',records)
json.dump({'counts':dict(Cts),'sign_mismatch':records},open('test_supplement_counts.json','w'),indent=2)
