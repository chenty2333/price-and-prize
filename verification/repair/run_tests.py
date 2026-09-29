"""Reproduce the NEW tests; does not relabel old logs as new executions."""
from __future__ import annotations
import argparse, collections, itertools, json, random, sys, time
from pathlib import Path
from independent_checker import validate, check, own_balanced
from round4 import (singleton_violation, solve_one_singleton, first_agent_first,
                    first_priority_adversary, connected_family, relabel_blocks,
                    repeat_profile)
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'prior'/'round3'))
import round3


def class_ranking(m,rng):
    """Random constructive ranking in the condition, with any short block last."""
    q=m//3
    blocks=[]
    for j in range(q):
        b=list(range(3*j,3*j+3));rng.shuffle(b);blocks.append(b)
    state=[0]*q;active=None;out=[]
    while len(out)<3*q:
        options=[j for j in range(q) if state[j]==2]
        if active is None:
            options += [j for j in range(q) if state[j]==0]
        else:
            options.append(active)
        j=rng.choice(options)
        out.append(blocks[j][state[j]])
        state[j]+=1
        if state[j]==1:active=j
        elif state[j]==2:active=None
    last=list(range(3*q,m));rng.shuffle(last)
    return out+last


def all_market(m):
    for assignments in itertools.product(tuple(itertools.permutations(range(3))),repeat=m//3):
        yield [c for a in assignments for c in a]


def fixed_owners(m,S):
    pairs=[[g for g in range(b,b+3) if g not in S]for b in range(0,m,3)]
    for choices in itertools.product((0,1),repeat=m//3):
        owner=[0]*m
        for g in S:owner[g]=1
        for pair,c in zip(pairs,choices):owner[pair[c]]=2
        yield owner


def run(per_m=200, relabels=3000, negatives=5000, sat_trials=3000):
    rng=random.Random(20260929);start=time.perf_counter();suites=[]
    ct=0
    for m in range(31):
        for _ in range(per_m):
            s1=list(range(m));rng.shuffle(s1)
            p=[list(range(m)),s1,class_ranking(m,rng)]
            validate(p);a=solve_one_singleton(p)
            assert a is not None and check(p,a) is None,(p,a)
            ct+=1
    suites.append({'name':'one_singleton_class','profiles':ct,'m_range':[0,30],'failures':0})
    ct=0
    for m in range(31):
        for _ in range(per_m):
            s1=list(range(m));rng.shuffle(s1)
            s2=list(range(m));j=0
            while j+1<m:
                if rng.randrange(2):s2[j],s2[j+1]=s2[j+1],s2[j];j+=2
                else:j+=1
            p=[list(range(m)),s1,s2]
            a=solve_one_singleton(p)
            assert a is not None and check(p,a) is None,(p,a)
            ct+=1
    suites.append({'name':'disjoint_adjacent_swaps_of_market','profiles':ct,'m_range':[0,30],'failures':0})
    auxiliary=random.Random(20260930)
    for _ in range(3000):
        q=auxiliary.randint(1,10);m=3*q
        blocks=[list(range(3*j,3*j+3))for j in range(q)]
        for block in blocks:auxiliary.shuffle(block)
        auxiliary.shuffle(blocks)
        s2=[g for block in blocks for g in block]
        for j in range(1,q):
            if auxiliary.randrange(2):s2[3*j-1],s2[3*j]=s2[3*j],s2[3*j-1]
        s1=list(range(m));auxiliary.shuffle(s1)
        p=[list(range(m)),s1,s2];a=solve_one_singleton(p)
        assert a is not None and check(p,a)is None
    suites.append({'name':'reordered_blocks_with_boundary_swaps','profiles':3000,'m_range':[3,30],'seed':20260930,'failures':0})
    attempts=0;ct=0
    while ct<negatives:
        m=3*rng.randint(2,10);s2=list(range(m));rng.shuffle(s2);attempts+=1
        result=first_priority_adversary(s2)
        if result is None:continue
        p,bad=result;a=first_agent_first(p)
        witness=check(p,a)
        assert witness is not None and witness['agent']==2,(p,a,bad)
        t=bad['length'];c=[sum(a[g]==i for g in p[2][:t]) for i in range(3)]
        assert c[1]-c[2]==len(bad['singleton_blocks'])>=2
        ct+=1
    suites.append({'name':'priority_rule_necessity_witnesses','witnesses':ct,'sampled_rankings':attempts,'m_range':[6,30],'failures':0})
    for _ in range(relabels):
        d=connected_family(rng.randint(2,5));p,owners,sets=relabel_blocks(d['profile'],[d['own_owner'],d['free_owner']],[d['S']],rng)
        validate(p);S=set(sets[0])
        for a in owners:assert check(p,a) is None
        assert own_balanced(p,owners[0]) and not own_balanced(p,owners[1])
        assert sum(owners[0][g]!=1 for g in S)==len(p[0])//6
        assert sum(owners[1][g]!=1 for g in S)==1
    suites.append({'name':'connected_distance_family_random_isomorphisms','profiles':relabels,'allocations_per_profile':2,'m_values':[12,18,24,30],'failures':0})
    for _ in range(relabels):
        copies=rng.randint(1,2);d=connected_family(2)
        p,owners,sets=relabel_blocks(d['profile'],[d['own_owner'],d['free_owner']],[d['S']],rng)
        for a in owners:
            pp,aa,SS=repeat_profile(p,a,sets[0],copies)
            validate(pp);assert check(pp,aa) is None
            assert sum(aa[g]!=1 for g in SS)==copies*sum(a[g]!=1 for g in sets[0])
    suites.append({'name':'repeated_common_modules_random_isomorphisms','profiles':relabels,'allocations_per_profile':2,'m_values':[12,24],'failures':0})
    nontrivial=0
    for _ in range(sat_trials):
        k=rng.randrange(3);v=rng.randint(1,6)
        f=[[rng.choice((-1,1))*rng.randint(1,v) for h in range(3)] for j in range(k)]
        p,S,cycles=round3.encode_3cnf(f)
        a=round3.try_cycle_repair(p)
        assert a is not None and check(p,a) is None and own_balanced(p,a)
        changes=sum(a[g]!=1 for g in S)
        assert changes<=1;nontrivial+=changes>0
    suites.append({'name':'audited_SAT_family_rankings_only_repair','profiles':sat_trials,'m_values':[6,18,30],'nontrivial_repairs':nontrivial,'failures':0})
    distance_records=[]
    for r in range(2,6):
        d=connected_family(r);p=d['profile'];m=len(p[0]);q=m//3;S=set(d['S'])
        fixed=sum(check(p,a) is None for a in fixed_owners(m,S));assert fixed==0
        rank1=[0]*m
        for t,g in enumerate(p[1]):rank1[g]=t
        histogram=collections.Counter()
        for picks in itertools.product(range(3),repeat=q):
            T=[3*j+picks[j]for j in range(q)]
            if len({rank1[g]//3 for g in T})==q:
                histogram[len(S-set(T))]+=1
        assert min(t for t in histogram if t)>0
        assert min(t for t in histogram if t)==r
        rec={'r':r,'m':m,'profile':p,'S':sorted(S),'fixed_completions':fixed,
             'matching_distance_histogram':dict(sorted(histogram.items())),
             'own_owner':d['own_owner'],'free_owner':d['free_owner']}
        if r<=3:
            all_hist=collections.Counter();own_hist=collections.Counter();examined=0
            for a in all_market(m):
                examined+=1
                if check(p,a) is None:
                    h=sum(a[g]!=1 for g in S);all_hist[h]+=1
                    if own_balanced(p,a):own_hist[h]+=1
            assert min(all_hist)==1 and min(own_hist)==r
            rec.update({'all_market_examined':examined,'fair_distance_histogram':dict(sorted(all_hist.items())),
                        'own_fair_distance_histogram':dict(sorted(own_hist.items()))})
        distance_records.append(rec)
    # Large r tests are direct checks, NOT brute force.
    for r in range(2,101):
        d=connected_family(r)
        for a in (d['own_owner'],d['free_owner']):assert check(d['profile'],a) is None
    d=connected_family(1000)
    for a in (d['own_owner'],d['free_owner']):assert check(d['profile'],a) is None
    small_repair_profile=[list(range(6)),[3,4,2,0,1,5],[2,5,0,1,3,4]]
    small_hist=collections.Counter();small_total=0
    for a in all_market(6):
        if check(small_repair_profile,a)is None:
            small_total+=1
            if own_balanced(small_repair_profile,a):
                small_hist[sum(a[g]!=1 for g in (2,5))]+=1
    assert small_total==18 and dict(small_hist)=={2:12}
    small_repair={'profile':small_repair_profile,'S':[2,5],
                  'market_allocations_examined':36,'fair':small_total,
                  'own_fair_distance_histogram':dict(small_hist),
                  'witness_owner':[0,1,2,0,1,2]}
    small=[[0,1,2,3],[3,0,2,1],[3,0,1,2]]
    bad=first_agent_first(small);w=check(small,bad);assert w is not None
    old=[
      ([0,3,1,6,4,9,7,10,2,5,8,11],[0,6,1,3,7,9,4,10,2,5,8,11],104,36),
      ([6,0,7,9,4,5,3,1,2,8],[7,4,5,6,8,9,0,3,2,1],87,14),
      ([0,5,4,9,1,8,2,3,7,6],[5,4,8,9,3,6,0,7,1,2],57,18),
      ([9,4,2,1,6,5,7,3,8,0],[1,4,5,3,6,0,9,2,7,8],78,8)]
    old_records=[]
    for s1,s2,total,own_total in old:
        m=len(s1);p=[list(range(m)),s1,s2];ct=co=0
        blocks=[tuple(itertools.permutations(range(3),min(3,m-b)))for b in range(0,m,3)]
        for seq in itertools.product(*blocks):
            a=[c for colors in seq for c in colors]
            if check(p,a) is None:ct+=1;co+=own_balanced(p,a)
        assert(ct,co)==(total,own_total)
        candidate=solve_one_singleton(p)
        if candidate is not None:assert check(p,candidate) is None
        old_records.append({'profile':p,'fair':ct,'own_fair':co,'new_condition_applies':candidate is not None})
    return {'status':'PASS','seed':20260929,'suites':suites,'distance_family':distance_records,
            'large_direct_checks':{'r_range':[2,100],'additional_r':1000,'largest_m':6000,'brute_forced':False},
            'small_full_block_repair_failure_without_suffix':small_repair,
            'minimal_priority_failure':{'profile':small,'owner':bad,'witness':w},
            'previous_obstructions':old_records,'seconds':time.perf_counter()-start}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--per-m',type=int,default=200)
    ap.add_argument('--relabels',type=int,default=3000);ap.add_argument('--negatives',type=int,default=5000)
    ap.add_argument('--sat-trials',type=int,default=3000);ap.add_argument('--output',default='test_results.json')
    ar=ap.parse_args();result=run(ar.per_m,ar.relabels,ar.negatives,ar.sat_trials)
    Path(ar.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ('distance_family','previous_obstructions')},indent=2))
