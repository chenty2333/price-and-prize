"""Executed random and deterministic round-five construction tests.

Fairness is checked by the direct checker, not by the search inequalities.
No claim of all-size existence follows from a finite random suite.
"""
from __future__ import annotations
import argparse, importlib.util, json, pathlib, random, time
from collections import Counter
from itertools import permutations, product
from independent_checker import check, own_balanced
from round5 import (counterexample13, shifted_modules, triple_counts, shift_budget,
                    forward_transfers, search_shift_balanced, solve_up_to_15,
                    solve_common_modules)
ROOT=pathlib.Path(__file__).resolve().parent


def checked(profile,owner,stats=None):
    error=check(profile,owner)
    assert error is None, (profile,owner,error)
    if stats is not None:
        M=stats['padded_m'];m=len(owner)
        pp=[s+list(range(m,M)) for s in profile]
        padded=stats['padded_owner']
        assert check(pp,padded) is None
        counts=triple_counts(pp,padded)
        moves=forward_transfers(counts)
        reconstruct=[1]*len(counts)
        for a,b in moves:
            assert a<b;reconstruct[a]+=1;reconstruct[b]-=1
        assert reconstruct==counts
        assert len(moves)==stats['shift_budget']==shift_budget(pp,padded)


def random_module_profile(m,rng):
    M=3*((m+2)//3);left=M;start=0;modules=[]
    while left:
        width=3*rng.randint(1,min(5,left//3));modules.append(list(range(start,start+width)))
        start+=width;left-=width
    result=[list(range(m))]
    for i in (1,2):
        order=list(range(len(modules)))
        if m<M:
            tail=order.pop();rng.shuffle(order);order.append(tail)
        else:rng.shuffle(order)
        sigma=[]
        for j in order:
            goods=[g for g in modules[j] if g<m];rng.shuffle(goods);sigma.extend(goods)
        result.append(sigma)
    return result


def module_isomorphism(copies,rng):
    base,own=shifted_modules(1);m=15*copies
    local=[];witness=[-1]*m
    for j in range(copies):
        block_order=list(range(5));rng.shuffle(block_order);image=[0]*15
        for new,old in enumerate(block_order):
            goods=list(range(3*old,3*old+3));rng.shuffle(goods)
            for pos,g in enumerate(goods):image[g]=15*j+3*new+pos
        local.append([[image[g] for g in s] for s in base[1:]])
        for g in range(15):witness[image[g]]=own[g]
    profile=[list(range(m))]
    for i in (0,1):
        order=list(range(copies));rng.shuffle(order)
        profile.append([g for j in order for g in local[j][i]])
    return profile,witness


def exact_counts(profile):
    m=len(profile[0]);out=Counter();examples=[]
    choices=[tuple(permutations(range(3),min(3,m-b))) for b in range(0,m,3)]
    for parts in product(*choices):
        a=[c for part in parts for c in part];out['market']+=1
        own=own_balanced(profile,a);out['own']+=own
        if check(profile,a) is None:
            out['fair']+=1;out['fair_own']+=own
            if not m%3:
                b=shift_budget(profile,a);out[f'fair_budget_{b}']+=1
                assert len(forward_transfers(triple_counts(profile,a)))==b
            if len(examples)<2:examples.append(a)
    return {'counts':dict(out),'examples':examples}


def obstructions():
    data=[
      ('serial12',[0,3,1,6,4,9,7,10,2,5,8,11],[0,6,1,3,7,9,4,10,2,5,8,11]),
      ('full_sigma1_balance10',[6,0,7,9,4,5,3,1,2,8],[7,4,5,6,8,9,0,3,2,1]),
      ('one_swap10',[0,5,4,9,1,8,2,3,7,6],[5,4,8,9,3,6,0,7,1,2]),
      ('frozen_class10',[9,4,2,1,6,5,7,3,8,0],[1,4,5,3,6,0,9,2,7,8]),
      ('LP_gap12',[10,0,2,1,3,5,4,6,8,7,9,11],[10,1,0,3,6,9,4,7,2,5,8,11]),
      ('six_good_repair_gap',[3,4,2,0,1,5],[2,5,0,1,3,4]),
      ('naive_gluing6',[0,3,1,4,2,5],[2,5,1,4,0,3]),
      ('Kempe_K33',[0,3,6,1,4,7,2,5,8],list(range(9)))
    ]
    return [(name,[list(range(len(a))),a,b]) for name,a,b in data]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--per-m',type=int,default=200)
    ap.add_argument('--isomorphisms',type=int,default=3000);ap.add_argument('--output',default='test_results.json');args=ap.parse_args()
    rng=random.Random(20260930);start=time.perf_counter();suites=[];records=[]
    def report():
        result={'status':'PASS_FOR_EXECUTED_CHECKS','seed':20260930,'suites':suites,'deterministic':records,
                'seconds':time.perf_counter()-start}
        (ROOT/args.output).write_text(json.dumps(result,indent=2)+'\n')
    for name,ms,solver in [('small_domain',range(16),solve_up_to_15),('general_shift_search',range(31),search_shift_balanced)]:
        count=found=none=orientations=0;hist=Counter();missing=[]
        for m in ms:
            for _ in range(args.per_m):
                a=list(range(m));b=a.copy();rng.shuffle(a);rng.shuffle(b);profile=[list(range(m)),a,b];stats={}
                answer=solver(profile,stats=stats);count+=1
                if answer is None:
                    none+=1;missing.append(profile)
                else:
                    found+=1;checked(profile,answer,stats);hist[stats['shift_budget']]+=1
                orientations+=stats.get('orientations',0)
        suites.append({'name':name,'profiles':count,'m_values':list(ms),'found':found,'restricted_infeasible':none,
                       'invalid_outputs':0,'budget_histogram':dict(hist),'orientations':orientations,'infeasible_profiles':missing})
        report();print('COMPLETE',name,count,flush=True)
    count=0;hist=Counter()
    for m in range(31):
        for _ in range(args.per_m):
            profile=random_module_profile(m,rng);stats={};answer=solve_common_modules(profile,stats)
            assert answer is not None;checked(profile,answer,stats);hist[stats['shift_budget']]+=1;count+=1
    suites.append({'name':'recognized_common_modules','profiles':count,'m_values':list(range(31)),
                   'invalid_outputs':0,'budget_histogram':dict(hist)});report();print('COMPLETE modules',count,flush=True)
    hist=Counter()
    for t in range(args.isomorphisms):
        copies=1+t%2;profile,witness=module_isomorphism(copies,rng)
        checked(profile,witness);assert shift_budget(profile,witness)==copies
        stats={};answer=search_shift_balanced(profile,copies,stats);assert answer is not None;checked(profile,answer,stats)
        assert stats['shift_budget']==copies
        lower={};assert search_shift_balanced(profile,copies-1,lower) is None
        hist[len(witness)]+=1
    suites.append({'name':'forced_shift_module_isomorphisms','profiles':args.isomorphisms,'size_histogram':dict(hist),
                   'fair_witnesses_checked':args.isomorphisms,'attaining_searches_checked':args.isomorphisms,
                   'lower_budget_rejections_checked':args.isomorphisms,'invalid_outputs':0});report();print('COMPLETE forced modules',flush=True)
    for name,profile in obstructions():
        stats={};answer=search_shift_balanced(profile,1,stats);assert answer is not None;checked(profile,answer,stats)
        records.append({'name':name,'m':len(answer),'owner':answer,'budget':stats['shift_budget']})
    # Prior connected family, independently written formula.
    for r in range(2,6):
        q=2*r;m=3*q
        s1=[g for j in range(q) for g in (3*((j-1)%q),3*((j-2)%q)+1,3*j+2)]
        s2=[0,3,1,4,7,10]+[3*j for j in range(2,q)]+[3*j+1 for j in range(4,q)]+[3*j+2 for j in range(q)]
        profile=[list(range(m)),s1,s2];stats={};answer=search_shift_balanced(profile,1,stats)
        assert answer is not None;checked(profile,answer,stats)
        records.append({'name':'connected_distance_family','r':r,'m':m,'owner':answer,'budget':stats['shift_budget']})
    base,_=counterexample13()
    for m in (13,14,15):
        pp=[s+list(range(13,m)) for s in base];data=exact_counts(pp)
        expected=206 if m==13 else 412
        assert data['counts']['fair']==expected and data['counts'].get('fair_own',0)==0
        records.append({'name':'minimal_own_obstruction','m':m,**data})
    for copies in (1,2,3,10,100):
        pp,owner=shifted_modules(copies);checked(pp,owner);assert shift_budget(pp,owner)==copies
        assert len(forward_transfers(triple_counts(pp,owner)))==copies
        records.append({'name':'large_explicit_modules','copies':copies,'m':15*copies,'budget':copies,'brute_forced':False})
    report();print('ALL CONSTRUCTION TESTS COMPLETE',flush=True)
if __name__=='__main__':main()
