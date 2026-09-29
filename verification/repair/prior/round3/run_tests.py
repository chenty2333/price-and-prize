"""Reproduce round-three tests. No external solver or numerical LP is used."""
from __future__ import annotations
from fractions import Fraction
from itertools import product, permutations
from pathlib import Path
import argparse
import json
import random
import time

from independent_checker import check_allocation, own_balance
from round3 import (validate_matching, residual_cycles, reset_candidates,
    all_fixed_candidates, solve_reset_matching, reset_certificate,
    signed_prefix_rows, fractional_check, repair_one_unit, encode_3cnf,
    formula_value, formula_owner, solve_encoded_family, try_cycle_repair)

SEED = 20260929


def assert_fair(profile, owner):
    error = check_allocation(profile, owner)
    assert error is None, (profile, owner, error)
    assert own_balance(profile, owner), (profile, owner)


def reset_random_suite(rng, per_q):
    result = {'profiles': 0, 'min_m': 3, 'max_m': 30,
              'full_pair_orientations_checked': 0, 'cycle_orientations_checked': 0,
              'integrally_feasible_profiles': 0, 'integrally_infeasible_profiles': 0,
              'suffix_profiles_with_exact_fractional_check': 0,
              'successful_conditional_repairs': 0, 'failures': 0}
    for q in range(1,11):
        m = 3*q
        for trial in range(per_q):
            S = [3*k+rng.randrange(3) for k in range(q)]
            SS = set(S)
            residual = [g for g in range(m) if g not in SS]
            rng.shuffle(residual)
            ss = S.copy()
            rng.shuffle(ss)
            sigma1 = [g for k in range(q) for g in (residual[2*k],residual[2*k+1],ss[k])]
            if trial % 2 == 0:
                sigma2 = rng.sample(residual,len(residual))+rng.sample(S,len(S))
            else:
                sigma2 = rng.sample(range(m),m)
            profile = [list(range(m)),sigma1,sigma2]
            validate_matching(profile,S,reset=True)
            brute_good = []
            brute_agent1_good = set()
            for owner in all_fixed_candidates(profile,S):
                result['full_pair_orientations_checked'] += 1
                c=[0,0,0]
                a1=True
                for g in sigma1:
                    c[owner[g]]+=1
                    if c[0]>c[1]+1 or c[2]>c[1]+1:
                        a1=False
                        break
                if a1:
                    brute_agent1_good.add(tuple(owner))
                if check_allocation(profile,owner) is None:
                    brute_good.append(tuple(owner))
            cycle_good=[]
            cycle_agent1_good=set()
            rows=signed_prefix_rows(profile,S)
            for choices,owner in reset_candidates(profile,S):
                result['cycle_orientations_checked'] += 1
                cycle_agent1_good.add(tuple(owner))
                signs=[1 if v==0 else -1 for v in choices]
                reduced_ok=all(sum(a*b for a,b in zip(row['d'],signs))>=row['lower'] for row in rows)
                direct_ok=check_allocation(profile,owner) is None
                assert reduced_ok == direct_ok,(profile,S,choices,rows)
                if direct_ok:
                    cycle_good.append(tuple(owner))
                repaired=repair_one_unit(profile,owner)
                if repaired is not None:
                    assert_fair(profile,repaired)
                    result['successful_conditional_repairs']+=1
            assert brute_agent1_good==cycle_agent1_good,(profile,S)
            assert sorted(brute_good)==sorted(cycle_good),(profile,S)
            answer=solve_reset_matching(profile,S)
            assert (answer is not None)==bool(brute_good)
            if answer is not None:
                assert_fair(profile,answer)
            result['integrally_feasible_profiles' if brute_good else 'integrally_infeasible_profiles']+=1
            if trial % 2 == 0:
                weights=[Fraction(0 if g in SS else 1,1 if g in SS else 2) for g in range(m)]
                assert fractional_check(profile,S,weights) is None
                result['suffix_profiles_with_exact_fractional_check']+=1
            result['profiles']+=1
    return result


def sat_suite(rng, count):
    result={'formulas':0,'assignment_anchor_cases':0,'repair_cases':0,
            'nontrivial_repairs':0,'fractional_checks':0,'min_m':None,'max_m':0,'failures':0}
    for case in range(count):
        n=rng.randrange(1,7)
        h=rng.randrange(0,11)
        clauses=[[rng.choice((-1,1))*rng.randrange(1,n+1) for _ in range(3)] for _ in range(h)]
        profile,S,cycles=encode_3cnf(clauses)
        m=len(profile[0]);assert m==12*h+6
        validate_matching(profile,S,reset=True)
        actual_components=residual_cycles(profile,S)
        assert len(actual_components)==len(cycles)
        vars=[c['variable'] for c in cycles if c['kind']=='variable']
        result['min_m']=m if result['min_m'] is None else min(result['min_m'],m)
        result['max_m']=max(result['max_m'],m)
        for vals in product((False,True),repeat=len(vars)):
            assignment=dict(zip(vars,vals))
            for anchor in (False,True):
                owner=formula_owner(profile,S,cycles,assignment,anchor)
                observed=check_allocation(profile,owner) is None
                predicted=anchor and formula_value(clauses,assignment)
                assert observed==predicted,(clauses,assignment,anchor,owner)
                result['assignment_anchor_cases']+=1
                if anchor:
                    repaired=repair_one_unit(profile,owner)
                    assert repaired is not None,(clauses,assignment)
                    assert_fair(profile,repaired)
                    assert sum((a==1)!=(b==1) for a,b in zip(owner,repaired)) in (0,2)
                    result['repair_cases']+=1
                    result['nontrivial_repairs']+=owner!=repaired
        weights=[Fraction(0) if g in S else Fraction(1,2) for g in range(m)]
        assert fractional_check(profile,S,weights) is None
        result['fractional_checks']+=1
        pp,repaired=solve_encoded_family(clauses)
        assert pp==profile
        assert_fair(pp,repaired)
        direct_answer=try_cycle_repair(profile)
        assert direct_answer is not None
        assert_fair(profile,direct_answer)
        assert len(set(S)-{g for g in range(m) if direct_answer[g]==1})<=1
        result['formulas']+=1
    return result


def small_sat_full_enumeration(rng,count):
    result={'formulas':0,'market_pair_orientations_checked':0,'failures':0,'max_m':30}
    for case in range(count):
        h=case%3
        n=rng.randrange(1,5)
        clauses=[[rng.choice((-1,1))*rng.randrange(1,n+1) for _ in range(3)] for _ in range(h)]
        profile,S,cycles=encode_3cnf(clauses)
        variables=[c['variable'] for c in cycles if c['kind']=='variable']
        sat_count=sum(formula_value(clauses,dict(zip(variables,vals)))
                      for vals in product((False,True),repeat=len(variables)))
        actual=0
        for owner in all_fixed_candidates(profile,S):
            actual+=check_allocation(profile,owner) is None
            result['market_pair_orientations_checked']+=1
        assert actual==sat_count,(clauses,actual,sat_count)
        result['formulas']+=1
    return result


def deterministic_tests():
    identity=list(range(12))
    profile=[identity,[10,0,2,1,3,5,4,6,8,7,9,11],
             [10,1,0,3,6,9,4,7,2,5,8,11]]
    S=[2,5,8,11]
    cert=reset_certificate(profile,S)
    assert len(cert['components'])==1 and len(cert['rejected'])==2 and not cert['solutions']
    initial=[0]*12
    for g in S:initial[g]=1
    for g in (10,1,4,7):initial[g]=2
    repaired=repair_one_unit(profile,initial)
    assert repaired==[0,2,1,0,2,1,0,2,1,1,2,0]
    assert_fair(profile,repaired)
    fixed=0;unrestricted=0;own=0;witnesses=[]
    for blocks in product(list(permutations((0,1,2))),repeat=4):
        owner=[i for block in blocks for i in block]
        if check_allocation(profile,owner) is None:
            unrestricted+=1
            own+=own_balance(profile,owner)
            fixed+=all(owner[g]==1 for g in S)
    assert fixed==0
    weights=[Fraction(0) if g in S else Fraction(1,2) for g in range(12)]
    assert fractional_check(profile,S,weights) is None
    # Distinct control: a fractionally infeasible fixed S, with a prefix witness.
    frac_bad_profile=[list(range(6)),list(range(6)),[2,5,0,1,3,4]]
    frac_bad_S=[2,5]
    frac_bad_weights=[Fraction(0) if g in frac_bad_S else Fraction(1,2) for g in range(6)]
    frac_bad=fractional_check(frac_bad_profile,frac_bad_S,frac_bad_weights)
    assert frac_bad['agent']==2 and frac_bad['t']==2
    # Unsatisfiable SAT controls, verified over all cycle choices.
    controls=[]
    formulas=[[[1,1,1],[-1,-1,-1]],
              [list(row) for row in product((1,-1),(2,-2),(3,-3))]]
    for formula in formulas:
        pp,ss,cycles=encode_3cnf(formula)
        cc=reset_certificate(pp,ss)
        assert not cc['solutions']
        rp,oo=solve_encoded_family(formula)
        assert_fair(rp,oo)
        controls.append({'clauses':formula,'m':len(pp[0]),'profile':pp,'S':ss,
                         'certificate':cc,'repaired_owner':oo})
    # Large all-m family test (not brute force).
    large_formula=[]
    for k in range(500):
        large_formula.extend([[k+1,k+1,k+1],[-k-1,-k-1,-k-1]])
    large_profile,large_owner=solve_encoded_family(large_formula)
    assert len(large_profile[0])==12006
    assert_fair(large_profile,large_owner)
    return {'gap':{'profile':profile,'S':S,'certificate':cert,
                   'all_market_allocations':1296,'market_balanced_fair':unrestricted,
                   'own_balanced_fair':own,'fixed_S_fair':fixed,
                   'initial_owner':initial,'repaired_owner':repaired,
                   'fractional_weights':['0' if g in S else '1/2' for g in range(12)]},
            'fractional_infeasibility_control':{'profile':frac_bad_profile,'S':frac_bad_S,'witness':frac_bad},
            'unsatisfiable_formula_controls':controls,
            'large_repair_test':{'clauses':1000,'m':12006,'passed':True}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reset-per-q',type=int,default=300)
    parser.add_argument('--formula-count',type=int,default=3000)
    parser.add_argument('--small-formula-count',type=int,default=300)
    parser.add_argument('--exchange-count',type=int,default=5000)
    parser.add_argument('--output',default='test_results.json')
    args=parser.parse_args()
    start=time.perf_counter()
    results={'seed':SEED,'status':'completed','suites':[]}
    results['suites'].append({'name':'reset_cycle_equivalence',**reset_random_suite(random.Random(SEED),args.reset_per_q)})
    print(results['suites'][-1],flush=True)
    results['suites'].append({'name':'SAT_reduction_and_one_edge_repair',**sat_suite(random.Random(SEED),args.formula_count)})
    print(results['suites'][-1],flush=True)
    results['suites'].append({'name':'small_SAT_full_pair_enumeration',**small_sat_full_enumeration(random.Random(SEED),args.small_formula_count)})
    print(results['suites'][-1],flush=True)
    results['deterministic']=deterministic_tests()
    from test_exchange import exchange_suite, old_obstructions
    results['suites'].append({'name':'general_parallel_edge_exchange',**exchange_suite(args.exchange_count)})
    results['previous_obstructions']=old_obstructions()
    results['rankings_only_cycle_repair']={'profiles':args.formula_count,'max_m':results['suites'][1]['max_m'],'failures':0,'same_formulas_as_SAT_suite':True}
    results['seconds']=time.perf_counter()-start
    Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
    print('Wrote',args.output,'in',results['seconds'],'seconds',flush=True)

if __name__=='__main__':
    main()
