"""Independent Python audit using the ORIGINAL round-three encoder.

Exhaustively audits every canonical formula through two clauses.  The default
also checks 1,000 signed three-clause formulas against all truth/anchor choices.
C++ audit_sat.cpp exhaustively covers all canonical three-clause formulas and
all raw pair orientations, with a separately written constructor/checker.
"""
from __future__ import annotations
import argparse,itertools,json,random,sys,time
from pathlib import Path
from independent_checker import check,validate,own_balanced
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'prior'/'round3'))
import round3


def partitions(n):
    a=[0]*n
    def rec(t,high):
        if t==n:
            yield tuple(a);return
        for v in range(high+2):
            a[t]=v;yield from rec(t+1,max(v,high))
    yield from rec(0,-1)


def formula(pattern,signs):
    vals=[-(v+1)if(signs>>t)&1 else v+1 for t,v in enumerate(pattern)]
    return [vals[t:t+3]for t in range(0,len(vals),3)]


def run(samples3=1000):
    start=time.perf_counter();records=[]
    for k in range(3):
        patterns=formulas=raw=survivors=comparisons=unsat=completions=0
        for pattern in partitions(3*k):
            patterns+=1;p0,S,cycles=round3.encode_3cnf(formula(pattern,0));validate(p0)
            alive=[]
            for a in round3.all_fixed_candidates(p0,S):
                raw+=1
                if check(p0,a,agents=(1,))is None:alive.append(a)
            survivors+=len(alive)
            metadata=[]
            for a in alive:
                assignment={c['variable']:a[c['a'][0]]==2 for c in cycles[:-1]}
                anchor=a[cycles[-1]['a'][0]]==2
                # Check ALL cycle goods, not only a representative.
                reconstructed=round3.formula_owner(p0,S,cycles,assignment,anchor)
                assert reconstructed==a
                metadata.append((a,assignment,anchor))
            for signs in range(1<<(3*k)):
                f=formula(pattern,signs);p,SS,cc=round3.encode_3cnf(f);validate(p)
                assert p[:2]==p0[:2]and SS==S and cc==cycles
                sols=0
                for a,assignment,anchor in metadata:
                    expected=anchor and all(any(assignment[abs(x)]==(x>0)for x in clause)for clause in f)
                    actual=check(p,a)is None
                    assert actual==expected,(f,assignment,anchor,p,a)
                    comparisons+=1;sols+=actual
                formulas+=1;unsat+=sols==0;completions+=sols
        records.append({'clauses':k,'variable_patterns':patterns,'canonical_formulas':formulas,
                        'raw_pair_orientations':raw,'agent1_survivors':survivors,
                        'signed_candidate_comparisons':comparisons,'unsatisfiable_formulas':unsat,
                        'fair_completions_total':completions})
    rng=random.Random(20260929);patterns=list(partitions(9));checks=repairs=0
    for _ in range(samples3):
        pattern=rng.choice(patterns);signs=rng.randrange(512);f=formula(pattern,signs)
        p,S,cycles=round3.encode_3cnf(f);validate(p)
        variables=[c['variable']for c in cycles[:-1]]
        for values in itertools.product((False,True),repeat=len(variables)):
            assignment=dict(zip(variables,values))
            sat=all(any(assignment[abs(l)]==(l>0)for l in clause)for clause in f)
            for anchor in (False,True):
                a=round3.formula_owner(p,S,cycles,assignment,anchor)
                assert (check(p,a)is None)==(anchor and sat)
                checks+=1
                if anchor:
                    out=round3.repair_one_unit(p,a)
                    assert out is not None and check(p,out)is None and own_balanced(p,out)
                    assert sum(out[g]!=1 for g in S)<=1
                    repairs+=1
    result={'status':'PASS','original_round3_encoder':True,'exhaustive_records':records,
            'sampled_three_clause_formulas':samples3,'three_clause_candidate_checks':checks,
            'three_clause_repairs':repairs,'seconds':time.perf_counter()-start}
    cpp=json.loads((ROOT/'audit_sat_results.json').read_text())
    for py,cc in zip(records,cpp['records']):
        for key,value in py.items():assert cc[key]==value,(key,value,cc[key])
    result['all_exhaustive_counts_match_cpp']=True
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--samples3',type=int,default=1000)
    ap.add_argument('--output',default='audit_reference_results.json');a=ap.parse_args()
    out=run(a.samples3);Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
