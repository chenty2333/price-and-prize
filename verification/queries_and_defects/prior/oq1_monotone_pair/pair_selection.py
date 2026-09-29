"""Exhaustive integer analysis of pair-selection rules, n=3.
Market-near-block owner words are generated independently of the profile.
Every such word is checked against every subjective prefix.
Thus no solver failure/timeout is interpreted as infeasibility.
"""
from collections import Counter
import itertools,json,os,time
import numpy as np
from numba import njit


def candidates(m):
    out=[];Ds=[]
    a=[0]*m;count=[0]*3
    def visit(k,d):
        if k==m:
            if max(count)-min(count)<=1:
                out.append(tuple(a));Ds.append(d)
            return
        t=k+1
        for i in range(3):
            a[k]=i;count[i]+=1
            delta=max(count)-min(count)
            if t%3:
                if delta<=1:visit(t,d)
            elif delta==0:visit(t,d)
            elif delta==2 and sorted(count)==[t//3-1,t//3,t//3+1]:
                visit(t,d|(1<<(t//3-1)))
            count[i]-=1
    visit(0,0)
    return np.array(out,dtype=np.int8),np.array(Ds,dtype=np.int16)

@njit(cache=True)
def subjective_scores(AA,P):
    out=np.full(len(AA),-1,dtype=np.int16)
    for k in range(len(AA)):
        a=AA[k];score=0;ok=True
        for i in range(3):
            c=np.zeros(3,dtype=np.int16)
            for g in P[i]:
                c[a[g]]+=1
                rival=max(c[(i+1)%3],c[(i+2)%3])
                slack=1+c[i]-rival
                if slack<0:ok=False;break
                score+=slack
            if not ok:break
        if ok:out[k]=score
    return out


def defect_tuple(d,m):return tuple(3*(q+1) for q in range(m//3) if d&(1<<q))

def inspect(P,AA,DD):
    m=len(P[0]);scores=subjective_scores(AA,np.array(P,dtype=np.int16));idx=np.where(scores>=0)[0]
    dfam=sorted(set(map(int,DD[idx])));safe={d:any(d&e==0 for e in dfam) for d in dfam}
    mind=min(d.bit_count() for d in dfam)
    minsets=[d for d in dfam if d.bit_count()==mind]
    lex=min(dfam,key=lambda d:defect_tuple(d,m))
    reverse_lex=min(dfam,key=lambda d:tuple(int(bool(d&(1<<q))) for q in range(m//3)))
    maxslack=int(scores[idx].max());maxids=[int(k) for k in idx if scores[k]==maxslack]
    # All tied minimizers tested; deterministic tie-break uses lexicographic owner vector.
    selected={
      'minimum_defects_all_ties': [int(k) for k in idx if int(DD[k]).bit_count()==mind],
      'lex_earliest_defect_tuple': [int(k) for k in idx if DD[k]==lex],
      'lex_avoid_early_defects': [int(k) for k in idx if DD[k]==reverse_lex],
      'maximum_total_subjective_slack_all_ties':maxids,
      'minimum_defects_then_maximum_slack':[],
    }
    mm=max(int(scores[k]) for k in selected['minimum_defects_all_ties'])
    selected['minimum_defects_then_maximum_slack']=[k for k in selected['minimum_defects_all_ties'] if scores[k]==mm]
    res={'m':m,'profile':P,'near_sd_count':len(idx),
         'defect_histogram':{str(defect_tuple(d,m)):int(np.sum(DD[idx]==d)) for d in dfam},
         'minimum_defects':mind,'has_pair':any(safe.values()),'has_pair_both_minimum':any(d&e==0 for d in minsets for e in minsets),'market_block_count':int(np.sum(DD[idx]==0)),
         'rules':{},'max_slack':maxslack}
    for name,ks in selected.items():
        bad=[k for k in ks if not safe[int(DD[k])]]
        rec={'selected_count':len(ks),'nonextendable_count':len(bad),'lex_owner_extendable':safe[int(DD[ks[0]])]}
        if bad:
            k=bad[0];rec['failure_owner']=AA[k].tolist();rec['failure_defects']=defect_tuple(int(DD[k]),m);rec['failure_score']=int(scores[k])
        res['rules'][name]=rec
    if res['has_pair']:
        da,db=next((d,e) for d in dfam for e in dfam if d&e==0)
        res['pair']=[AA[next(k for k in idx if DD[k]==d)].tolist() for d in (da,db)]
    return res


def main():
    profiles=json.load(open('small_hard_profiles.json'))
    assert len({tuple(map(tuple,P)) for P in profiles})==146
    cache={};results=[];t0=time.time()
    for m in sorted(set(len(P[0]) for P in profiles)):
        AA,DD=candidates(m);cache[m]=(AA,DD)
        print('Near-block candidate universe:',m,len(AA),flush=True)
        for P in profiles:
            if len(P[0])!=m:continue
            r=inspect(P,AA,DD);assert r['market_block_count']==0
            results.append(r)
        print('Completed size',m,flush=True)
    counts={'profiles':len(results),'sizes':dict(Counter(r['m'] for r in results)),
            'minimum_defect_histogram':dict(Counter(r['minimum_defects'] for r in results)),
            'pair_found':sum(r['has_pair'] for r in results),'both_minimum_failures':sum(not r['has_pair_both_minimum'] for r in results),'rules':{}}
    for name in results[0]['rules']:
        failures=[r for r in results if r['rules'][name]['nonextendable_count']]
        counts['rules'][name]={'profiles_with_bad_tie':len(failures),
                             'profiles_deterministic_lex_owner_fails':sum(not r['rules'][name]['lex_owner_extendable'] for r in results)}
        if failures:
            smallest=min(failures,key=lambda r:r['m'])
            counts['rules'][name]['smallest_failure']=smallest
    json.dump(results,open('pair_selection_all.json','w'),indent=2)
    json.dump(counts,open('pair_selection_counts.json','w'),indent=2)
    for name,info in counts['rules'].items():print(name,{k:v for k,v in info.items() if k!='smallest_failure'},flush=True)
    print('Profiles by size:',counts['sizes']);print('Minimum defect counts:',counts['minimum_defect_histogram'])
    print('Disjoint pairs:',counts['pair_found']);print('No pair with both witnesses minimum-defect:',counts['both_minimum_failures']);print('Exact subjective witness checks:',sum(r['near_sd_count'] for r in results))

if __name__=='__main__':main()
