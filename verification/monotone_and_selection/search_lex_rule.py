"""Targeted search against lexicographically-earliest defect-set selection.
Not a random existence benchmark. m=10, complete 3000-word near-block universe.
The objective suppresses the empty, {3}, and {9} defect sets, leaving the
possibility of an extendable {6} and a nonextendable lex-earliest {3,6}.
"""
import random,json,time,math
import numpy as np
from pair_selection import candidates,subjective_scores,inspect

rng=random.Random(491734)
AA,DD=candidates(10)
forbidden=AA[np.isin(DD,[0,1,4])]
base=[[3,6,0,8,5,2,1,4,9,7],[3,6,8,5,4,0,7,9,2,1],[0,3,6,8,4,2,9,1,5,7]]
profiles=json.load(open('small_hard_profiles.json'))
new=[[7,4,2,5,0,8,9,6,3,1],[4,2,0,3,7,5,9,1,8,6],[4,0,3,2,7,5,9,1,8,6]]
seeds=[P for P in profiles if len(P[0])==10 and P!=new]
seen=set();nprop=0;best=100000;answer=None
start=time.monotonic()
for restart in range(30):
    P=np.array(seeds[restart%len(seeds)] if restart else base,dtype=np.int16)
    val=int(np.sum(subjective_scores(forbidden,P)>=0))
    for step in range(4000):
        nprop+=1;Q=P.copy();i=rng.randrange(3);a,b=rng.sample(range(10),2)
        Q[i,a],Q[i,b]=Q[i,b],Q[i,a]
        key=tuple(Q.flat);seen.add(key)
        q=int(np.sum(subjective_scores(forbidden,Q)>=0))
        if q<best:
            best=q;print('best forbidden count',best,'proposal',nprop,flush=True)
        if q==0:
            ans=inspect(Q.tolist(),AA,DD)
            if ans['has_pair'] and ans['rules']['lex_earliest_defect_tuple']['nonextendable_count']:
                answer=ans;break
        temp=.15+2.5*(1-(step%1000)/1000)
        if q<=val or rng.random()<math.exp((val-q)/temp):P=Q;val=q
    if answer is not None:break
log={'proposals':nprop,'distinct_proposals':len(seen),'best_forbidden_count':best,'answer':answer}
json.dump(log,open('search_lex_rule.json','w'),indent=2)
print('Proposals:',nprop,'distinct:',len(seen),'best:',best,'failure_found:',answer is not None)
if answer:print(json.dumps(answer,indent=2))
