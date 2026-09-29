"""Targeted finite search: minimize all near-block SD witnesses with <=1 defect.
Every objective evaluation checks its entire fixed candidate universe.
"""
import json,sys,random,math,time,hashlib
from pathlib import Path
import numpy as np
from numba import njit
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'prior/oq1_monotone_pair'))
from pair_selection import inspect

@njit(cache=True)
def agent_mask(AA,rank,i):
 out=np.ones(len(AA),np.bool_)
 for k in range(len(AA)):
  c0=c1=c2=0
  for g in rank:
   q=AA[k,g]
   if q==0:c0+=1
   elif q==1:c1+=1
   else:c2+=1
   own=c0 if i==0 else (c1 if i==1 else c2)
   if max(c0,c1,c2)>own+1:out[k]=False;break
 return out

@njit(cache=True)
def trial_cost(AA,rank,i,other):
 total=0
 for k in range(len(AA)):
  if not other[k]:continue
  c0=c1=c2=0;ok=True
  for g in rank:
   q=AA[k,g]
   if q==0:c0+=1
   elif q==1:c1+=1
   else:c2+=1
   own=c0 if i==0 else (c1 if i==1 else c2)
   if max(c0,c1,c2)>own+1:ok=False;break
  total+=ok
 return total

def run(m,restarts,steps):
 arr=np.load(ROOT/f'candidates_{m}.npz');F=arr['F'];AA,DD=arr['A'],arr['D']
 corpus=json.load(open(ROOT/'prior/oq1_monotone_pair/small_hard_profiles.json'))
 seeds=[P for P in corpus if len(P[0])==m]
 rng=random.Random(392846+m);seen=set();props=0;best=len(F);BP=None;found=[]
 for restart in range(restarts):
  if restart%3==2 and BP is not None:
   P=np.array(BP,dtype=np.int16)
   for _ in range(3):
    i=rng.randrange(3);x,y=rng.sample(range(m),2);P[i,x],P[i,y]=P[i,y],P[i,x]
  elif seeds:P=np.array(seeds[restart%len(seeds)],dtype=np.int16)
  else:P=np.array([rng.sample(range(m),m) for _ in range(3)],dtype=np.int16)
  masks=np.array([agent_mask(F,P[i],i) for i in range(3)])
  val=int(np.sum(np.all(masks,axis=0)))
  for step in range(steps):
   i=rng.randrange(3);x,y=sorted(rng.sample(range(m),2));q=P[i].copy()
   if rng.random()<.85:q[x],q[y]=q[y],q[x]
   else:q[x:y+1]=q[x:y+1][::-1]
   other=masks[(i+1)%3]&masks[(i+2)%3]
   cost=int(trial_cost(F,q,i,other));props+=1
   old=P[i].copy();P[i]=q;seen.add(P.tobytes());P[i]=old
   if cost<best:
    best=cost;BP=P.copy();BP[i]=q;BP=BP.tolist()
    print('m',m,'best',best,'proposal',props,flush=True)
   if cost==0:
    P[i]=q;r=inspect(P.tolist(),AA,DD);found.append(r)
    print('FOUND m',m,'min',r['minimum_defects'],'pair',r['has_pair'],flush=True)
    break
   temp=.35+max(1.,best/8)*(1-(step%1000)/1000)
   if cost<=val or rng.random()<math.exp((val-cost)/temp):
    P[i]=q;val=cost;masks[i]=agent_mask(F,q,i)
  if found:break
  if (restart+1)%10==0:print('completed',m,restart+1,'restarts',flush=True)
 rec={'m':m,'seed':392846+m,'restarts_started':restart+1,'steps_per_restart':steps,'proposals':props,'distinct_proposals':len(seen),'forbidden_universe':len(F),'near_block_universe':len(AA),'minimum_forbidden_count':best,'best_profile':BP,'found':found}
 if BP is not None:rec['best_analysis']=inspect(BP,AA,DD)
 json.dump(rec,open(ROOT/f'min2_search_{m}.json','w'),indent=2)
 print('RESULT',m,props,len(seen),best,len(found),flush=True)

if __name__=='__main__':
 m=int(sys.argv[1]);run(m,int(sys.argv[2]) if len(sys.argv)>2 else 30,4000)
