from ordinal_market import ReversalPath,solve
from exact1_circle import exact1,common_market_cardinal
from itertools import permutations
from collections import Counter
from math import isqrt
import random,json
from pathlib import Path
ROOT=Path(__file__).parent

def subjective(a,P):
 for i,r in enumerate(P):
  c=[0,0]
  for x in r:
   c[a[x]]+=1
   if c[1-i]>c[i]+1:return False
 return True

def ef1(a,vals,C=frozenset()):
 B=[frozenset(g for g,i in enumerate(a) if i==j) for j in range(2)]
 for i in range(2):
  f=vals[i];j=1-i
  if f(B[i])>=f(B[j]):continue
  if any(f(B[i])>=f(B[j]-{g}) for g in B[j]-C):continue
  if any(f(B[i]-{g})>=f(B[j]) for g in B[i]&C):continue
  return False
 return True

def generator(rng,m,kind):
 w=[rng.randrange(1,21) for _ in range(m)]
 if kind==0:return lambda S:sum(w[x] for x in S)
 if kind==1:
  W=[[rng.randrange(21) for _ in range(m)] for _ in range(4)]
  return lambda S:max(sum(z[x] for x in S) for z in W)
 if kind==2:
  cap=rng.randrange(1,sum(w)+1)
  return lambda S:min(cap,sum(w[x] for x in S))
 if kind==3:
  z=[sum(1<<j for j in rng.sample(range(24),rng.randrange(1,6))) for _ in range(m)]
  def f(S):
   x=0
   for g in S:x|=z[g]
   return x.bit_count()
  return f
 if kind==4:return lambda S:isqrt(100*sum(w[x] for x in S))
 if kind==5:
  edges=[(rng.randrange(m),rng.randrange(m),rng.randrange(1,20)) for _ in range(m)]
  return lambda S:sum(w[x] for x in S)**2+sum(b for x,y,b in edges if x in S and y in S)
 assert m<=10
 T=[rng.randrange(201) for _ in range(1<<m)];T[0]=0
 for g in range(m):
  for s in range(1<<m):
   if s>>g&1:T[s]=max(T[s],T[s^(1<<g)])
 assert all(T[s]<=T[s|(1<<g)] for s in range(1<<m) for g in range(m))
 return lambda S:T[sum(1<<x for x in S)]

if __name__=='__main__':
 stats=Counter();maxqueries=0
 for m in range(1,9):
  for p in permutations(range(m)):
   P=[list(range(m)),p];path=ReversalPath.build(P);stats['exhaustive_profiles']+=1
   assert path.owner(path.steps)==tuple(1-x for x in path.owner(0))
   for t in range(path.steps+1):
    a=path.owner(t);assert subjective(a,P);stats['exhaustive_path_states']+=1
    if t:
     b=path.owner(t-1);changed=[g for g in range(m) if a[g]!=b[g]]
     assert len(changed) in (1,2)
     if len(changed)==2:assert a[changed[0]]!=a[changed[1]]
 print('Exhaustive compressed-path profiles:',stats['exhaustive_profiles'],flush=True)
 print('Exhaustive path states:',stats['exhaustive_path_states'],flush=True)
 rng=random.Random(92847438)
 for mode in ['goods','chores','mixed']:
  for t in range(2000):
   kind=t%7;m=rng.randrange(1,11 if kind==6 else 65)
   P=[rng.sample(range(m),m) for _ in range(2)]
   C=frozenset() if mode=='goods' else (frozenset(range(m)) if mode=='chores' else frozenset(g for g in range(m) if rng.randrange(2)))
   f=generator(rng,m,kind);offset=f(C)
   value=lambda S:f(S^C)-offset
   calls=[0]
   def counted(S):calls[0]+=1;return value(S)
   ans=solve(P,counted,C);a=ans['owner'];stats[mode+'_instances']+=1
   assert ans['queries']==calls[0]
   s=(m+1)//2;bound=3+2*(s-1).bit_length();assert calls[0]<=bound
   maxqueries=max(maxqueries,calls[0]);stats['solver_queries']+=calls[0]
   assert subjective(tuple(x^int(g in C) for g,x in enumerate(a)),P)
   assert ef1(a,[value,value],C)
   R=ans['market_repair'];B=[frozenset(g for g,i in enumerate(a) if i==j) for j in range(2)]
   if R is not None:
    i,j,g=R['envier'],R['remove_from'],R['item'];assert g in B[j]
    assert (j==i)==(g in C)
    assert value(B[i]-{g})>=value(B[1-i]) if j==i else value(B[i])>=value(B[1-i]-{g})
    stats['repair_certificates']+=1
   path=ReversalPath.build(P)
   ws=[]
   for i in range(2):
    z=[0]*m
    for k,g in enumerate(P[i]):z[g]=m-k
    ws.append(z)
   subjective_vals=[lambda S,z=z:sum((-1 if g in C else 1)*z[g] for g in S) for z in ws]
   for k in range(path.steps+1):
    o=path.owner(k,C);stats['random_path_states']+=1
    assert subjective(path.owner(k),P)
    assert ef1(o,subjective_vals,C)
  print(mode,'completed',2000,flush=True)
 for t in range(2000):
  kind=t%7;m=rng.randrange(1,11 if kind==6 else 33)
  f=generator(rng,m,kind);g=generator(rng,m,(kind+1)%6);h=generator(rng,m,(kind+3)%6)
  ans=exact1(m,f,g);X,Y=ans['bundles'];a=tuple(0 if x in X else 1 for x in range(m))
  assert X.isdisjoint(Y) and X|Y==set(range(m))
  assert ef1(a,[f,f]) and ef1(a,[g,g])
  out=common_market_cardinal(m,f,h,g)
  assert ef1(out['owner'],[f,h]) and ef1(out['owner'],[g,g])
  stats['exact1_and_cardinal_instances']+=1
  stats['exact1_queries']+=ans['queries']
 print('Exact1/common-cardinal completed:',stats['exact1_and_cardinal_instances'],flush=True)
 stats['max_solver_queries']=maxqueries
 print(json.dumps(dict(stats),indent=2),flush=True)
 json.dump(dict(stats),open(ROOT/'test_counts.json','w'),indent=2)
