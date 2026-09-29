"""Full 3^m enumeration, independent of the search generator and numba."""
from itertools import product
from collections import Counter
import json
from pathlib import Path
ROOT=Path(__file__).parent
P=[[0,7,2,5,1,11,8,3,6,9,4,10],
   [0,11,2,7,1,5,10,9,3,8,6,4],
   [5,0,2,11,7,1,4,3,9,8,6,10]]

def sd(a,P):
 for i,r in enumerate(P):
  c=[0,0,0]
  for g in r:
   c[a[g]]+=1
   if max(c)>c[i]+1:return False
 return True

def defect(a):
 c=[0,0,0];D=[]
 for t,i in enumerate(a,1):
  c[i]+=1
  if t%3 and max(c)-min(c)>1:return None
  if t%3==0 and len(set(c))!=1:D.append(t)
 return tuple(D)

def audit(P):
 m=len(P[0]);total=0;near=[]
 for a in product(range(3),repeat=m):
  if not sd(a,P):continue
  total+=1;D=defect(a)
  if D is not None:near.append((a,D))
 hist=Counter(D for a,D in near)
 delta=min(map(len,hist))
 pairs=sum(set(d).isdisjoint(e) for k,(a,d) in enumerate(near) for b,e in near[k+1:])
 res={'profile':P,'all_allocations':3**m,'SD_EF1':total,'near_SD_EF1':len(near),'minimum_defects':delta,'defect_histogram':{str(d):n for d,n in sorted(hist.items())},'disjoint_pairs':pairs,'common_subjective_prefix_lengths':[t for t in range(1,m) if set(P[0][:t])==set(P[1][:t])==set(P[2][:t])], 'near_witnesses':[{'owner':a,'defects':d} for a,d in near]}
 print('All allocations:',3**m)
 print('Subjectively SD-EF1:',total)
 print('Near-block and SD-EF1:',len(near))
 print('Defect histogram:',dict(sorted(hist.items())))
 print('Minimum defects:',delta)
 print('Minimum-defect allocations:',sum(n for d,n in hist.items() if len(d)==delta))
 print('Unordered disjoint pairs:',pairs)
 print('Common proper subjective prefix lengths:',res['common_subjective_prefix_lengths'])
 return res
if __name__=='__main__':
 import sys
 if len(sys.argv)>1:P=json.load(open(sys.argv[1]))
 ans=audit(P)
 json.dump(ans,open(ROOT/'no_disjoint_exhaustive.json','w'),indent=2)
