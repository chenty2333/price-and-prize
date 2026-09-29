import json,sys,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'prior/oq1_monotone_pair'))
from pair_selection import subjective_scores,inspect
base=json.load(open(ROOT/'min2_search_12.json'))['best_profile']
z=np.load(ROOT/'candidates_12.npz');AA,D,F=z['A'],z['D'],z['F']
tried=0;zero=0;best=None
for i in range(3):
 for x,y in itertools.combinations(range(12),2):
  P=np.array(base,dtype=np.int16);P[i,x],P[i,y]=P[i,y],P[i,x];tried+=1
  if np.any(subjective_scores(F,P)>=0):continue
  zero+=1
  shared=[t for t in range(1,12) if set(P[0,:t])==set(P[1,:t])==set(P[2,:t])]
  if not shared:
   r=inspect(P.tolist(),AA,D)
   best={'mutation':[i,x,y],'profile':P.tolist(),'analysis':r}
   break
 if best:break
print('Mutations checked',tried,'preserve min>=2',zero,'no common prefix found',bool(best))
if best:
 print(best['profile']);print(best['analysis']['defect_histogram'])
 json.dump(best['profile'],open(ROOT/'interleaved_no_disjoint_profile.json','w'),indent=2)
json.dump({'tried':tried,'zero_objective':zero,'answer':best},open(ROOT/'refine_no_disjoint.json','w'),indent=2)
