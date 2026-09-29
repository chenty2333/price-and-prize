from pathlib import Path
import sys,json,numpy as np
from numba import njit
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'prior/oq1_followup'))
sys.path.insert(0,str(ROOT/'prior/oq1_search'))
from core import universe,filter_sd
from exact_pair import prepare,pair_covers
from cover import find_cover,failure_rows,exact_branch_certificate
@njit(cache=True)
def search_pairs(AA):
 C,bad=prepare(AA,3);tested=0;binary=0
 for i in range(len(AA)):
  for j in range(i):
   tested+=1
   if bad[i]&bad[j]:continue
   binary+=1
   if pair_covers(C[i],C[j]):return i,j,tested,binary
 return -1,-1,tested,binary
P=json.load(open(ROOT/'min2_search_12.json'))['best_profile']
U=universe(3,12);AA=U[filter_sd(U,np.array(P))]
print('SD allocations',len(AA),flush=True)
i,j,tested,binary=search_pairs(AA)
print('Pairs examined',tested,'binary-compatible pairs',binary,'cover found',i>=0,flush=True)
res={'SD_count':len(AA),'pairs_examined':int(tested),'binary_compatible_pairs':int(binary),'pair_found':i>=0}
if i>=0:
 res['A']=AA[i].tolist();res['B']=AA[j].tolist()
 res['certificate']=exact_branch_certificate([failure_rows(AA[i],3),failure_rows(AA[j],3)])
 print('PAIR',res['A'],res['B'],flush=True)
else:
 res['general_cover']=find_cover(AA,3)
 print('GENERAL COVER',res['general_cover']['status'],len(res['general_cover'].get('subset',[])),flush=True)
json.dump(res,open(ROOT/'unrestricted_cover_12.json','w'),indent=2)
