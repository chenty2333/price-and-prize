from pathlib import Path
"""Exact two-allocation cover checker, using integer Farkas intervals."""
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'oq1_search'))
from core import *

@njit(cache=True)
def two_rows_separated(a,b):
 # Is some lambda in [0,1] feasible for lambda*a+(1-lambda)*b<=0?
 ln=0;ld=1;hn=1;hd=1
 for g in range(len(a)):
  d=int(a[g])-int(b[g]);bb=int(b[g])
  if d==0:
   if bb>0:return False
  elif d>0:
   num=-bb;den=d
   if num*hd<hn*den:hn=num;hd=den
  else:
   num=bb;den=-d
   if num*ld>ln*den:ln=num;ld=den
  if ln*hd>hn*ld:return False
 return True

@njit(cache=True)
def pair_covers(CA,CB):
 for a in CA:
  if a.max()<=0:continue
  for b in CB:
   if b.max()<=0:continue
   if not two_rows_separated(a,b):return False
 return True

@njit(cache=True)
def prepare(owners,n):
 D=violations(owners,n);bad=np.zeros(len(owners),np.int64)
 for z in range(len(owners)):
  for p in range(n*(n-1)):
   s=0
   for g in range(owners.shape[1]):
    s+=D[z,p,g];D[z,p,g]=s
    if s>0:bad[z]|=1<<g
 return D,bad

@njit(cache=True)
def neighbours(owners,n,swaps):
 m=owners.shape[1];N=n**m;lookup=np.full(N,-1,np.int32);codes=np.zeros(len(owners),np.int64);powers=np.array([n**g for g in range(m)],np.int64)
 for z in range(len(owners)):
  c=0
  for g in range(m):c+=int(owners[z,g])*powers[g]
  codes[z]=c;lookup[c]=z
 D,bad=prepare(owners,n);tested=0
 for z in range(len(owners)):
  a=owners[z];code=codes[z]
  for g in range(m):
   if swaps:
    for h in range(g+1,m):
     if a[g]==a[h]:continue
     cb=code+(int(a[h])-int(a[g]))*(powers[g]-powers[h]);w=lookup[cb]
     if w<0 or w>=z:continue
     if bad[z]&bad[w]:continue
     tested+=1
     if pair_covers(D[z],D[w]):return z,w,tested
   else:
    for who in range(n):
     if who==a[g]:continue
     cb=code+(who-int(a[g]))*powers[g];w=lookup[cb]
     if w<0 or w>=z:continue
     if bad[z]&bad[w]:continue
     tested+=1
     if pair_covers(D[z],D[w]):return z,w,tested
 return -1,-1,tested

if __name__=='__main__':
 import json,collections,time
 from audit_pairs import records
 out=[];C=collections.Counter();start=time.time()
 for ix,r in enumerate(records):
  P=np.array(r['profile']);u=universe(3,r['m']);aa=u[filter_sd(u,P)];rec={'index':ix,'m':r['m'],'sd_count':len(aa)}
  for k,name in [(False,'move'),(True,'swap')]:
   z,w,t=neighbours(aa,3,k);rec[name]={'found':z>=0,'binary_surviving_pairs_tested':int(t)}
   if z>=0:rec[name].update(A=aa[z].tolist(),B=aa[w].tolist())
   C[name+'_yes' if z>=0 else name+'_no']+=1
  out.append(rec)
  if (ix+1)%10==0:print(ix+1,dict(C),round(time.time()-start,2),flush=True)
  json.dump(out,open('general_local_pairs.json','w'),indent=2)
 print('FINAL',dict(C),flush=True)
