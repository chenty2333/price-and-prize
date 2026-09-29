"""MILP separation plus independently checkable exact Farkas cover certificates."""
from core import *
from scipy.optimize import milp, Bounds, LinearConstraint, linprog
from scipy.sparse import coo_array
from fractions import Fraction
import json, time

def failure_rows(owner,n):
    """Non-dominated, possibly positive rows in market-gap (delta) coordinates."""
    C=np.cumsum(violations(np.array([owner],dtype=np.uint8),n)[0],axis=1)
    C=np.unique(C,axis=0)
    C=C[np.max(C,axis=1)>0]
    C=np.array([x for k,x in enumerate(C) if not any(
        np.all(x<=y) and np.any(x<y) for l,y in enumerate(C) if l!=k)],dtype=np.int64)
    if C.size==0:return np.empty((0,len(owner)),dtype=np.int64)
    return C

def master(rowsets, m, time_limit=20):
    """Maximize a common strict-violation margin over normalized market gaps."""
    h=m # Conservative epsilon cap; M=2m is valid for all balanced allocations.
    M=2*m
    count=sum(len(rs) for rs in rowsets)
    nv=m+1+count
    c=np.zeros(nv);c[m]=-1
    lower=np.zeros(nv);upper=np.ones(nv);upper[m]=h
    integ=np.zeros(nv,dtype=np.uint8);integ[m+1:]=1
    rr=[];cc=[];vv=[];lbs=[];ubs=[]
    def add(d,lo,hi):
        r=len(lbs)
        for j,v in d.items():
            if v:rr.append(r);cc.append(j);vv.append(v)
        lbs.append(lo);ubs.append(hi)
    add({g:1 for g in range(m)},1,1)
    offset=m+1
    for rs in rowsets:
        if len(rs)==0:
            return {'epsilon':0.,'delta':np.eye(1,m)[0], 'status':0,'empty_rowset':True}
        add({offset+p:1 for p in range(len(rs))},1,1)
        for p,row in enumerate(rs):
            d={g:int(row[g]) for g in range(m)}
            d[m]=-1;d[offset+p]=-M
            add(d,-M,np.inf)
        offset+=len(rs)
    A=coo_array((np.array(vv,dtype=float),(np.array(rr,dtype=np.int32),np.array(cc,dtype=np.int32))),shape=(len(lbs),nv)).tocsc()
    result=milp(c,integrality=integ,bounds=Bounds(lower,upper),
        constraints=LinearConstraint(A,np.array(lbs),np.array(ubs)),
        options={'time_limit':time_limit,'mip_rel_gap':0.0})
    return {'epsilon':None if result.x is None else float(result.x[m]),
            'delta':None if result.x is None else result.x[:m],
            'status':int(result.status),'message':result.message,
            'mip_gap':getattr(result,'mip_gap',None)}

def exact_branch_certificate(rowsets,max_nodes=100000):
    """Prove that no delta >= 0, sum(delta)=1 makes one row from each set >0.

    A pruned branch records rational lambda >= 0, sum lambda=1 with lambda*C<=0.
    These certificates are verified using Fraction, independently of LP tolerance.
    """
    m=len(rowsets[0][0]) if len(rowsets[0]) else 0
    leaves=[];nodes=0
    def rec(chosen,indices,depth):
        nonlocal nodes
        nodes+=1
        if nodes>max_nodes:raise RuntimeError('Exact certificate node limit')
        if chosen:
            C=np.array(chosen,dtype=float)
            objective=np.r_[np.zeros(m),-1.]
            Aub=np.column_stack([-C,np.ones(len(chosen))])
            sol=linprog(objective,A_ub=Aub,b_ub=np.zeros(len(chosen)),
                        A_eq=np.array([np.r_[np.ones(m),0.]]),b_eq=[1.],
                        bounds=[(0,None)]*m+[(None,None)],method='highs')
            assert sol.status==0, sol.message
            eps=sol.x[-1]
            if eps<=1e-8:
                weights=[Fraction(float(-x)).limit_denominator(1_000_000)
                         for x in sol.ineqlin.marginals]
                assert all(w>=0 for w in weights)
                assert sum(weights)==1
                assert all(sum(weights[k]*int(chosen[k][g]) for k in range(len(chosen)))<=0 for g in range(m)), (chosen,weights,eps)
                leaves.append({'path':indices.copy(),'weights':[str(w) for w in weights]})
                return True
        if depth==len(rowsets):
            return False
        for j,row in enumerate(rowsets[depth]):
            if not rec(chosen+[row],indices+[j],depth+1):return False
        return True
    ok=rec([],[],0)
    if not ok:return None
    cert={'rowsets':[rs.tolist() for rs in rowsets], 'leaves':leaves,'nodes':nodes}
    assert verify_exact_certificate(cert)
    return cert

def verify_exact_certificate(cert):
    """Pure Python exact checker: requires neither NumPy nor an optimization solver."""
    sets=cert['rowsets'];paths=[]
    for leaf in cert['leaves']:
        path=leaf['path'];weights=list(map(Fraction,leaf['weights']))
        if len(path)!=len(weights) or not all(w>=0 for w in weights) or sum(weights)!=1:return False
        rows=[sets[k][j] for k,j in enumerate(path)]
        if rows and not all(sum(w*row[g] for w,row in zip(weights,rows))<=0 for g in range(len(rows[0]))):return False
        paths.append(tuple(path))
    paths=set(paths)
    def complete(prefix):
        if prefix in paths:return True
        k=len(prefix)
        if k==len(sets):return False
        return all(complete(prefix+(j,)) for j in range(len(sets[k])))
    return complete(())

def find_cover(sd,n,limit=25):
    m=sd.shape[1]
    # Start at a moderately flat, strictly decreasing market vector.
    v=np.linspace(1,.1,m)
    subset=[];rowsets=[];steps=[]
    for iteration in range(limit):
        margins=market_margins(sd,v,n)
        best=int(np.argmin(margins))
        if margins[best]>1e-7:
            return {'status':'candidate-counterexample','v':v.tolist(),
                    'minimum_violation':float(margins[best])}
        owner=sd[best]
        if any(np.array_equal(owner,a) for a in subset):
            raise RuntimeError('MILP oracle repeated an allocation')
        subset.append(owner.copy());rowsets.append(failure_rows(owner,n))
        res=master(rowsets,m)
        steps.append({k:x for k,x in res.items() if k!='delta'})
        if res['status']!=0:
            return {'status':'inconclusive-MILP','steps':steps,'subset':[a.tolist() for a in subset]}
        if res['epsilon']<=1e-7:
            cert=exact_branch_certificate(rowsets)
            if cert is None:raise RuntimeError('Numeric claim failed exact cover check')
            return {'status':'exact-cover','subset':[a.tolist() for a in subset],
                    'steps':steps,'certificate':cert}
        v=np.cumsum(res['delta'][::-1])[::-1]
    return {'status':'iteration-limit','steps':steps,'subset':[a.tolist() for a in subset]}

if __name__=='__main__':
    import sys
    infile=sys.argv[1]
    raw=json.load(open(infile))
    n,m=raw['n'],raw['m'];u=universe(n,m)
    records=[];start=time.time()
    for k,p in enumerate(raw['hard_profiles']):
        p=np.array(p,dtype=np.int64)
        inds=filter_sd(u,p);sd=u[inds]
        assert not any(market_block(a,n) for a in sd)
        result=find_cover(sd,n)
        rec={'n':n,'m':m,'profile':p.tolist(),'sd_count':len(sd),'block_count':0,**result}
        records.append(rec)
        print('COVER',k,'sd',len(sd),rec['status'],'k',len(rec.get('subset',[])),
              'exact leaves',len(rec.get('certificate',{}).get('leaves',[])),
              'elapsed',round(time.time()-start,2),flush=True)
        with open(f'covers_{n}_{m}.json','w') as f:json.dump(records,f,indent=2)
        if rec['status']=='candidate-counterexample':break
