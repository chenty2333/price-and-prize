"""Ordinal two-agent fairness; logarithmic common-market value queries.
Goods: negatives=frozenset(), value is inclusion-monotone.
Chores/mixed: negatives is a common sign set, rankings order absolute
subjective magnitudes, and value is monotone in those signs.
All item/agent indices are zero based. No numerical tolerances are used.
"""
from dataclasses import dataclass
from typing import Callable, Sequence, Any

@dataclass(frozen=True)
class ReversalPath:
    initial: tuple[int,...]
    flip_time: tuple[int,...]
    steps: int

    @classmethod
    def build(cls, rankings: Sequence[Sequence[int]]) -> 'ReversalPath':
        if len(rankings)!=2: raise ValueError('Exactly two rankings required')
        m=len(rankings[0]); target=set(range(m))
        if any(len(r)!=m or set(r)!=target for r in rankings):
            raise ValueError('Each ranking must be a permutation of range(m)')
        pp=[list(r) for r in rankings]
        if m%2:
            for r in pp:r.append(m)
        N=len(pp[0]); mates=[]
        for r in pp:
            mate=[-1]*N
            for k in range(0,N,2):
                x,y=r[k:k+2];mate[x]=y;mate[y]=x
            mates.append(mate)
        red,blue=mates;own=[-1]*N;components=[]
        for seed in range(N):
            if own[seed]>=0:continue
            own[seed]=0;stack=[seed];comp=[]
            while stack:
                x=stack.pop();comp.append(x)
                for y in (red[x],blue[x]):
                    if own[y]<0:own[y]=1-own[x];stack.append(y)
                    else:assert own[y]!=own[x]
            components.append(comp)
        initial=tuple(own[:m]);when=[-1]*N;rank2=[0]*N
        for k,g in enumerate(pp[1]):rank2[g]=k
        step=0
        for comp in components:
            first=min(comp,key=lambda g:rank2[g])
            start=first if own[first]==0 else blue[first];x=start
            while True:
                y=red[x];assert own[x]==0 and own[y]==1
                step+=1;when[x]=when[y]=step
                own[x],own[y]=1,0;x=blue[y]
                if x==start:break
        assert step==(m+1)//2 and tuple(own[:m])==tuple(1-i for i in initial)
        return cls(initial,tuple(when[:m]),step)

    def owner(self,t:int,negatives:frozenset[int]=frozenset())->tuple[int,...]:
        if not 0<=t<=self.steps:raise ValueError('Invalid path index')
        return tuple(i ^ int(w<=t) ^ int(g in negatives)
                     for g,(i,w) in enumerate(zip(self.initial,self.flip_time)))

    def bundles(self,t:int)->tuple[frozenset[int],frozenset[int]]:
        a=self.owner(t)
        return tuple(frozenset(g for g,i in enumerate(a) if i==j) for j in range(2))

def solve(rankings:Sequence[Sequence[int]],
          value:Callable[[frozenset[int]],Any],
          negatives:frozenset[int]=frozenset())->dict:
    """Return an allocation and a specific market repair certificate.
    <= 3+2*ceil(log2(ceil(m/2))) value queries for m>=1.
    The implicit ranking-dependent path has ceil(m/2)+1 states.
    """
    path=ReversalPath.build(rankings);m=len(path.initial);C=frozenset(negatives)
    if not C<=set(range(m)):raise ValueError('Unknown negative item')
    calls=0
    def F(S):
        nonlocal calls
        calls+=1
        return value(S ^ C)
    def result(t,repair=None):
        return {'owner':path.owner(t,C),'path_index':t,'market_repair':repair,
                'queries':calls,'steps':path.steps}
    if m==0:return result(0)
    lo=0;hi=path.steps;B=path.bundles(0);lv=(F(B[0]),F(B[1]))
    if lv[0]==lv[1]:return result(0)
    hv=lv[::-1]
    while hi-lo>1:
        mid=(lo+hi)//2;B=path.bundles(mid);mv=(F(B[0]),F(B[1]))
        if mv[0]==mv[1]:return result(mid)
        if (mv[0]>mv[1])==(lv[0]>lv[1]):lo=mid;lv=mv
        else:hi=mid;hv=mv
    L,R=path.bundles(lo),path.bundles(hi)
    i=0 if lv[0]>lv[1] else 1;j=1-i
    out=L[i]-R[i]
    assert len(out)==1, 'Opposite signs require an item to leave the high benefit bundle'
    g=next(iter(out))
    if g not in C:choose_left=F(L[i]-{g})<=lv[j]
    else:choose_left=F(L[j]|{g})>=lv[i]
    t=lo if choose_left else hi
    high=i if choose_left else j;low=1-high
    repair={'envier':low,'remove_from':low if g in C else high,'item':g}
    return result(t,repair)
