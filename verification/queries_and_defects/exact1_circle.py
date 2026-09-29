"""Two-function Exact1 by the moving-knives algorithm of Goldberg et al.,
Consensus Halving for Sets of Items, Appendix A, Algorithm A.1/Theorem A.2.
Sets are represented as bit masks; value oracles take frozensets.
"""
from typing import Callable,Any

def exact1(m:int,f:Callable,g:Callable)->dict:
    if m<0:raise ValueError('Negative item count')
    if m==0:return {'bundles':(frozenset(),frozenset()),'queries':0,'states':1}
    if m==1:return {'bundles':(frozenset({0}),frozenset()),'queries':0,'states':1}
    calls=0;full=(1<<m)-1;cache=[{},{}]
    def val(i,S):
        nonlocal calls
        if S not in cache[i]:
            calls+=1;cache[i][S]=(f if i==0 else g)(frozenset(x for x in range(m) if S>>x&1))
        return cache[i][S]
    def pieces(a,b):return ((1<<b)-1)^((1<<a)-1)
    def good(i,a,b):
        S=pieces(a,b);T=full^S;s=val(i,S);t=val(i,T)
        if s<t:
            end=set()
            if b<m:end.add(b)
            elif a>0:end.add(0)
            if a>0:end.add(a-1)
            elif b<m:end.add(m-1)
            assert end
            return any(s>=val(i,T^(1<<x)) for x in end)
        if t<s:
            assert a<b
            return any(t>=val(i,S^(1<<x)) for x in {a,b-1})
        return True
    k=next(k for k in range(1,m) if good(0,0,k))
    a,b=0,k;states=0
    while True:
        states+=1
        assert good(0,a,b)
        if good(1,a,b):
            S=pieces(a,b)
            return {'bundles':(frozenset(x for x in range(m) if S>>x&1),
                               frozenset(x for x in range(m) if not S>>x&1)),
                    'queries':calls,'states':states}
        choices=[]
        if a<k and a<b:choices.append((a+1,b))
        if b<m:choices.append((a,b+1))
        if a<k and b<m:choices.append((a+1,b+1))
        viable=[(c,d) for c,d in choices if good(0,c,d)]
        if not viable:raise AssertionError('No feasible knife movement')
        a,b=viable[0]
        assert states<=m+1

def common_market_cardinal(m:int,v1:Callable,v2:Callable,u:Callable)->dict:
    ans=exact1(m,v1,u);X,Y=ans['bundles']
    a,b=v2(X),v2(Y)
    bundles=(Y,X) if a>=b else (X,Y)
    return {'owner':tuple(0 if g in bundles[0] else 1 for g in range(m)),
            'queries':ans['queries']+2,'cut_states':ans['states']}
