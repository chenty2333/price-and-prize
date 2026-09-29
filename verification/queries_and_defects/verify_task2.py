"""Exact small obstructions to steps, NOT to doubly EF1 existence."""
from itertools import product,combinations
from ordinal_market import ReversalPath
from test_algorithms import subjective,ef1

def additive(w):return lambda S:sum(w[g] for g in S)
def failures(a,fs):
 B=[frozenset(g for g,x in enumerate(a) if x==i) for i in range(2)]
 return tuple(fs[i](B[i])<min((fs[i](B[1-i]-{g}) for g in B[1-i]),default=0) for i in range(2))
# Smallest possible bundle sizes for an opposite-direction swap jump.
P=[list(range(4))]*2;u1=additive((1,0,0,1));u2=additive((1,0,1,0))
path=ReversalPath.build(P)
a,b=path.owner(0),path.owner(1)
assert subjective(a,P) and subjective(b,P)
assert failures(a,[u1,u2])==(False,True)
assert failures(b,[u1,u2])==(True,False)
print('Four-good swap jump:',a,failures(a,[u1,u2]),'->',b,failures(b,[u1,u2]))
# Moves also fail for heterogeneous functions.
P=[[0,1,2],[2,0,1]];u1=additive((0,1,1));u2=additive((1,1,0))
a,b=(0,0,1),(0,1,1)
assert subjective(a,P) and subjective(b,P)
assert failures(a,[u1,u2])==(False,True)
assert failures(b,[u1,u2])==(True,False)
print('Three-good move jump:',a,failures(a,[u1,u2]),'->',b,failures(b,[u1,u2]))
# An entire standard reversal path misses all allocator-fair solutions.
m=6;P=[list(range(m))]*2;path=ReversalPath.build(P)
u1=additive((0,0,0,1,0,1));u2=additive((0,1,0,1,0,0))
for t in range(path.steps+1):
 a=path.owner(t);assert subjective(a,P);assert any(failures(a,[u1,u2]))
 print('Six-good path:',t,a,failures(a,[u1,u2]))
S=[a for a in product(range(2),repeat=m) if subjective(a,P)]
good=[a for a in S if ef1(a,[u1,u2])]
assert good
print('All six-good SD allocations:',len(S),'allocator-EF1 among them:',len(good))
# Non-additive subjectives with strict singleton ranking 0>1>...>5.
E1=set(combinations([1,3,5],2))|set(combinations([0,3,5],2))
E2=set(combinations([1,3,4],2))|set(combinations([1,3,5],2))
def valuation(E):
 return lambda S:sum(6-g for g in S)+100*any(set(e)<=S for e in E)
f1,f2=valuation(E1),valuation(E2)
for f in [f1,f2]:
 assert [f(frozenset({g})) for g in range(6)]==[6,5,4,3,2,1]
 for s in range(64):
  T=frozenset(g for g in range(6) if s>>g&1)
  assert all(f(T)<=f(T|{g}) for g in range(6))
assert all(not ef1(path.owner(t),[f1,f2]) for t in range(path.steps+1))
print('Strict-singleton-rank monotone example: EF1 path states:',0)
# Fixed-circle two-knives does not solve arbitrary four-function input.
F=[additive(w) for w in [(1,1,0,0),(0,1,1,0),(0,0,1,1),(1,0,0,1)]]
def doubly(a):return ef1(a,F[:2]) and ef1(a,F[2:])
solutions=[a for a in product(range(2),repeat=4) if doubly(a)]
arc_masks={0,15}
for start in range(4):
 for length in range(1,4):arc_masks.add(sum(1<<((start+k)%4) for k in range(length)))
arc=[a for a in product(range(2),repeat=4) if sum(1<<g for g in range(4) if a[g]==0) in arc_masks]
assert solutions==[(0,1,0,1),(1,0,1,0)] and not any(map(doubly,arc))
print('Four-function example: all allocations:',16,'doubly EF1:',len(solutions),'circular allocations:',len(arc),'circular doubly EF1:',0)
# Opposite-sign two-item impossibility, all four allocations.
v=additive((2,-1));market=additive((1,1))
for a in product(range(2),repeat=2):
 x=ef1(a,[v,v],frozenset({1}));y=ef1(a,[market,market])
 assert not(x and y)
 print('Sign mismatch:',a,x,y)
print('All exact Task 2 and sign-mismatch checks passed.')
