"""Independent integer check of every saved three-agent pair."""
import json
from verify_joint_selection_obstruction import rows
records=json.load(open('pair_selection_all.json'))
for r in records:
    P=r['profile'];A,B=r['pair']
    for a in (A,B):
        for i,ranking in enumerate(P):
            c=[0,0,0]
            for g in ranking:
                c[a[g]]+=1
                assert max(c)<=c[i]+1
    assert all(x+y<=0 for ca in rows(A) for cb in rows(B) for x,y in zip(ca,cb))
print('Release audit: exact independent pair certificates checked:',len(records))
print('Cross-row integer inequalities:',sum(36*r['m'] for r in records))
