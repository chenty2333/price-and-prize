from itertools import permutations, product
from pathlib import Path
import json,random,time
from round3 import repair_parallel_edge
from independent_checker import check_allocation,own_balance


def direct_eligible(profile,owner):
    """Independent direct evaluation of every hypothesis and candidate swap."""
    m=len(owner)
    count=[0,0,0]
    for g in profile[1]:
        count[owner[g]]+=1
        if max(count)>count[1]+1:return []
    prefix=[];count=[0,0,0];bad=[]
    for t,g in enumerate(profile[2],1):
        count[owner[g]]+=1
        prefix.append(count.copy())
        if count[1]>count[2]+1 or count[0]>count[2]+2:return []
        if count[0]>count[2]+1:bad.append(t)
    if not bad:return [list(owner)]
    rank1={g:t for t,g in enumerate(profile[1])}
    rank2={g:t+1 for t,g in enumerate(profile[2])}
    ans=[]
    for k in range(0,m,3):
        a=next(g for g in range(k,k+3) if owner[g]==1)
        b=next(g for g in range(k,k+3) if owner[g]==0)
        if rank1[a]//3!=rank1[b]//3 or rank1[b]>=rank1[a]:continue
        lo,hi=rank2[b],rank2[a]-1
        if any(not lo<=t<=hi for t in bad):continue
        if any(prefix[t-1][1]>prefix[t-1][2] for t in range(lo,hi+1)):continue
        new=owner.copy();new[a],new[b]=new[b],new[a]
        ans.append(new)
    return ans


def exchange_suite(count=5000):
    rng=random.Random(20260929)
    stats={'profiles':count,'max_m':30,'returned_fair':0,'nontrivial_repairs':0,'failures':0}
    for case in range(count):
        q=case%10+1;m=3*q
        owner=[i for k in range(q) for i in rng.sample(range(3),3)]
        classes=[[g for g in range(m) if owner[g]==i] for i in range(3)]
        for a in classes:rng.shuffle(a)
        sigma1=[]
        for k in range(q):sigma1+=rng.sample([classes[i][k] for i in range(3)],3)
        sigma2=rng.sample(range(m),m)
        profile=[list(range(m)),sigma1,sigma2]
        ans=repair_parallel_edge(profile,owner)
        expected=direct_eligible(profile,owner)
        assert (ans is not None)==bool(expected),(profile,owner,ans,expected)
        if ans is not None:
            assert ans in expected
            assert check_allocation(profile,ans) is None,(profile,owner,ans)
            assert own_balance(profile,ans)
            assert [g for g in range(m) if owner[g]==2]==[g for g in range(m) if ans[g]==2]
            stats['returned_fair']+=1
            stats['nontrivial_repairs']+=ans!=owner
    return stats


def old_obstructions():
    data=[
      ('round1_serial', [0,3,1,6,4,9,7,10,2,5,8,11],[0,6,1,3,7,9,4,10,2,5,8,11],104,36),
      ('round1_full_balance',[6,0,7,9,4,5,3,1,2,8],[7,4,5,6,8,9,0,3,2,1],87,14),
      ('round2_one_swap',[0,5,4,9,1,8,2,3,7,6],[5,4,8,9,3,6,0,7,1,2],57,18),
      ('round2_frozen_bundle',[9,4,2,1,6,5,7,3,8,0],[1,4,5,3,6,0,9,2,7,8],78,8)]
    results=[]
    for name,s1,s2,expected,expected_own in data:
        m=len(s1);profile=[list(range(m)),s1,s2]
        all_count=fair=own=0
        choices=[list(permutations(range(3),min(3,m-k))) for k in range(0,m,3)]
        for parts in product(*choices):
            owner=[i for part in parts for i in part];all_count+=1
            if check_allocation(profile,owner) is None:
                fair+=1;own+=own_balance(profile,owner)
        assert fair==expected and own==expected_own,(name,fair,own)
        results.append({'name':name,'m':m,'allocations_checked':all_count,
                        'fair':fair,'own_balanced_fair':own})
    return results

if __name__=='__main__':
    t=time.perf_counter()
    result={'seed':20260929,'exchange':exchange_suite(),'previous_obstructions':old_obstructions()}
    Path(__file__).with_name('exchange_test_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result);print('seconds',time.perf_counter()-t)
