"""Independent exact full-state coverage checker (no learned subfamilies).

Candidates are generated as explicit owners and checked by direct prefix counts.
Prefix masks are computed by exact bit-sliced integer arithmetic; --self-test
crosschecks that arithmetic against direct counts. No endpoint symmetry is used.
"""
from __future__ import annotations
from itertools import permutations,product
from functools import lru_cache
import argparse,gc,json,math,pathlib,random,subprocess,time
@lru_cache(None)
def partition_count(m):
    return math.factorial(m)//(6**(m//3)*math.factorial(m//3))
def unrank_partition(m,index):
    if not 0<=index<partition_count(m):raise ValueError('Partition index out of range')
    goods=list(range(m));blocks=[]
    while goods:
        count=partition_count(len(goods)-3);choice,index=divmod(index,count)
        pairs=[(i,j) for i in range(1,len(goods)) for j in range(i+1,len(goods))]
        i,j=pairs[choice];block=(goods[0],goods[i],goods[j]);blocks.append(block)
        goods=[g for g in goods if g not in block]
    return tuple(blocks)
def candidates_for(blocks,family='own'):
    m=3*len(blocks);out=[]
    for choices in product(tuple(permutations(range(3))),repeat=len(blocks)):
        owner=[-1]*m
        for B,colors in zip(blocks,choices):
            for g,c in zip(B,colors):owner[g]=c
        counts=[sum(owner[g]==1 for g in range(t,t+3)) for t in range(0,m,3)]
        if family in ('own','full') and counts!=[1]*len(blocks):continue
        if family=='full' and any(len({owner[g] for g in range(t,t+3)})!=3 for t in range(0,m,3)):continue
        if family in ('shift1','shift2'):
            height=0;surplus=sum(max(c-1,0) for c in counts);bad=False
            for c in counts:
                height+=c-1
                if height<0:bad=True
            if bad or height or surplus>int(family[-1]):continue
        seen=[0,0,0]
        for c in owner:
            seen[c]+=1
            if seen[0]>seen[1]+1 or seen[2]>seen[1]+1:break
        else:out.append(tuple(sum(1<<g for g in range(m) if owner[g]==i) for i in range(3)))
    return out

def prefix_tables(candidates,m):
    """Gray-code traversal; parallel binary counters across exact candidate bits."""
    n=len(candidates);allbits=(1<<n)-1;q=m//3;nbits=(q+1).bit_length()
    rolebits=[[sum(1<<a for a,A in enumerate(candidates) if (A[i]>>g)&1) for g in range(m)] for i in range(3)]
    planes=[[0]*nbits for _ in range(3)];valid=[0]*(1<<m);safe=[0]*(1<<m);old=0
    def leq(a,b):
        eq=allbits;less=0
        for k in range(nbits-1,-1,-1):
            less|=eq&~a[k]&b[k];eq&=~(a[k]^b[k])
        return (less|eq)&allbits
    threshold=[allbits if ((q-1)>>b)&1 else 0 for b in range(nbits)]
    for t in range(1<<m):
        mask=t^(t>>1)
        if t:
            changed=old^mask;g=changed.bit_length()-1;adding=bool(mask&changed)
            for i in range(3):
                carry=rolebits[i][g]
                for b in range(nbits):
                    prev=planes[i][b];planes[i][b]^=carry
                    carry&=prev if adding else ~prev
        old=mask;carry=allbits;plus=[]
        for plane in planes[2]:plus.append(plane^carry);carry&=plane
        valid[mask]=leq(planes[0],plus)&leq(planes[1],plus)
        safe[mask]=leq(threshold,planes[2])
    return valid,safe

def verify_partition(blocks,family='own',crosscheck_masks=0):
    m=3*len(blocks);start=time.perf_counter();A=candidates_for(blocks,family);n=len(A)
    valid,safe=prefix_tables(A,m)
    rng=random.Random(20261001)
    masks=list(range(1<<m)) if crosscheck_masks<0 else rng.sample(range(1<<m),min(crosscheck_masks,1<<m))
    for T in masks:
        v=r=0
        for j,B in enumerate(A):
            d,s,x=[(z&T).bit_count() for z in B]
            if d<=x+1 and s<=x+1:v|=1<<j
            if x>=m//3-1:r|=1<<j
        assert (v,r)==(valid[T],safe[T]),('mask table mismatch',T)
    memo=[set() for _ in range(1<<m)];calls=hits=prunes=states=0;path=[];witness=None;allgoods=(1<<m)-1
    def uncovered(T,alive):
        nonlocal calls,hits,prunes,states,witness
        calls+=1
        if not alive:witness=path.copy();return True
        if alive&safe[T]:prunes+=1;return False
        if alive in memo[T]:hits+=1;return False
        children=[];remaining=allgoods^T
        while remaining:
            bit=remaining&-remaining;remaining-=bit;U=T|bit;child=alive&valid[U]
            if child&safe[U]:prunes+=1
            else:children.append((child.bit_count(),bit,child))
        children.sort()
        for _,bit,child in children:
            path.append(bit.bit_length()-1)
            if uncovered(T|bit,child):return True
            path.pop()
        memo[T].add(alive);states+=1;return False
    bad=uncovered(0,(1<<n)-1)
    return {'family':family,'candidates':n,'covered':not bad,'uncovered_prefix':witness,
            'recursive_calls':calls,'memo_states':states,'memo_hits':hits,'safe_prunes':prunes,
            'direct_mask_checks':len(masks),'seconds':time.perf_counter()-start}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--m',type=int,default=15);ap.add_argument('--indices',default='40877,765432,1401399,123456,271828,314159,1000000,555555,987654,112233')
    ap.add_argument('--cpp', help='Path to verify_reference, NOT the hierarchical primary verifier');ap.add_argument('--output',default='python_spot_checks.json');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    root=pathlib.Path(args.output);records=[];start=time.perf_counter()
    if args.self_test:
        for m in (6,9):
            for idx in sorted(set([0,partition_count(m)//2,partition_count(m)-1])):
                for family in ('own','shift1','full'):
                    result=verify_partition(unrank_partition(m,idx),family,crosscheck_masks=-1)
                    assert result['covered'];records.append({'m':m,'index':idx,**result});gc.collect()
        root.write_text(json.dumps({'status':'PASS','self_test':True,'records':records,'seconds':time.perf_counter()-start},indent=2));return
    for index in map(int,args.indices.split(',')):
        blocks=unrank_partition(args.m,index);attempts=[verify_partition(blocks,'own',crosscheck_masks=128)]
        gc.collect()
        if not attempts[0]['covered']:
            attempts.append(verify_partition(blocks,'shift1',crosscheck_masks=128));gc.collect()
        assert attempts[-1]['covered'],(blocks,attempts)
        record={'m':args.m,'partition_index':index,'blocks':blocks,'attempts':attempts}
        if args.cpp:
            cmd=[args.cpp,'--m',str(args.m),'--first',str(index),'--last',str(index+1),'--no-symmetry','--family','shift1','--adaptive-own']
            cpp=json.loads(subprocess.check_output(cmd,text=True));record['cpp']=cpp
            assert cpp['status']=='PASS' and cpp['min_candidates']==attempts[-1]['candidates']
            assert cpp['own_failures']==(not attempts[0]['covered'])
        records.append(record)
        report={'status':'PASS_FOR_LISTED_PARTITIONS','full_domain':False,'method':'exact full-state memoization; no learned cores or symmetry',
                'records':records,'seconds':time.perf_counter()-start}
        root.write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE Python partition',index,'seconds',sum(a['seconds'] for a in attempts),flush=True)
if __name__=='__main__':main()
