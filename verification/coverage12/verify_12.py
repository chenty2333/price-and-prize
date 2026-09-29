"""Independent, small Python implementation of the exact coverage verifier.

Usage:
  python verify_12.py --samples 40 --crosscheck ./verify_12
  python verify_12.py --all

The --all option checks every partition, not random profiles. The C++ port
is faster. No external packages are used. Only exact integer arithmetic.
"""
from __future__ import annotations
from itertools import permutations, product
from functools import lru_cache
import argparse,json,random,subprocess,time


def triple_partitions(goods):
    """Each partition into unordered triples, exactly once."""
    if not goods:
        yield ()
        return
    a=goods[0]
    for ix in range(1,len(goods)):
        for iy in range(ix+1,len(goods)):
            block=(a,goods[ix],goods[iy])
            rest=tuple(g for g in goods if g not in block)
            for tail in triple_partitions(rest):
                yield (block,)+tail


def candidates_for(blocks):
    """Generate with explicit owners; independently check agent 1's prefixes."""
    result=[]
    for choices in product(tuple(permutations((0,1,2))),repeat=4):
        owner=[-1]*12
        for block,colors in zip(blocks,choices):
            for g,c in zip(block,colors):
                owner[g]=c
        if any(sum(owner[g]==1 for g in range(b,b+3))!=1 for b in range(0,12,3)):
            continue
        counts=[0,0,0]
        for g in range(12):
            counts[owner[g]]+=1
            if counts[0]>counts[1]+1 or counts[2]>counts[1]+1:
                break
        else:
            result.append(tuple(sum(1<<g for g in range(12) if owner[g]==i) for i in range(3)))
    return result


def verify_partition(blocks, fully_balanced=False):
    candidates=candidates_for(blocks)
    if fully_balanced:
        candidates=[A for A in candidates if all((mask & (7<<b)).bit_count()==1 for mask in A for b in (0,3,6,9))]
    valid=[];safe=[]
    for S in range(1<<12):
        goodmask=safemask=0
        for k,(A0,A1,A2) in enumerate(candidates):
            c0,c1,c2=(S&A0).bit_count(),(S&A1).bit_count(),(S&A2).bit_count()
            if c0<=c2+1 and c1<=c2+1:
                goodmask|=1<<k
            if c2>=3:
                safemask|=1<<k
        valid.append(goodmask);safe.append(safemask)
    @lru_cache(None)
    def uncovered(S,alive):
        if not alive:
            return ()
        if safe[S]&alive:
            return None
        for g in range(12):
            if not (S>>g)&1:
                T=S|(1<<g)
                suffix=uncovered(T,alive&valid[T])
                if suffix is not None:
                    return (g,)+suffix
        return None
    witness=uncovered(0,(1<<len(candidates))-1)
    states=uncovered.cache_info().currsize
    uncovered.cache_clear()
    return {'candidates':len(candidates),'covered':witness is None,
            'uncovered_prefix':witness,'cached_states':states}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--all',action='store_true')
    ap.add_argument('--samples',type=int,default=40)
    ap.add_argument('--crosscheck',default=None)
    ap.add_argument('--output',default='python_verifier_results.json')
    args=ap.parse_args()
    parts=list(triple_partitions(tuple(range(12))))
    assert len(parts)==15400 and len(set(parts))==15400
    rng=random.Random(20260929)
    selected=list(range(15400)) if args.all else sorted(set([0,15399]+rng.sample(range(1,15399),max(0,args.samples-2))))
    records=[];start=time.perf_counter()
    for j in selected:
        result=verify_partition(parts[j]);result['partition_index']=j
        assert result['covered'],(parts[j],result)
        if args.crosscheck:
            data=json.loads(subprocess.check_output([args.crosscheck,'1',str(j),str(j+1)]))
            assert data['status']=='PASS' and data['partitions']==1
            assert data['min_candidates']==result['candidates']==data['max_candidates']
        records.append(result)
    report={'status':'PASS','full_enumeration':args.all,'partitions_checked':len(records),
            'crosschecked_cpp':bool(args.crosscheck),'seconds':time.perf_counter()-start,'records':records}
    with open(args.output,'w') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__':main()
