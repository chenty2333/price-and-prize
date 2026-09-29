"""False candidate families must yield actual bad ranking prefixes."""
from __future__ import annotations
from itertools import permutations,product
import json,pathlib,subprocess,time
from verify_python import candidates_for,verify_partition,unrank_partition
ROOT=pathlib.Path(__file__).resolve().parent

def assert_uncovered(blocks,result,family):
    assert not result['covered']
    prefix=result['uncovered_prefix'];m=3*len(blocks)
    sigma=prefix+[g for g in range(m) if g not in prefix]
    candidates=candidates_for(blocks,family)
    for masks in candidates:
        owner=[next(i for i in range(3) if (masks[i]>>g)&1) for g in range(m)]
        count=[0,0,0];bad=False
        for g in sigma:
            count[owner[g]]+=1
            if count[0]>count[2]+1 or count[1]>count[2]+1:bad=True;break
        assert bad
    return len(candidates)

def main():
    start=time.perf_counter();records=[]
    B=((0,2,9),(1,7,8),(3,10,11),(4,5,6))
    result=verify_partition(B,'full',crosscheck_masks=128)
    n=assert_uncovered(B,result,'full')
    records.append({'name':'full_sigma1_balance_negative_control','blocks':B,'all_candidates_rejected_by_direct_prefix_scan':n,**result})
    B=unrank_partition(15,40877)
    # Use the already completed independent full-state result, then check its
    # actual witness afresh without the cached mask tables or recursive routine.
    report=json.loads((ROOT/'python_spot_checks.json').read_text())
    row=next(r for r in report['records'] if r['partition_index']==40877)
    result=row['attempts'][0];n=assert_uncovered(B,result,'own')
    records.append({'name':'own_balance_negative_control','blocks':B,'all_candidates_rejected_by_direct_prefix_scan':n,**result})
    out={'status':'PASS','records':records,'seconds':time.perf_counter()-start}
    (ROOT/'negative_controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
