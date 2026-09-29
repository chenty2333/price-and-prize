"""Independent audit of every recorded endpoint orbit and all integer log sums.

This checks exhaustive domain accounting, not every recursive coverage inference.
The latter is independently spot-checked by verify_python.py.
"""
from __future__ import annotations
import argparse,csv,json,pathlib,time
from verify_python import unrank_partition,partition_count
from test_symmetry import signature_and_weight


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',default='coverage15');ap.add_argument('--output',default='log_audit.json');args=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parent
    directory=root/args.directory;report=json.loads((directory/'verification.json').read_text())
    assert report['status']=='PASS'
    m=report['m'];start=time.perf_counter();seen=set();sums=[0]*12
    keys=['G_orbits_checked','partitions_represented','coverage_problems','recursive_calls','learned_cores','retained_cores',
          'subsumption_hits','safe_prunes','own_first_failures','shift_first_failures','H_suborbits_checked','own_failures_in_suborbits']
    for first,last in report['shard_ranges']:
        shard=json.loads((directory/f'{first:09d}_{last:09d}.json').read_text())
        assert shard['status']=='PASS'
        assert shard['first_partition']==first and shard['last_partition_exclusive']==last
        path=directory/f'{first:09d}_{last:09d}.csv';local=[0]*12
        with path.open() as f:
            for row in csv.reader(f):
                assert len(row)==13
                values=list(map(int,row[:12]));idx=values[0];assert first<=idx<last
                blocks=unrank_partition(m,idx);sig,weight=signature_and_weight(blocks,m,3)
                assert sig not in seen,('duplicate orbit',idx);seen.add(sig)
                assert weight==values[1],('wrong orbit size',idx,weight,values[1])
                values[0]=1
                for j,v in enumerate(values):local[j]+=v;sums[j]+=v
        for key,value in zip(keys,local):assert shard[key]==value,(path,key,shard[key],value)
    for key,value in zip(keys,sums):assert report[key]==value,(key,report[key],value)
    assert sums[1]==partition_count(m)
    N={t:partition_count(t) for t in range(0,m+1,3)}
    formula=(N[m]+(6*(m-2)+4)*N[m-3]+(9*(m-4)*(m-5)+12*(m-5)+4)*N[m-6])//36
    assert len(seen)==formula
    result={'status':'PASS','m':m,'every_recorded_orbit_independently_checked':True,
            'distinct_G_orbits':len(seen),'sum_orbit_sizes':sums[1],
            'all_shard_and_aggregate_integer_counters_agree':True,
            'integer_totals':dict(zip(keys,sums)),'seconds':time.perf_counter()-start,
            'scope':'Domain accounting and integer-log audit; not an independent replay of all coverage states.'}
    (root/args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
