"""Compile and exhaustively verify all canonical partitions in recorded shards.
A missing/failed shard prevents aggregate PASS. No probabilistic pruning.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, pathlib, platform, subprocess, time
COUNTS={6:10,9:280,12:15400,15:1401400,18:190590400}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--m',type=int,default=15,choices=COUNTS)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--shard-size',type=int,default=20000)
    ap.add_argument('--output-dir',default='coverage15');args=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parent;out=pathlib.Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    source=root/'verify_cover.cpp';exe=out/'verify_cover';command=['c++','-O3','-march=native','-std=c++17',str(source),'-o',str(exe)]
    subprocess.run(command,check=True);start=time.perf_counter();ranges=[(a,min(a+args.shard_size,COUNTS[args.m])) for a in range(0,COUNTS[args.m],args.shard_size)]
    def run(bounds):
        a,b=bounds;stem=f'{a:09d}_{b:09d}'
        cmd=[str(exe),'--m',str(args.m),'--first',str(a),'--last',str(b),'--records',str(out/(stem+'.csv'))]
        with (out/(stem+'.json')).open('w') as f,(out/(stem+'.log')).open('w') as log:proc=subprocess.run(cmd,stdout=f,stderr=log)
        data=json.loads((out/(stem+'.json')).read_text())
        if proc.returncode!=0 or data['status']!='PASS':raise RuntimeError(f'Shard {stem} failed: {data}')
        print(f'COMPLETE {stem}: {data["G_orbits_checked"]} representatives, {data["seconds"]:.3f} s',flush=True)
        return data
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(run,ranges))
    assert [r['first_partition'] for r in rows]==[a for a,b in ranges]
    assert sum(r['partitions_represented'] for r in rows)==COUNTS[args.m]
    result={'status':'PASS','m':args.m,'family':'one_forward_shift','full_domain_partitions':COUNTS[args.m],
      'symmetry_group':'S_3 on first triple times S_3 on final triple; subdivide under S_2 times S_3 on fallback',
      'shards':len(rows),'workers':args.workers,'wall_seconds':time.perf_counter()-start,
      'sum_shard_seconds':sum(r['seconds'] for r in rows),
      'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
      'compile_command':command,'machine':platform.platform(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    for k in ['G_orbits_checked','partitions_represented','coverage_problems','recursive_calls','learned_cores','retained_cores','subsumption_hits','safe_prunes','own_first_failures','shift_first_failures','H_suborbits_checked','own_failures_in_suborbits']:result[k]=sum(r[k] for r in rows)
    result['min_candidates_over_attempts']=min(r['min_candidates_over_attempts'] for r in rows if r['G_orbits_checked']);result['max_candidates_over_attempts']=max(r['max_candidates_over_attempts'] for r in rows)
    result['max_calls_per_G_orbit']=max(r['max_calls_per_G_orbit'] for r in rows)
    hist={}
    for r in rows:
        for k,v in r['orbit_histogram'].items():hist[k]=hist.get(k,0)+v
    result['orbit_histogram']=hist;result['shard_ranges']=ranges
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
