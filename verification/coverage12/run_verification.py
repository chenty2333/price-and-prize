"""Compile and rerun the full 12-good coverage verification in exact chunks."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,pathlib,subprocess


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--chunk-size',type=int,default=1000)
    ap.add_argument('--output',default='verification_12_rerun.json')
    args=ap.parse_args()
    if args.workers<1 or args.chunk_size<1:raise ValueError('Arguments must be positive')
    here=pathlib.Path(__file__).resolve().parent
    source=here/'verify_12.cpp';binary=here/'verify_12'
    subprocess.run(['g++','-O3','-std=c++17',str(source),'-o',str(binary)],check=True)
    jobs=[(a,min(a+args.chunk_size,15400)) for a in range(0,15400,args.chunk_size)]
    def run(job):
        a,b=job
        p=subprocess.run([str(binary),'1',str(a),str(b)],text=True,capture_output=True,check=True)
        result=json.loads(p.stdout)
        assert result['status']=='PASS' and result['own_balance']
        assert result['first_partition']==a and result['last_partition_exclusive']==b
        assert result['partitions']==b-a
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        chunks=list(pool.map(run,jobs))
    expected=0
    for c in chunks:
        assert c['first_partition']==expected
        expected=c['last_partition_exclusive']
    assert expected==15400
    data={'status':'PASS','m':12,'own_balance':True,'partitions':15400,
          'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    for k in ('nodes','memo_states','memo_hits','safe_prunes'):
        data[k]=sum(c[k] for c in chunks)
    data.update(min_candidates=min(c['min_candidates'] for c in chunks),
                max_candidates=max(c['max_candidates'] for c in chunks),
                max_nodes_per_partition=max(c['max_nodes_per_partition'] for c in chunks),chunks=chunks)
    pathlib.Path(args.output).write_text(json.dumps(data,indent=2))
    print(json.dumps({k:v for k,v in data.items() if k!='chunks'},indent=2))

if __name__=='__main__':main()
