"""Rebuild the independent C++ enumerator and reproduce the explicit counts."""
from __future__ import annotations
import argparse,json,pathlib,subprocess,time
from round5 import counterexample13,shifted_modules
ROOT=pathlib.Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--skip30',action='store_true');ap.add_argument('--output',default='bruteforce_reproduction.json');args=ap.parse_args()
    exe=ROOT/'brute_profile';subprocess.check_call(['c++','-O3','-std=c++17',str(ROOT/'brute_profile.cpp'),'-o',str(exe)])
    start=time.perf_counter();base,_=counterexample13();records=[]
    for m in [13,14,15]+([] if args.skip30 else [30]):
        profile=[s+list(range(13,m)) for s in base] if m<=15 else shifted_modules(2)[0]
        text=f'{m}\n'+'\n'.join(' '.join(map(str,s)) for s in profile[1:])+'\n'
        result=json.loads(subprocess.check_output([str(exe)],input=text,text=True))
        assert result['fair_own_balanced_allocations']==0
        assert result['fair_allocations']=={13:206,14:412,15:412,30:169744}[m]
        assert result['market_allocations']=={13:3888,14:7776,15:7776,30:60466176}[m]
        if m in (15,30):assert result['shift_histogram']=={str(m//15):result['fair_allocations']}
        records.append(result)
    report={'status':'PASS','records':records,'seconds_excluding_compilation':time.perf_counter()-start}
    (ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
