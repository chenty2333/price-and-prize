"""Reproducible random tests, obstruction enumeration, and structural checks."""
from __future__ import annotations
import argparse,collections,itertools,json,random,time
from round2 import *

SEED=20260929
OBS=[
    [list(range(12)),[0,3,1,6,4,9,7,10,2,5,8,11],[0,6,1,3,7,9,4,10,2,5,8,11]],
    [list(range(10)),[6,0,7,9,4,5,3,1,2,8],[7,4,5,6,8,9,0,3,2,1]],
]
FAIL_SWAP=[list(range(10)),[0,5,4,9,1,8,2,3,7,6],[5,4,8,9,3,6,0,7,1,2]]
FAIL_FREEZE=[list(range(10)),[9,4,2,1,6,5,7,3,8,0],[1,4,5,3,6,0,9,2,7,8]]

def own_balanced(p,o):
    return all(sum(o[g]==1 for g in p[1][b:b+3])<=1 for b in range(0,len(o),3))

def fixed_family(p,mode):
    m=len(p[0]);cl=[g%3 for g in range(m)];out=set()
    if mode=='swap':
        for perm in itertools.permutations(range(3)):
            base=tuple(perm[c] for c in cl);out.add(base)
            for b in range(0,m,3):
                for g,h in itertools.combinations(range(b,min(b+3,m)),2):
                    o=list(base);o[g],o[h]=o[h],o[g];out.add(tuple(o))
    elif mode=='freeze':
        for c in range(3):
            choices=[]
            for b in range(0,m,3):
                rem=[g for g in range(b,min(b+3,m)) if cl[g]!=c]
                choices.append([list(zip(rem,ps)) for ps in itertools.permutations((0,2),len(rem))])
            for pp in itertools.product(*choices):
                o=[1]*m
                for picks in pp:
                    for g,x in picks:o[g]=x
                out.add(tuple(o))
    else:
        raise ValueError(mode)
    return [list(o) for o in sorted(out)]

def enumerate_profile(p):
    allcount=valid=balanced_own=both_balanced=0;witness=None
    for o in all_market_allocations(len(p[0])):
        allcount+=1
        if check_allocation(p,o) is None:
            valid+=1
            balanced_own+=own_balanced(p,o)
            both_balanced+=all(len(set(o[g] for g in p[1][b:b+3]))==len(p[1][b:b+3]) for b in range(0,len(o),3))
            if witness is None:witness=o
    return {'profile':p,'market_allocations':allcount,'valid_allocations':valid,
            'valid_own_balanced':balanced_own,'valid_all_classes_balanced_sigma1':both_balanced,'witness':witness}

def kempe_certificate():
    p1=[0,3,6,1,4,7,2,5,8]
    vertices=[]
    for o in all_market_allocations(9):
        if all(len({o[g] for g in p1[b:b+3]})==3 for b in (0,3,6)):
            vertices.append(tuple(o))
    edges={o:set() for o in vertices}
    for o in vertices:
        for a,b in itertools.combinations(range(3),2):
            unused={g for g in range(9) if o[g] in (a,b)}
            while unused:
                root=min(unused);component={root};stack=[root];unused.remove(root)
                while stack:
                    g=stack.pop()
                    neighbors={h for h in unused if h//3==g//3 or h%3==g%3}
                    unused-=neighbors;component|=neighbors;stack.extend(neighbors)
                no=list(o)
                for g in component:no[g]=a+b-no[g]
                no=tuple(no)
                assert no in edges
                edges[o].add(no)
    unseen=set(vertices);sizes=[]
    while unseen:
        stack=[unseen.pop()];size=0
        while stack:
            o=stack.pop();size+=1
            new=edges[o]&unseen;unseen-=new;stack.extend(new)
        sizes.append(size)
    assert len(vertices)==12 and sorted(sizes)==[6,6]
    return {'sigma1':p1,'colorings':len(vertices),'kempe_component_sizes':sorted(sizes)}

def generate_modules(m,rng):
    M=3*((m+2)//3);segments=[];b=0
    while b<M:
        width=3*rng.randint(1,min(4,(M-b)//3));segments.append(list(range(b,b+width)));b+=width
    pp=[list(range(m))]
    for i in (1,2):
        chunks=[c[:] for c in segments]
        if m<M and chunks:
            last=chunks.pop();rng.shuffle(chunks);chunks.append(last)
        else:rng.shuffle(chunks)
        ranked=[]
        for ch in chunks:
            real=[g for g in ch if g<m];rng.shuffle(real)
            virtual=[g for g in ch if g>=m]
            ranked.extend(real+virtual)
        pp.append([g for g in ranked if g<m])
    return pp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='test_results.json')
    ap.add_argument('--general-per-m',type=int,default=1000)
    args=ap.parse_args();rng=random.Random(SEED);start=time.perf_counter();report={'seed':SEED,'suites':[]}
    def suite(name,ms,reps,solver,generator,require_own):
        total=0;stats_total=collections.Counter();max_orientations=0;max_matchings=0
        for m in ms:
            for _ in range(reps):
                p=generator(m,rng);stats={}
                o=solver(p,stats)
                assert o is not None,(name,p)
                assert check_allocation(p,o) is None,(name,p,o)
                if require_own:assert own_balanced(p,o),(name,p,o)
                total+=1;stats_total.update(stats)
                max_orientations=max(max_orientations,stats.get('orientations',0))
                max_matchings=max(max_matchings,stats.get('matchings',0))
        result={'name':name,'tests':total,'failures':0,'max_m':max(ms),'own_balance_checked':require_own,'search_totals':dict(stats_total),'max_orientations':max_orientations,'max_matchings':max_matchings}
        report['suites'].append(result);print(json.dumps(result),flush=True)
    random_profile=lambda m,r:[list(range(m)),r.sample(range(m),m),r.sample(range(m),m)]
    suite('certified_up_to_12',range(13),500,solve_up_to_12,random_profile,True)
    suite('restricted_exact_search_general_NO_EXISTENCE_THEOREM_above_12',range(1,31),args.general_per_m,search_own_balanced,random_profile,True)
    suite('common_modules_up_to_12',range(31),200,lambda p,stats:solve_modules(p),generate_modules,True)
    report['obstructions']=[]
    for name,p in [('previous_serial',OBS[0]),('previous_double_balance',OBS[1]),('one_swap',FAIL_SWAP),('fixed_own_class',FAIL_FREEZE)]:
        data=enumerate_profile(p);data['name']=name
        o=solve_up_to_12(p);assert check_allocation(p,o) is None and own_balanced(p,o)
        data['solver_witness']=o
        if name in ('one_swap','fixed_own_class'):
            mode='swap' if name=='one_swap' else 'freeze'
            cl=[g%3 for g in range(len(p[0]))]
            assert all(len(set(cl[g] for g in p[1][b:b+3]))==len(p[1][b:b+3]) for b in range(0,len(cl),3))
            ff=fixed_family(p,mode);violations=[check_allocation(p,o) for o in ff]
            assert all(e is not None for e in violations)
            data.update(base_classes=cl,restricted_family_size=len(ff),restricted_valid=0,
                        restricted_candidates=[{'owner':o,'violation':e} for o,e in zip(ff,violations)])
        report['obstructions'].append(data)
    assert report['obstructions'][0]['valid_allocations']==104
    assert report['obstructions'][1]['valid_allocations']==87
    word_profile=[list(range(10)),list(range(10)),[1,4,0,3,6,9,7,2,5,8]]
    rejected=[]
    for chosen in range(3):
        rest=[c for c in range(3) if c!=chosen]
        mapping={chosen:2,rest[0]:1,rest[1]:0}
        owner=[mapping[g%3] for g in range(10)]
        e=check_allocation(word_profile,owner)
        assert e is not None and e['agent']==2
        rejected.append({'chosen_class':chosen,'violation':e})
    report['chooser_failure']={'profile':word_profile,'class_word':'BBAAAABCCC','rejections':rejected}
    report['kempe']=kempe_certificate()
    p=[list(range(6)),[0,3,1,4,2,5],[2,5,1,4,0,3]];o=[0,1,2,0,1,2]
    error=check_allocation(p,o);assert error is not None
    # Each one-block restriction is trivially market-balanced SD-EF1.
    for b in (0,3):
        local=[list(range(3))]+[[g-b for g in s if b<=g<b+3] for s in p[1:]]
        assert check_allocation(local,o[b:b+3]) is None
    report['gluing_failure']={'profile':p,'owner':o,'violation':error,'each_chunk_valid':True}
    report['random_tests']=sum(s['tests'] for s in report['suites'])
    report['seconds']=time.perf_counter()-start
    with open(args.output,'w') as f:json.dump(report,f,indent=2)
    print('TOTAL',report['random_tests'],'seconds',report['seconds'])

if __name__=='__main__':main()
