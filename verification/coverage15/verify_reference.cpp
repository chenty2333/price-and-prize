// Exact three-agent ranking coverage, with learned covering subfamilies.
// Build: c++ -O3 -std=c++17 verify_cover.cpp -o verify_cover
// No probabilistic pruning, no hash-only equality, no candidate thinning.
#include <bits/stdc++.h>
using namespace std;using U=uint64_t;using Blocks=vector<array<int,3>>;
struct Cover {
 int m,q,N,w;Blocks blocks;string family;
 vector<array<unsigned,3>> candidates;vector<U> valid,robust;
 vector<vector<vector<U>>> cores;
 uint64_t calls=0,hits=0,safe=0,learned=0,retained=0;
 vector<int>path,witness;
 Cover(const Blocks& bs,string fam):m(3*bs.size()),q(bs.size()),blocks(bs),family(fam){
  unsigned a[3]={};int perm[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
  function<void(int)>gen=[&](int b){if(b==q){
   if(family=="own"||family=="full")for(int t=0;t<m;t+=3){if(__builtin_popcount(a[1]&(7u<<t))!=1)return;
    if(family=="full"&&(__builtin_popcount(a[0]&(7u<<t))!=1||__builtin_popcount(a[2]&(7u<<t))!=1))return;}
   if(family=="shift1"||family=="shift2"){
    int surplus=0,height=0,limit=family=="shift1"?1:2;
    for(int t=0;t<m;t+=3){int d=__builtin_popcount(a[1]&(7u<<t))-1;height+=d;if(height<0)return;if(d>0)surplus+=d;}
    if(height!=0||surplus>limit)return;
   }
   for(int t=1;t<=m;t++){unsigned p=(1u<<t)-1;int s=__builtin_popcount(a[1]&p);
    if(__builtin_popcount(a[0]&p)>s+1||__builtin_popcount(a[2]&p)>s+1)return;}
   candidates.push_back({a[0],a[1],a[2]});return;
  }
  for(auto&p:perm){for(int t=0;t<3;t++)a[p[t]]|=1u<<blocks[b][t];gen(b+1);for(int t=0;t<3;t++)a[p[t]]^=1u<<blocks[b][t];}};
  gen(0);N=candidates.size();w=(N+63)/64;size_t cells=size_t(1u<<m)*w;
  if(cells>150000000ull)throw runtime_error("Precomputed masks exceed memory limit; this partition was NOT verified.");
  valid.assign(cells,0);robust.assign(cells,0);cores.resize(1u<<m);
  for(unsigned p=0;p<(1u<<m);p++)for(int c=0;c<N;c++){
   int x=__builtin_popcount(candidates[c][2]&p);
   if(__builtin_popcount(candidates[c][0]&p)<=x+1&&__builtin_popcount(candidates[c][1]&p)<=x+1)valid[size_t(p)*w+c/64]|=1ull<<(c%64);
   if(x>=q-1)robust[size_t(p)*w+c/64]|=1ull<<(c%64);
  }
 }
 bool subset(const vector<U>&a,const vector<U>&b)const{for(int j=0;j<w;j++)if(a[j]&~b[j])return false;return true;}
 bool uncovered(unsigned p,const vector<U>&alive,vector<U>&proof){
  calls++;proof.assign(w,0);bool any=false;
  for(int j=0;j<w;j++){U r=alive[j]&robust[size_t(p)*w+j];if(r){proof[j]=r&-r;safe++;return false;}any|=alive[j]!=0;}
  if(!any){witness=path;return true;}
  for(auto&old:cores[p])if(subset(old,alive)){hits++;proof=old;return false;}
  struct Child{int g,count;bool safe;vector<U>a;};vector<Child>chs;
  for(int g=0;g<m;g++)if(!(p>>g&1)){
   unsigned t=p|(1u<<g);Child ch{g,0,false,vector<U>(w)};
   for(int j=0;j<w;j++){ch.a[j]=alive[j]&valid[size_t(t)*w+j];ch.count+=__builtin_popcountll(ch.a[j]);if(ch.a[j]&robust[size_t(t)*w+j])ch.safe=true;}
   chs.push_back(move(ch));
  }
  sort(chs.begin(),chs.end(),[](const Child&a,const Child&b){if(a.safe!=b.safe)return !a.safe;if(a.count!=b.count)return a.count<b.count;return a.g<b.g;});
  vector<U>cp;
  for(auto&ch:chs){unsigned t=p|(1u<<ch.g);
   if(ch.safe){bool reused=false;for(int j=0;j<w;j++)if(ch.a[j]&robust[size_t(t)*w+j]&proof[j]){reused=true;break;}
    if(!reused)for(int j=0;j<w;j++){U r=ch.a[j]&robust[size_t(t)*w+j];if(r){proof[j]|=r&-r;break;}}safe++;
   }else{path.push_back(ch.g);if(uncovered(t,ch.a,cp))return true;path.pop_back();for(int j=0;j<w;j++)proof[j]|=cp[j];}
  }
  auto&list=cores[p];for(size_t i=0;i<list.size();){if(subset(proof,list[i])){list[i]=move(list.back());list.pop_back();retained--;}else i++;}
  list.push_back(proof);learned++;retained++;return false;
 }
 bool run(){vector<U>a(w,~0ull),p;if(w&&N%64)a.back()=(1ull<<(N%64))-1;return !uncovered(0,a,p);}
};
vector<unsigned> pkey(const Blocks&b){vector<unsigned>k;for(auto t:b)k.push_back((1u<<t[0])|(1u<<t[1])|(1u<<t[2]));sort(k.begin(),k.end());return k;}
vector<vector<unsigned>> orbit(const Blocks&b,int m){vector<vector<unsigned>>o;array<int,3>tail{m-3,m-2,m-1};do{for(int f=0;f<2;f++){Blocks t=b;for(auto&z:t)for(int&g:z){if(g<2)g^=f;else if(g>=m-3)g=tail[g-(m-3)];}o.push_back(pkey(t));}}while(next_permutation(tail.begin(),tail.end()));sort(o.begin(),o.end());o.erase(unique(o.begin(),o.end()),o.end());return o;}
int main(int argc,char**argv){try{
 int m=15,first=0,last=-1;bool sym=true,adaptive=false;string family="own",rec="";
 for(int i=1;i<argc;i++){string s=argv[i];auto val=[&](){if(i+1>=argc)throw runtime_error("Missing argument");return string(argv[++i]);};
 if(s=="--m")m=stoi(val());else if(s=="--first")first=stoi(val());else if(s=="--last")last=stoi(val());else if(s=="--no-symmetry")sym=false;else if(s=="--adaptive-own")adaptive=true;else if(s=="--family")family=val();else if(s=="--records")rec=val();else throw runtime_error("Unknown option "+s);}
 const int totals[]={0,1,10,280,15400,1401400,190590400};
 if(m<6||m>18||m%3)throw runtime_error("m must be 6,9,12,15,18");if(last<0)last=totals[m/3];
 if(first<0||last<first||last>totals[m/3])throw runtime_error("Invalid partition range");
 if(family!="own"&&family!="full"&&family!="all"&&family!="shift1"&&family!="shift2")throw runtime_error("Unknown family");if(family=="all"&&sym)throw runtime_error("Use --no-symmetry with family all");
 ofstream records;if(!rec.empty()){records.open(rec);if(!records)throw runtime_error("Cannot open records file");}
 int index=0,checked=0,minN=INT_MAX,maxN=0,failidx=-1;uint64_t ownfailures=0,represented=0,calls=0,hits=0,safe=0,learned=0,retained=0,maxcalls=0;array<uint64_t,13>hist{};
 Blocks b,failedblocks;vector<int>failedprefix;bool failed=false;auto start=chrono::steady_clock::now();
 function<void(unsigned)>parts=[&](unsigned mask){if(index>=last||failed)return;if(mask){int a=__builtin_ctz(mask);unsigned rest=mask^(1u<<a);for(int x=a+1;x<m;x++)if(rest>>x&1)for(int y=x+1;y<m;y++)if(rest>>y&1){b.push_back({a,x,y});parts(rest^(1u<<x)^(1u<<y));b.pop_back();}return;}
 int idx=index++;if(idx<first)return;int weight=1;if(sym){auto o=orbit(b,m);if(pkey(b)!=o.front())return;weight=o.size();}
 auto t=chrono::steady_clock::now();auto cptr=make_unique<Cover>(b,adaptive?"own":family);bool ok=cptr->run();
 uint64_t pre_calls=0,pre_hits=0,pre_safe=0,pre_learned=0,pre_retained=0;
 if(adaptive&&!ok&&family!="own"){
  ownfailures++;pre_calls=cptr->calls;pre_hits=cptr->hits;pre_safe=cptr->safe;pre_learned=cptr->learned;pre_retained=cptr->retained;
  cptr.reset();cptr=make_unique<Cover>(b,family);ok=cptr->run();
 }
 Cover&c=*cptr;c.calls+=pre_calls;c.hits+=pre_hits;c.safe+=pre_safe;c.learned+=pre_learned;c.retained+=pre_retained;
 double seconds=chrono::duration<double>(chrono::steady_clock::now()-t).count();
 checked++;represented+=weight;hist[weight]++;calls+=c.calls;hits+=c.hits;safe+=c.safe;learned+=c.learned;retained+=c.retained;maxcalls=max(maxcalls,c.calls);minN=min(minN,c.N);maxN=max(maxN,c.N);
 if(records)records<<idx<<","<<weight<<","<<c.N<<","<<c.calls<<","<<c.learned<<","<<c.retained<<","<<c.hits<<","<<c.safe<<","<<seconds<<"\n";
 if(!ok){failed=true;failedblocks=b;failedprefix=c.witness;failidx=idx;return;}
 if(checked%1000==0)cerr<<"checked "<<checked<<" represented "<<represented<<" calls "<<calls<<" seconds "<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<"\n";
 };parts((1u<<m)-1);
 cout<<"{\n\"status\":\""<<(failed?"FAIL":"PASS")<<"\",\n\"m\":"<<m<<",\n\"family\":\""<<family<<"\",\n\"symmetry\":"<<(sym?"true":"false")<<",\n\"adaptive_own\":"<<(adaptive?"true":"false")<<",\n\"own_failures\":"<<ownfailures<<",\n\"first_partition\":"<<first<<",\n\"last_partition_exclusive\":"<<last<<",\n\"partitions_checked\":"<<checked<<",\n\"partitions_represented\":"<<represented<<",\n\"recursive_calls\":"<<calls<<",\n\"learned_cores\":"<<learned<<",\n\"retained_cores\":"<<retained<<",\n\"subsumption_hits\":"<<hits<<",\n\"safe_prunes\":"<<safe<<",\n\"min_candidates\":"<<(checked?minN:0)<<",\n\"max_candidates\":"<<maxN<<",\n\"max_calls_per_partition\":"<<maxcalls<<",\n\"seconds\":"<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<",\n\"orbit_histogram\":{";
 bool comma=false;for(int i=1;i<=12;i++)if(hist[i]){if(comma)cout<<",";comma=true;cout<<"\""<<i<<"\":"<<hist[i];}cout<<"}";
 if(failed){cout<<",\n\"failed_partition_index\":"<<failidx<<",\n\"failed_blocks\":[";for(size_t i=0;i<failedblocks.size();i++){if(i)cout<<",";auto t=failedblocks[i];cout<<"["<<t[0]<<","<<t[1]<<","<<t[2]<<"]";}cout<<"],\n\"uncovered_prefix\":[";for(size_t i=0;i<failedprefix.size();i++){if(i)cout<<",";cout<<failedprefix[i];}cout<<"]";}
 cout<<"\n}\n";return failed?1:0;
 }catch(const exception&e){cerr<<"ERROR: "<<e.what()<<"\n";return 2;}}
