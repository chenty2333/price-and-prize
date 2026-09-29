// Exact three-agent ranking coverage, with learned covering subfamilies.
// Build: c++ -O3 -std=c++17 verify_cover.cpp -o verify_cover
// No probabilistic pruning, no hash-only equality, no candidate thinning.
#include <bits/stdc++.h>
using namespace std;using U=uint64_t;using Blocks=vector<array<int,3>>;
struct Cover {
 int m,q,N,w;Blocks blocks;string family;bool firstfull=false;
 vector<array<unsigned,3>> candidates;vector<U> valid,robust;
 vector<vector<vector<U>>> cores;
 uint64_t calls=0,hits=0,safe=0,learned=0,retained=0;
 vector<int>path,witness;
 Cover(const Blocks& bs,string fam):m(3*bs.size()),q(bs.size()),blocks(bs),family(fam){
  if(family.size()>6&&family.substr(family.size()-6)=="_first"){firstfull=true;family.resize(family.size()-6);}
  unsigned a[3]={};int perm[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
  function<void(int)>gen=[&](int b){if(b==q){
   if(firstfull)for(int i=0;i<3;i++)if(__builtin_popcount(a[i]&7u)!=1)return;
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
// Canonicalization under S_h on the first h labels and S_t on the final t.
// A block is determined up to this group by its fixed labels and mutable counts.
struct OrbitInfo {Blocks canonical;uint64_t weight;};
OrbitInfo orbit_info(const Blocks&bs,int m,int h,int t){
 unsigned all=(1u<<m)-1,hm=(1u<<h)-1,tm=((1u<<t)-1)<<(m-t);vector<unsigned> sig;
 uint64_t fact[]={1,1,2,6,24,120,720};uint64_t den=1;
 for(auto B:bs){unsigned mask=(1u<<B[0])|(1u<<B[1])|(1u<<B[2]);int a=__builtin_popcount(mask&hm),b=__builtin_popcount(mask&tm);unsigned fixed=mask&~(hm|tm);sig.push_back(fixed|(unsigned(a)<<m)|(unsigned(b)<<(m+2)));den*=fact[a]*fact[b];}
 sort(sig.begin(),sig.end());Blocks out;int nh=0,nt=m-t;
 for(size_t i=0;i<sig.size();){size_t j=i+1;while(j<sig.size()&&sig[j]==sig[i])j++;if((sig[i]&all)==0)den*=fact[j-i];i=j;}
 for(unsigned code:sig){unsigned fixed=code&all;int a=(code>>m)&3,b=(code>>(m+2))&3;vector<int>v;
  while(fixed){int g=__builtin_ctz(fixed);fixed&=fixed-1;v.push_back(g);}while(a--)v.push_back(nh++);while(b--)v.push_back(nt++);sort(v.begin(),v.end());out.push_back({v[0],v[1],v[2]});}
 sort(out.begin(),out.end());return {out,fact[h]*fact[t]/den};
}
vector<Blocks> small_suborbits(const Blocks&bs,int m){
 map<vector<unsigned>,Blocks> reps;array<int,3>head{0,1,2};
 do{array<int,3>tail{m-3,m-2,m-1};do{Blocks b=bs;for(auto&B:b)for(int&g:B){if(g<3)g=head[g];else if(g>=m-3)g=tail[g-(m-3)];}auto H=orbit_info(b,m,2,3);reps.emplace(pkey(H.canonical),H.canonical);}while(next_permutation(tail.begin(),tail.end()));}while(next_permutation(head.begin(),head.end()));
 vector<Blocks> out;for(auto&kv:reps)out.push_back(kv.second);return out;
}
struct Totals {
 uint64_t calls=0,hits=0,safe=0,learned=0,retained=0,problems=0,own_first_failures=0,shift_first_failures=0,own_failures_in_suborbits=0,suborbits=0;
 int minN=INT_MAX,maxN=0;vector<int>witness;Blocks failedblocks;
 void add(const Cover&c){calls+=c.calls;hits+=c.hits;safe+=c.safe;learned+=c.learned;retained+=c.retained;problems++;minN=min(minN,c.N);maxN=max(maxN,c.N);}
 bool attempt(const Blocks&b,const string&family){Cover c(b,family);bool ok=c.run();add(c);if(!ok){witness=c.witness;failedblocks=b;}return ok;}
 bool cover_orbit(const Blocks&b,int m){
  if(attempt(b,"own_first"))return true;own_first_failures++;
  if(attempt(b,"shift1_first"))return true;shift_first_failures++;
  auto hs=small_suborbits(b,m);uint64_t weights=0;
  for(auto&H:hs){suborbits++;weights+=orbit_info(H,m,2,3).weight;if(attempt(H,"own"))continue;own_failures_in_suborbits++;if(!attempt(H,"shift1"))return false;}
  if(weights!=orbit_info(b,m,3,3).weight)throw runtime_error("Suborbit weights do not sum to the parent orbit");return true;
 }
};
int main(int argc,char**argv){try{
 int m=15,first=0,last=-1;string rec="";bool single=false;int singleindex=-1;
 for(int i=1;i<argc;i++){string s=argv[i];auto val=[&](){if(i+1>=argc)throw runtime_error("Missing argument");return string(argv[++i]);};
 if(s=="--m")m=stoi(val());else if(s=="--first")first=stoi(val());else if(s=="--last")last=stoi(val());else if(s=="--records")rec=val();else if(s=="--single-index"){single=true;singleindex=stoi(val());}else throw runtime_error("Unknown option "+s);}
 const int totals[]={0,1,10,280,15400,1401400,190590400};if(m<6||m>18||m%3)throw runtime_error("m must be 6,9,12,15,18");if(last<0)last=totals[m/3];if(single){first=singleindex;last=singleindex+1;}if(first<0||last<first||last>totals[m/3])throw runtime_error("Invalid partition range");
 ofstream records;if(!rec.empty()){records.open(rec);if(!records)throw runtime_error("Cannot open record file");}
 int index=0,checked=0,failidx=-1;uint64_t represented=0,maxcalls=0;array<uint64_t,37> hist{};Blocks b;Totals total;bool failed=false;auto start=chrono::steady_clock::now();
 function<void(unsigned)>parts=[&](unsigned mask){if(index>=last||failed)return;if(mask){int a=__builtin_ctz(mask);unsigned rest=mask^(1u<<a);for(int x=a+1;x<m;x++)if(rest>>x&1)for(int y=x+1;y<m;y++)if(rest>>y&1){b.push_back({a,x,y});parts(rest^(1u<<x)^(1u<<y));b.pop_back();}return;}
 int idx=index++;if(idx<first)return;auto orb=orbit_info(b,m,3,3);if(!single&&pkey(b)!=pkey(orb.canonical))return;
 auto t=chrono::steady_clock::now();Totals c;bool ok=c.cover_orbit(b,m);double seconds=chrono::duration<double>(chrono::steady_clock::now()-t).count();
 checked++;represented+=orb.weight;hist[orb.weight]++;maxcalls=max(maxcalls,c.calls);
 total.calls+=c.calls;total.hits+=c.hits;total.safe+=c.safe;total.learned+=c.learned;total.retained+=c.retained;total.problems+=c.problems;total.own_first_failures+=c.own_first_failures;total.shift_first_failures+=c.shift_first_failures;total.own_failures_in_suborbits+=c.own_failures_in_suborbits;total.suborbits+=c.suborbits;total.minN=min(total.minN,c.minN);total.maxN=max(total.maxN,c.maxN);
 if(records)records<<idx<<","<<orb.weight<<","<<c.problems<<","<<c.calls<<","<<c.learned<<","<<c.retained<<","<<c.hits<<","<<c.safe<<","<<c.own_first_failures<<","<<c.shift_first_failures<<","<<c.suborbits<<","<<c.own_failures_in_suborbits<<","<<seconds<<"\n";
 if(!ok){failed=true;failidx=idx;total.failedblocks=c.failedblocks;total.witness=c.witness;return;}
 if(checked%1000==0)cerr<<"checked "<<checked<<" represented "<<represented<<" calls "<<total.calls<<" seconds "<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<"\n";
 };parts((1u<<m)-1);
 cout<<"{\n\"status\":\""<<(failed?"FAIL":"PASS")<<"\",\n\"m\":"<<m<<",\n\"family\":\"one_forward_shift\",\n\"method\":\"G36 first-triple-balanced families; H12 subdivision when necessary; learned exact covering subfamilies\",\n\"first_partition\":"<<first<<",\n\"last_partition_exclusive\":"<<last<<",\n\"G_orbits_checked\":"<<checked<<",\n\"partitions_represented\":"<<represented<<",\n\"coverage_problems\":"<<total.problems<<",\n\"recursive_calls\":"<<total.calls<<",\n\"learned_cores\":"<<total.learned<<",\n\"retained_cores\":"<<total.retained<<",\n\"subsumption_hits\":"<<total.hits<<",\n\"safe_prunes\":"<<total.safe<<",\n\"own_first_failures\":"<<total.own_first_failures<<",\n\"shift_first_failures\":"<<total.shift_first_failures<<",\n\"H_suborbits_checked\":"<<total.suborbits<<",\n\"own_failures_in_suborbits\":"<<total.own_failures_in_suborbits<<",\n\"min_candidates_over_attempts\":"<<(checked?total.minN:0)<<",\n\"max_candidates_over_attempts\":"<<total.maxN<<",\n\"max_calls_per_G_orbit\":"<<maxcalls<<",\n\"seconds\":"<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<",\n\"orbit_histogram\":{";
 bool comma=false;for(int i=1;i<=36;i++)if(hist[i]){if(comma)cout<<",";comma=true;cout<<"\""<<i<<"\":"<<hist[i];}cout<<"}";
 if(failed){cout<<",\n\"failed_G_representative_index\":"<<failidx<<",\n\"failed_blocks\":[";for(size_t i=0;i<total.failedblocks.size();i++){if(i)cout<<",";auto t=total.failedblocks[i];cout<<"["<<t[0]<<","<<t[1]<<","<<t[2]<<"]";}cout<<"],\n\"uncovered_prefix\":[";for(size_t i=0;i<total.witness.size();i++){if(i)cout<<",";cout<<total.witness[i];}cout<<"]";}
 cout<<"\n}\n";return failed?1:0;
 }catch(const exception&e){cerr<<"ERROR: "<<e.what()<<"\n";return 2;}}
