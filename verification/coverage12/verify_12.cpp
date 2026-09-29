#include <bits/stdc++.h>
using namespace std;using U=uint64_t;
struct Cover {
 int m=12,q=4,w,N;vector<array<unsigned,3>> candidates;vector<array<int,3>> blocks;
 vector<U> valid,robust;unordered_set<string> memo;unsigned long long nodes=0,leaves=0,hits=0;vector<int>path,witness;
 bool own=true;
 Cover(vector<array<int,3>> bs,bool own_=true):blocks(bs),own(own_){
  unsigned masks[3]={};int perm[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
  function<void(int)>gen=[&](int b){if(b==q){if(own)for(int t=0;t<m;t+=3)if(__builtin_popcount(masks[1]&(7u<<t))!=1)return;for(int t=1;t<=m;t++){unsigned S=(1u<<t)-1;int x=__builtin_popcount(S&masks[1]);if(__builtin_popcount(S&masks[0])>x+1 || __builtin_popcount(S&masks[2])>x+1)return;}candidates.push_back({masks[0],masks[1],masks[2]});return;}for(auto &p:perm){for(int t=0;t<3;t++)masks[p[t]]|=1u<<blocks[b][t];gen(b+1);for(int t=0;t<3;t++)masks[p[t]]^=1u<<blocks[b][t];}};gen(0);
  N=candidates.size();w=(N+63)/64;valid.assign((1<<m)*w,0);robust.assign((1<<m)*w,0);
  for(unsigned S=0;S<(1u<<m);S++)for(int a=0;a<N;a++){
   int x=__builtin_popcount(S&candidates[a][2]);
   if(__builtin_popcount(S&candidates[a][0])<=x+1 && __builtin_popcount(S&candidates[a][1])<=x+1)valid[S*w+a/64]|=1ull<<(a%64);
   if(x>=q-1)robust[S*w+a/64]|=1ull<<(a%64);
  }
 }
 string key(unsigned S,const vector<U>&a){string s((char*)&S,sizeof S);s.append((char*)a.data(),w*sizeof(U));return s;}
 bool dfs(unsigned S,const vector<U>&a){
  nodes++;bool any=false;
  for(int j=0;j<w;j++){if(a[j]&robust[S*w+j]){leaves++;return false;}any|=a[j]!=0;}
  if(!any){witness=path;return true;}
  string k=key(S,a);if(memo.count(k)){hits++;return false;}
  struct Ch{int g,count;vector<U>a;};vector<Ch>chs;
  for(int g=0;g<m;g++)if(!(S>>g&1)){unsigned T=S|(1u<<g);Ch ch;ch.g=g;ch.count=0;ch.a.resize(w);bool safe=false;for(int j=0;j<w;j++){ch.a[j]=a[j]&valid[T*w+j];ch.count+=__builtin_popcountll(ch.a[j]);if(ch.a[j]&robust[T*w+j]){safe=true;break;}}if(!safe)chs.push_back(move(ch));else leaves++;}
  sort(chs.begin(),chs.end(),[](const Ch&x,const Ch&y){return x.count<y.count;});
  for(auto&ch:chs){path.push_back(ch.g);if(dfs(S|(1u<<ch.g),ch.a))return true;path.pop_back();}
  memo.insert(move(k));return false;
 }
 bool run(){vector<U>a(w,~0ull);if(N%64)a.back()=(1ull<<(N%64))-1;return dfs(0,a);}
};
int main(int argc,char**argv){
 bool own=argc>1?atoi(argv[1]):true; int first=argc>2?atoi(argv[2]):0,last=argc>3?atoi(argv[3]):15400,index=0;
 unsigned long long totalnodes=0,totalmemo=0,totalhits=0,totalsafe=0;int total=0,mincand=INT_MAX,maxcand=0;unsigned long long maxnodes=0;
 auto start=chrono::steady_clock::now();vector<array<int,3>>b;
 function<void(unsigned)> parts=[&](unsigned mask){
  if(mask){int a=__builtin_ctz(mask);unsigned rest=mask^(1u<<a);
   for(int x=a+1;x<12;x++)if(rest>>x&1)for(int y=x+1;y<12;y++)if(rest>>y&1){b.push_back({a,x,y});parts(rest^(1u<<x)^(1u<<y));b.pop_back();}return;
  }
  if(index<first || index>=last){index++;return;} index++; Cover c(b,own);bool fail=c.run();total++;totalnodes+=c.nodes;totalmemo+=c.memo.size();totalhits+=c.hits;totalsafe+=c.leaves;mincand=min(mincand,c.N);maxcand=max(maxcand,c.N);maxnodes=max(maxnodes,c.nodes);
  if(fail){cout<<"FAIL\n";for(auto x:b)cout<<x[0]<<","<<x[1]<<","<<x[2]<<";";cout<<"\n";for(int x:c.witness)cout<<x<<",";cout<<"\n";exit(1);}
  if(total%1000==0)cerr<<"checked "<<total<<" nodes "<<totalnodes<<" seconds "<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<"\n";
 };
 parts((1u<<12)-1);
 cout<<"{\n  \"status\": \"PASS\",\n  \"first_partition\": "<<first<<",\n  \"last_partition_exclusive\": "<<last<<",\n  \"own_balance\": "<<(own?"true":"false")<<",\n  \"partitions\": "<<total<<",\n  \"nodes\": "<<totalnodes<<",\n  \"memo_states\": "<<totalmemo<<",\n  \"memo_hits\": "<<totalhits<<",\n  \"safe_prunes\": "<<totalsafe<<",\n  \"min_candidates\": "<<mincand<<",\n  \"max_candidates\": "<<maxcand<<",\n  \"max_nodes_per_partition\": "<<maxnodes<<",\n  \"seconds\": "<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<"\n}\n";
}
