// Exhaustive, independent audit of all canonical ordered 3-CNF formulas with
// k <= 3 clauses.  Unused variables are omitted; variable names are normalized
// by first occurrence. Literal and clause order and all signs are retained.
// No SAT solver and no cycle-feasibility oracle are used to filter allocations.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using U=uint64_t;
struct Record {int k=0; U patterns=0,formulas=0,raw=0,survivors=0,comparisons=0,sat=0,unsat=0,completions=0;double seconds=0;};
struct Candidate {U X; unsigned literals; bool anchor;};
void require(bool b,const char* text){if(!b)throw std::runtime_error(text);}

static bool direct_ok(const std::vector<int>& order,U X,U S,int who) {
 int c[3]={0,0,0};
 for(int g:order) {
  int owner=(S>>g)&1?1:((X>>g)&1?2:0);++c[owner];
  if(c[0]>c[who]+1||c[1]>c[who]+1||c[2]>c[who]+1)return false;
 }
 return true;
}
void audit_pattern(const std::vector<int>& ids,Record& rec) {
 const int k=rec.k,L=3*k,v=L?*std::max_element(ids.begin(),ids.end())+1:0;
 const int q=4*k+2,m=3*q;
 std::vector<std::vector<int>> occurrences(v);
 for(int t=0;t<L;++t)occurrences[ids[t]].push_back(t);
 std::vector<int> A(L),B(L),aa(k+2),bb(k+2),order1;U S=0;
 std::vector<U> variable_A(v,0),variable_B(v,0);U anchorA=0,anchorB=0;
 int offset=0;
 for(int group=0;group<=v;++group){
  const int d=group<v?int(occurrences[group].size()):k+2;
  for(int j=0;j<d;++j){
   const int a=3*(offset+(j+d-1)%d)+1,b=3*(offset+j),z=3*(offset+j)+2;
   order1.push_back(a);order1.push_back(b);order1.push_back(z);S|=U(1)<<z;
   if(group<v){int t=occurrences[group][j];A[t]=a;B[t]=b;variable_A[group]|=U(1)<<a;variable_B[group]|=U(1)<<b;}
   else{aa[j]=a;bb[j]=b;anchorA|=U(1)<<a;anchorB|=U(1)<<b;}
  }offset+=d;
 }
 require(offset==q,"q mismatch");
 U seen=0;for(int g:order1){require(g>=0&&g<m&&!((seen>>g)&1),"sigma1 is not a permutation");seen|=U(1)<<g;}
 require(seen==((U(1)<<m)-1),"missing good");
 for(int j=0;j<q;++j)require((S>>order1[3*j+2])&1,"not reset");
 std::vector<U> survivors;
 for(unsigned choice=0;choice<(1u<<q);++choice){
  U X=0;for(int j=0;j<q;++j)X|=U(1)<<(3*j+((choice>>j)&1));
  ++rec.raw;if(direct_ok(order1,X,S,1))survivors.push_back(X);
 }
 std::sort(survivors.begin(),survivors.end());
 std::vector<Candidate> candidates;std::vector<U> expected;
 for(unsigned assignment=0;assignment<(1u<<v);++assignment){
  U X0=0;for(int u=0;u<v;++u)X0|=(assignment>>u)&1?variable_A[u]:variable_B[u];
  unsigned lit=0;for(int t=0;t<L;++t)if((assignment>>ids[t])&1)lit|=1u<<t;
  for(int anchor=0;anchor<2;++anchor){U X=X0|(anchor?anchorA:anchorB);candidates.push_back({X,lit,bool(anchor)});expected.push_back(X);}
 }
 std::sort(expected.begin(),expected.end());require(survivors==expected,"raw agent-1 family disagrees with cycle claim");
 rec.survivors+=survivors.size();
 for(unsigned signs=0;signs<(1u<<L);++signs){
  std::vector<int> order2;order2.reserve(m);order2.push_back(aa[0]);order2.push_back(aa[1]);
  for(int j=0;j<k;++j){
   for(int h=0;h<3;++h){int t=3*j+h;order2.push_back((signs>>t)&1?B[t]:A[t]);}
   order2.push_back(bb[j]);
   for(int h=0;h<3;++h){int t=3*j+h;order2.push_back((signs>>t)&1?A[t]:B[t]);}
   order2.push_back(aa[j+2]);
  }
  order2.push_back(bb[k]);order2.push_back(bb[k+1]);
  for(int j=0;j<q;++j)order2.push_back(3*j+2);
  require(int(order2.size())==m,"sigma2 length mismatch");seen=0;
  for(int t=0;t<m;++t){int g=order2[t];require(g>=0&&g<m&&!((seen>>g)&1),"sigma2 is not a permutation");seen|=U(1)<<g;require(bool((S>>g)&1)==(t>=2*q),"not a suffix");}
  U solutions=0,satisfying=0;
  for(const Candidate& c:candidates){
   bool sat=true;unsigned truth=c.literals^signs;
   for(int j=0;j<k;++j)if(((truth>>(3*j))&7u)==0){sat=false;break;}
   bool expected_ok=c.anchor&&sat;
   bool ok=direct_ok(order2,c.X,S,2);
   ++rec.comparisons;
   if(ok!=expected_ok){std::cerr<<"k="<<k<<" signs="<<signs<<" X="<<c.X<<" ids=";for(int x:ids)std::cerr<<x<<',';std::cerr<<'\n';throw std::runtime_error("SAT iff failed");}
   solutions+=ok;satisfying+=expected_ok;
  }
  require(solutions==satisfying,"decision iff failed");
  ++rec.formulas;if(solutions)++rec.sat;else++rec.unsat;rec.completions+=solutions;
 }
 ++rec.patterns;
 if(k==3 && rec.patterns%1000==0)std::cerr<<"k=3 patterns="<<rec.patterns<<" formulas="<<rec.formulas<<'\n';
}
void rgs(std::vector<int>& ids,int p,int hi,Record& rec){
 if(p==int(ids.size())){audit_pattern(ids,rec);return;}
 for(int u=0;u<=hi+1;++u){ids[p]=u;rgs(ids,p+1,std::max(hi,u),rec);}
}
int main(int argc,char**argv){
 try{
  int maxk=argc>1?std::stoi(argv[1]):3;std::string out=argc>2?argv[2]:"audit_sat_results.json";
  require(maxk>=0&&maxk<=3,"maxk must be 0..3");std::vector<Record> records;
  for(int k=0;k<=maxk;++k){Record r;r.k=k;auto start=std::chrono::steady_clock::now();std::vector<int>ids(3*k);rgs(ids,0,-1,r);r.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();records.push_back(r);std::cerr<<"completed k="<<k<<" formulas="<<r.formulas<<" seconds="<<r.seconds<<'\n';}
  std::ofstream f(out);f<<"{\n\"status\":\"PASS\",\n\"universe\":\"Ordered three-literal clauses, variable names normalized by first occurrence; all signs, repeated literals and tautologies included; no unused variables.\",\n\"method\":\"Enumerate every raw residual-pair orientation for every unsigned variable pattern, check sigma1 directly, then enumerate every sign vector and directly check all surviving owners against every sigma2 prefix. Compare every candidate with direct truth-table evaluation.\",\n\"records\":[\n";
  for(size_t i=0;i<records.size();++i){auto&r=records[i];if(i)f<<",\n";f<<"{\"clauses\":"<<r.k<<",\"variable_patterns\":"<<r.patterns<<",\"canonical_formulas\":"<<r.formulas<<",\"raw_pair_orientations\":"<<r.raw<<",\"agent1_survivors\":"<<r.survivors<<",\"signed_candidate_comparisons\":"<<r.comparisons<<",\"satisfiable_formulas\":"<<r.sat<<",\"unsatisfiable_formulas\":"<<r.unsat<<",\"fair_completions_total\":"<<r.completions<<",\"seconds\":"<<r.seconds<<"}";}
  f<<"\n]}\n";std::cout<<out<<'\n';
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
