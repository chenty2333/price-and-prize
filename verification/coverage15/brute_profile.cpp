// Independent brute-force profile checker. Input: m, then sigma1, then sigma2.
// Enumerates every market-balanced allocation; no matching or coverage routines.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <vector>
using namespace std;
int main(){int m;if(!(cin>>m)||m<0||m>45)return 2;vector<vector<int>>rank(2,vector<int>(m));for(auto&s:rank)for(int&g:s)cin>>g;
 for(auto s:rank){sort(s.begin(),s.end());for(int g=0;g<m;g++)if(s[g]!=g)return 2;}
 uint64_t total=0,fair=0,own=0,fairown=0;map<int,uint64_t>shifts;vector<int>owner(m),example;
 auto start=chrono::steady_clock::now();
 function<void(int)>search=[&](int b){if(b==m){total++;bool ob=true;
 for(int t=0;t<m;t+=3){int c=0;for(int j=t;j<min(m,t+3);j++)c+=owner[rank[0][j]]==1;if((t+3<=m&&c!=1)||(t+3>m&&c>1)){ob=false;break;}}
 own+=ob;
 for(int i=0;i<2;i++){int count[3]={};for(int g:rank[i]){count[owner[g]]++;for(int j=0;j<3;j++)if(count[j]>count[i+1]+1)return;}}
 fair++;fairown+=ob;if(example.empty())example=owner;
 if(m%3==0){int u=0,height=0;for(int t=0;t<m;t+=3){int c=0;for(int j=t;j<t+3;j++)c+=owner[rank[0][j]]==1;u+=max(0,c-1);height+=c-1;if(height<0)throw 1;}shifts[u]++;}
 return;}
 int len=min(3,m-b);array<int,3>p{0,1,2};do{for(int j=0;j<len;j++)owner[b+j]=p[j];search(b+len);}while(next_permutation(p.begin(),p.end()));
 // A short block has repeated completions of its injection in the permutations;
 // correct multiplicities in the reported counts at the end.
 };search(0);int divisor=m%3==1?2:1;total/=divisor;fair/=divisor;own/=divisor;fairown/=divisor;
 cout<<"{\"m\":"<<m<<",\"market_allocations\":"<<total<<",\"own_balanced_allocations\":"<<own<<",\"fair_allocations\":"<<fair<<",\"fair_own_balanced_allocations\":"<<fairown<<",\"shift_histogram\":{";bool first=true;for(auto[k,v]:shifts){if(!first)cout<<",";first=false;cout<<"\""<<k<<"\":"<<v;}cout<<"},\"fair_example\":[";for(int i=0;i<(int)example.size();i++){if(i)cout<<",";cout<<example[i];}cout<<"],\"seconds\":"<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<"}\n";
}
