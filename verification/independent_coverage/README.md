# Second, separately written coverage checkers

Setting: three agents, `sigma_0 = pi`, agent 1's ranking normalized to the identity, market triples an arbitrary unordered
partition `Q`. For every partition, the recursion on states `(T, L)` (prefix set, candidates that survived agent 2's earlier
prefix constraints) decides whether every ranking of agent 2 is accepted by some candidate.

* `src/cover_ours_general.cpp` — twelve goods (also usable for other sizes with `-DMGOODS=`), no symmetry, no covering cores.

  ```sh
  g++ -O2 -std=c++17 -DMGOODS=12 -o cover12 src/cover_ours_general.cpp
  OWN=1 ./cover12 0 1                 # own-bundle-balanced candidates (the m ≤ 12 theorem): 0 uncovered, ~1 minute
  ./cover12 0 1                       # all market-balanced candidates: 0 uncovered
  OWN=1 NEGCTRL=1 ./cover12 0 1       # negative control (all three bundles rainbow on identity triples): 12,150 uncovered
  ```
  Recorded outputs: `results/cover12_ours/`. The run with `OWN=1` reproduces the number of memoized covered states (43,515,662) and
  the candidate range (64 to 1,296) of `verification/coverage12`.

* `src/cover15_ours.cpp` — fifteen goods over all market-balanced candidates, only the order-12 symmetry group (swap goods 0, 1;
  permute goods 12, 13, 14; proof in the file header), 194,250 orbit representatives. Options `MEMO_CAP`, `RESUME_FROM`, `NEGCTRL`.

  ```sh
  g++ -O2 -std=c++17 -march=native -o cover15 src/cover15_ours.cpp
  MEMO_CAP=1500000 ./cover15 <worker> <nworkers> 1401400 orbits   # workers in parallel; about 50 CPU-hours, ~1.5 GB each
  ```
  The per-worker JSON lines report `uncovered` and the sum of orbit weights, which must add up to 1,401,400.

`src/summarize_cover15.py` prints the progress and the final verdict of a fifteen-good run from `results/cover15_ours/` (complete, no uncovered partition, orbit weights summing to 1,401,400).
