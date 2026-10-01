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

* `src/cover15_ours.cpp` — an analogous checker for fifteen goods over all market-balanced candidates (order-12 symmetry group, proof in the file header). It needs about 50 CPU-hours and was not run to completion; the paper does not use it.
