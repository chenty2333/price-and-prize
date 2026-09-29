# Twelve-good coverage with own-bundle balance (Theorem 5, m ≤ 12)

For three agents with agent 0's ranking equal to the market order, every
profile with at most twelve goods has a market-block-balanced, subjectively
SD-EF1 allocation in which agent 1 also receives exactly one good of each of
its own ranking triples. Smaller instances reduce to twelve goods by padding
with universally bottom-ranked goods. The code names this setting
"Conjecture B".

## Method

Normalize agent 1's ranking to the identity. An instance is then an unordered
partition of the goods into four market triples (15,400 partitions). For each
partition, the candidate family contains the market-balanced allocations
(at most 6^4 = 1,296) that satisfy agent 1's constraints and own-bundle balance.
The verifier walks every prefix of agent 2's ranking. It keeps the exact state
(prefix set, surviving candidates) and accepts only when every one-good
extension is covered.

| Quantity | Value |
|---|---:|
| Partitions | 15,400 (all covered) |
| Recursive calls | 70,831,414 |
| Memoized covered states | 43,515,662 |
| Candidate family size | 64–1,296 |

## Reproduction

```sh
python run_verification.py --workers 4 --output rerun12.json      # C++17 compiler needed
python verify_12.py --samples 40 --crosscheck ./verify_12 --output python_rerun.json
python verify_12.py --all --output python_full.json               # optional, slow
python run_tests.py --general-per-m 1000 --output tests_rerun.json
```

`run_verification.py` compiles `verify_12.cpp` and checks that the partition
ranges cover 0..15,400 without gaps. A partial range is not a verification.

## Files

* `verify_12.cpp`: exhaustive verifier. Memoization uses full state equality.
  Its complete output is in `verification_12.json` and `verification_chunks/`.
* `verify_12.py`: separate readable Python implementation. It agrees with the
  C++ verifier on 40 complete partitions (`python_verifier_results.json`).
* `negative_control.json`: requiring every bundle to be balanced in agent 1's
  ranking exposes an uncovered prefix, as expected.
* `round2.py`: direct prefix checker, constructive solver through twelve goods,
  and the common-module solver (Corollary 6). `run_tests.py` holds the
  construction tests (`test_results.json`).
