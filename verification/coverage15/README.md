# Fifteen-good coverage (Theorem 5, m ≤ 15)

Computer-assisted part of Theorem 5: with three agents and agent 0's ranking
equal to the market order, every profile with at most fifteen goods has a
market-block-balanced, subjectively SD-EF1 allocation. The code names this
setting "Conjecture B". Also here: the thirteen-good failure of own-bundle
balance, and the repeated-module family showing that no constant number of
shifts suffices.

## What is certified

After normalizing agent 1's ranking to the identity, an instance is determined
by the unordered partition of the fifteen goods into market triples (1,401,400
partitions). For each partition, `verify_cover.cpp` proves that for **every**
ranking of agent 2 some candidate allocation is fair. It uses exact
prefix-state recursion with reusable covering cores and two proved endpoint
symmetries (82,670 orbits). `COVERAGE.md` gives the soundness argument for the
normalization, the symmetries, the core pruning and the candidate hierarchy
(own-bundle-balanced candidates first, one-shift candidates as fallback).

| Quantity | Value |
|---|---:|
| Partitions / orbits | 1,401,400 / 82,670 |
| Coverage attempts | 101,840 |
| Recursive calls | 15,937,540,828 |
| Candidate family size | 80–7,776 |

`coverage15/` holds per-orbit CSV records and per-shard JSON summaries. The
aggregate record is `coverage15/verification.json`. Build paths in that file
are redacted.

## Reproduction

C++17 compiler and Python 3.10+ (standard library only).

```sh
python run_verification.py --m 15 --workers 4 --output-dir rerun15   # ~2 CPU-hours
python audit_logs.py --directory rerun15 --output rerun_audit.json
```

`reproduced/` records a rerun on a second machine (10 workers, 739 s wall).
Every deterministic counter and all 82,670 per-orbit records are identical to
the recorded run. `audit_logs.py` independently reconstructs every orbit and
its weight, and checks that the weights sum to 1,401,400. It audits accounting,
not the recursive coverage inferences.

Independent checks of the coverage logic:

* `verify_python.py`: a separate full-state checker without cores or
  symmetry, run on ten complete partitions (`python_spot_checks.json`).
  `verify_reference.cpp` is its C++ comparison program.
* `test_symmetry.py`: explicit orbit and candidate-invariance tests, including
  a negative control for an invalid larger group.
* `../../src/cover15_ours.cpp`: a separately written checker over **all**
  market-balanced candidates. It uses no cores and only a 12-element symmetry
  proved in its header. See the top-level README for its status.

## Other files

* `round5.py`: thirteen-good counterexample, repeated-module construction,
  shift-budget search, finite solver, common-module solver.
* `independent_checker.py`, `brute_profile.cpp`, `run_bruteforce.py`: direct
  checks of the original prefix and block conditions.
* `strong_counterexample13.json`, `own_failure_reproduction.json`: the
  thirteen-good profile (206 fair market-balanced allocations, none
  own-balanced).
* `repeated_module30*`: the 30-good two-module profile (169,744 fair
  allocations out of 60,466,176, all with shift budget two).
* `verification_12_hierarchical.json`: this verifier re-run on all 15,400
  twelve-good partitions.
* `run_tests.py`, `test_negative_controls.py`: construction tests and negative
  controls; `test_results.json`, `negative_controls.json` hold their outputs.

Successes of heuristic or restricted searches above fifteen goods do not
extend the theorem.
