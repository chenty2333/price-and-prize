# Prescribed completion and exchange repair (Sections 5–6)

Code for the alternating-cycle description (Lemma 10), the 3-SAT reduction
(Theorem 11), the twelve-good integrality gap, the parallel-edge exchange
(Lemma 12) and the one-exchange repair of the reduction family (Theorem 13).
Python 3.10+, standard library only.

```sh
python run_tests.py            # full suites (11,300 generated inputs)
python run_tests.py --reset-per-q 2 --formula-count 5 --small-formula-count 6 \
    --exchange-count 20 --output smoke_results.json   # quick check
```

Main interfaces in `round3.py`:

* `solve_reset_matching(profile, S)`: exact for reset matchings, O(m 2^c).
* `all_fixed_candidates(profile, S)`: all residual orientations for any matching.
* `fractional_check(profile, S, weights)`: the fixed-matching LP, in exact
  rational arithmetic.
* `encode_3cnf(clauses)`: the reduction, returning `(profile, S, metadata)`.
* `repair_parallel_edge`, `repair_one_unit`, `try_cycle_repair`,
  `solve_encoded_family`: the exchange lemma, its suffix corollary, and the
  rankings-only repair.

A profile is `[sigma_0, sigma_1, sigma_2]` with `sigma_0 == list(range(m))`,
and an allocation is an `owner[g]` array with values 0, 1, 2.
`independent_checker.py` evaluates the original prefix and block conditions,
not the derived constraints or SAT semantics. Recorded results are in
`test_results.json`.
