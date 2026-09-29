# Repair distance, one-singleton rule, reduction audit (Sections 4–6)

* **Theorem 14** (repair distance): `connected_family(r)` builds the m = 6r
  profile. Its own-bundle-balanced repair distance is exactly r, and one
  replacement suffices without own-bundle balance.
* **Theorem 7** (one-singleton rule): `solve_one_singleton(profile)`, plus a
  witness generator for the converse.
* **Exhaustive reduction audit** for Theorem 11: `audit_sat.cpp` enumerates all
  10,840,297 canonical three-literal formulas with at most three clauses. For
  each it checks every residual orientation against the original prefix
  conditions and compares the outcome with the truth table.

```sh
python run_tests.py
c++ -O3 -std=c++17 audit_sat.cpp -o audit_sat && ./audit_sat 3 audit_sat_results.json
python audit_reference.py      # cross-check against the Section 5 encoder in prior/round3
```

Python code uses only the standard library. `independent_checker.py` checks
the original definitions directly. Distance from a prescribed bundle S is
`|S \ A_1|`. `prior/round3` is a copy of `../completion`, imported by the audit.

```python
from round4 import solve_one_singleton, connected_family
from independent_checker import check, own_balanced
example = connected_family(4)                     # m = 24
assert check(example['profile'], example['own_owner']) is None
assert own_balanced(example['profile'], example['own_owner'])
```

`solve_one_singleton` returns None when the ranking condition fails. That does
not mean no fair allocation exists.
