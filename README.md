# Subjective SD-EF1 and Market EF1

Code, data, and paper sources for *Subjective SD-EF1 and Market EF1: Two-Agent
Existence and Barriers for Three Agents* (anonymous submission).

The paper studies Open Question 1 of Barman, Ebadian, Latifian and Shah,
*Fair Division with Market Values*. The question asks whether indivisible goods
can always be allocated so that every agent is SD-EF1 with respect to its own
ranking, while the allocation is EF1 with respect to a common market valuation.

## Where each result is checked

| Result (paper) | Status | Code |
|---|---|---|
| Two agents, any monotone market (Thm 1, Cor 4) | proof | `verification/monotone_and_selection`, `verification/queries_and_defects`, `src/verify_monotone_ours.py`, `src/verify_logquery.py` |
| Three agents, market-aligned, m ≤ 12 with own-bundle balance (Thm 5) | computer-assisted | `verification/coverage12` |
| Three agents, market-aligned, m ≤ 15 (Thm 5) | computer-assisted | `verification/coverage15`, `src/cover15_ours.cpp` |
| Own-bundle balance fails at 13 goods | proof + enumeration | `verification/coverage15`, `src/verify_thirteen.py` |
| One-singleton rule, all m (Thm 7) | proof | `verification/repair`, `src/verify_repair_onesingleton.py` |
| Prescribed completion is NP-complete (Thm 11) | proof + exhaustive audit | `verification/completion`, `verification/repair` |
| Exchange repair of the reduction family (Lemma 12, Thm 13) | proof | `verification/completion` |
| Ω(m) repair distance (Thm 14) | proof | `verification/repair`, `src/verify_repair_onesingleton.py` |
| Disjoint-defect pairs need not exist (12 goods) | enumeration | `verification/queries_and_defects`, `src/verify_defect_counterexample.py` |
| Pair-selection rules fail (10 goods) | enumeration | `verification/monotone_and_selection` |
| General three agents: no counterexample found | computational evidence | `src/` (exact decision procedure), `results/` |

The analytic results rely on the written proofs; the tests check them and their
implementations. The finite existence theorem also relies on exhaustive
computation. Nothing here is formally verified.

## Layout

```
paper/          LaTeX sources and compiled PDFs (main paper, technical appendix)
src/            decision procedure, searches, and separately written checkers
results/        recorded outputs of src/
verification/   one directory per result, each with its own README
  coverage12/              m ≤ 12 coverage with own-bundle balance
  coverage15/              m ≤ 15 coverage; thirteen-good failure; repeated modules
  completion/              cycle description, 3-SAT reduction, exchange repair
  repair/                  repair distance, one-singleton rule, exhaustive reduction audit
  monotone_and_selection/  two-agent reversal path; pair-selection obstructions
  queries_and_defects/     O(log m) query algorithm; twelve-good disjoint-defect counterexample
```

Allocations are `owner[g]` arrays. A three-agent profile is
`[sigma_0, sigma_1, sigma_2]`, and in the market-aligned setting
`sigma_0 = list(range(m))`. The code calls this setting "Conjecture B".

## Running

Python 3.10+ and a C++17 compiler. Most verification scripts need only the
standard library. The decision procedure in `src/` uses NumPy and SciPy (HiGHS),
and two targeted searches use Numba.

```sh
pip install -r requirements.txt
python verification/coverage15/run_verification.py --m 15 --workers 4 --output-dir rerun15
bash src/run_all.sh
bash paper/build.sh
```

The fifteen-good coverage takes about two CPU-hours. Each directory's README
lists its own commands and recorded outputs. Randomized searches use fixed
seeds but time budgets, so their instance counts depend on machine speed.

## Independent checks

Where two implementations exist, they were written separately and agree.

* The fifteen-good coverage run was reproduced on a second machine with
  identical deterministic counters and per-orbit records
  (`verification/coverage15/reproduced`).
* `src/cover15_ours.cpp` checks the same theorem over all market-balanced
  candidates, without learned cores and with a smaller, separately proved
  symmetry group.
* The `src/verify_*.py` scripts re-derive the counts of the thirteen-good,
  repair-distance, one-singleton and twelve-good results with a separate
  checker (`src/fd.py`).
