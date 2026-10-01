# Subjective SD-EF1 and Market EF1

Code, data, and paper sources for *Subjective SD-EF1 and Market EF1: Two-Agent
Existence and Barriers for Three Agents* (anonymous submission).

The paper studies Open Question 1 of Barman, Ebadian, Latifian and Shah,
*Fair Division with Market Values*. The question asks whether indivisible goods
can always be allocated so that every agent is SD-EF1 with respect to its own
ranking, while the allocation is EF1 with respect to a common market valuation.

## Where each result is checked

Numbers refer to the main paper.

| Result | Status | Code |
|---|---|---|
| Two agents, any monotone market (Thm 3.1, chores, mixed items) | proof | `verification/monotone_and_selection`, `verification/queries_and_defects`, `src/verify_monotone_ours.py`, `src/verify_logquery.py` |
| Any cover of the monotone valuations has Ω(√m) members (Prop 3.4) | proof; counting checked by machine | `src/audit_extra.py A` |
| Minimum cover size ⌈m/2⌉ (m = 7, 8); path is an exact cover with all members necessary (m = 9–11); two allocations for additive markets on 600 hard profiles | computation | `verification/covers_two_agents`, `src/n2_cover.py` |
| Three agents, market-aligned, m ≤ 12 with own-bundle balance (Thm 4.1) | computer-assisted, two implementations | `verification/coverage12`, `verification/independent_coverage` (`src/cover_ours_general.cpp`) |
| Three agents, market-aligned, m ≤ 15 (Thm 4.1) | computer-assisted | `verification/coverage15` |
| Own-bundle balance fails at 13 goods | proof + enumeration | `verification/coverage15`, `src/verify_thirteen.py`, `src/audit_extra.py B` |
| One-singleton rule, all m (Thm 4.2) | proof | `verification/repair`, `src/verify_repair_onesingleton.py` |
| One arbitrary ranking, block-contiguous others, any n (Appendix E) | proof | `src/verify_block_contiguous.py` |
| Prescribed completion is NP-complete (Thm 5.3) | proof + exhaustive audit | `verification/completion`, `verification/repair` |
| Exchange repair of the reduction family (Lemma 5.4, Thm 5.5) | proof | `verification/completion` |
| Ω(m) repair distance (Thm 5.6) | proof | `verification/repair`, `src/verify_repair_onesingleton.py` |
| Disjoint-defect pairs need not exist (12 goods; two agents, 11 goods) (Prop 5.7) | enumeration | `verification/queries_and_defects`, `src/verify_defect_counterexample.py`, `src/audit_extra.py C`, `src/n2_cover.py` |
| Pair-selection rules fail (10 goods) | enumeration | `verification/monotone_and_selection` |
| General three agents: no counterexample found | computational evidence | `src/` (exact decision procedure), `results/` |

The analytic results rely on the written proofs; the tests check them and their
implementations. The finite existence theorem also relies on exhaustive
computation. Nothing here is formally verified.

## Layout

```
paper/          LaTeX sources (AAMAS 2027 format) and compiled PDFs: main paper, technical appendix
src/            decision procedure, searches, and separately written checkers
results/        recorded outputs of src/
verification/   one directory per result, each with its own README
  coverage12/              m ≤ 12 coverage with own-bundle balance
  coverage15/              m ≤ 15 coverage; thirteen-good failure; repeated modules
  independent_coverage/    second, separately written coverage checkers (12 and 15 goods)
  covers_two_agents/       cover sizes for two agents (Section 3.1), audits of Prop 3.4 and of the 12-good profile
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
standard library. The decision procedure and the cover experiments use NumPy and
SciPy (HiGHS), and two targeted searches use Numba.

```sh
pip install -r requirements.txt
python verification/coverage15/run_verification.py --m 15 --workers 4 --output-dir rerun15
bash src/run_all.sh
cd src && python n2_cover.py all        # cover experiments E1-E5, about 8 minutes on 12 cores
cd src && python audit_extra.py all     # Prop 3.4, thirteen goods, twelve-good defects
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
* `src/cover_ours_general.cpp` re-checks the twelve-good theorem with a
  different traversal: all 15,400 partitions are covered, and the number of
  memoized states (43,515,662) and the candidate-family range (64 to 1,296)
  equal those of the shipped verifier (`results/cover12_ours`). The negative
  control leaves 12,150 partitions uncovered.
* The `src/verify_*.py` scripts re-derive the counts of the thirteen-good,
  repair-distance, one-singleton and twelve-good results with a separate
  checker (`src/fd.py`); `src/audit_extra.py` and `src/n2_cover.py` do so with
  code that does not import `fd.py` or `cegar.py`.
