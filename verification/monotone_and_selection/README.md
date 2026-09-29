# Two-agent reversal path and pair-selection obstructions

**Two agents (Theorem 1, Corollary 4, mixed items).** `monotone_two.py`
implements the ranking-only SD-EF1 reversal path for goods, chores, and mixed
items with a common sign set. Given an inclusion-monotone market oracle, it
selects a market-EF1 state. Audits, all in exact integer arithmetic:

* `test_monotone.py`: all normalized ranking pairs for m ≤ 8 (46,233
  profiles); all 168 monotone Boolean functions on four goods; 9,000 random
  goods/chores/mixed instances with full allocation enumeration.
  Output: `test_monotone.log`, `test_monotone_counts.json`.
* `test_supplement.py`: 3,000 instances forcing crossings between the two
  failure regions (`test_supplement.log`).
* `verify_sign_mismatch.py`: two items on which existence fails when agents
  and market disagree about signs.

**Three agents: pair-selection rules (technical appendix).**
`verify_joint_selection_obstruction.py` and `verify_short.py` enumerate all 3^10
allocations of a ten-good profile. Selecting the lexicographically earliest
defect set, the maximum subjective slack, or two minimum-defect witnesses
never yields a disjoint-defect pair. `verify_slack_obstruction.py` refutes
maximum-slack selection on a second profile. `pair_selection.py` analyses the
selection rules on 146 small hard profiles (`small_hard_profiles.json`), and
`release_audit.py` re-checks every saved pair certificate. `search_lex_rule.py`
is the targeted search that found the ten-good profile. It is not needed to
check any certificate.

Standard library suffices for the two-agent tests and the standalone
obstruction verifiers. `pair_selection.py` and `search_lex_rule.py` need NumPy
(`requirements.txt`; versions in `environment.txt`).
