#!/usr/bin/env bash
# Rerun the checks in src/ (Python 3.10+, numpy, scipy; a C++17 compiler for
# cover15_ours). Outputs go to ../results/. Randomized searches use fixed seeds
# but time budgets, so their instance counts depend on machine speed.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}

# Two agents
$PY verify_monotone_ours.py          # reversal path vs 16,000 monotone valuations; ten-good obstruction
$PY verify_logquery.py               # O(log m) query algorithm
$PY n2_constructive.py 120           # additive construction and checker
$PY n2_lemmaB_adversarial.py 8,10,12,14 120
$PY n2_constructive_hard.py          # construction on hard profiles

# Three agents, market-aligned setting
$PY conjB_check.py                   # exhaustive existence check, m <= 8
$PY verify_thirteen.py               # thirteen-good own-balance failure (206 fair, 0 own-balanced)
$PY verify_repair_onesingleton.py    # Theorem 14 family (r = 2, 3, 4) and Theorem 7 rule
g++ -O3 -march=native -std=c++17 cover15_ours.cpp -o cover15_ours
# Full fifteen-good run (about 70 CPU-hours, about 2 GB per worker):
#   for k in $(seq 0 8); do ./cover15_ours $k 9 1401400 orbits > ../results/cover15_ours/worker_$k.json & done; wait

# General three agents: exhaustive small cases and exact decisions
$PY milp_sanity.py
$PY oq1_probe.py 8
for nm in "2 7" "2 8" "3 5" "3 6" "3 7"; do $PY exhaustive.py $nm; done   # n = 3, m = 7: all 21,350,046,480 profiles
$PY n3_search.py --budget 300
$PY cegar_validate.py                # decision procedure vs enumeration; EFX positive control
$PY n3_cegar_search.py --ms 13,14,15,16,17,18,20,21 --per_m 2 --budget 2400
$PY redecide_undecided.py
$PY verify_certificates.py           # LP-only re-verification of refuting sets
$PY n3_certificates.py
$PY n3_gluing.py search 1200         # concatenated hard profiles (up to 36 goods)
$PY n3_gluing_ls.py 1200
$PY redecide_robust.py
$PY disjoint_defect_test.py          # disjoint-defect pairs on saved hard profiles
$PY two_cover_test.py
$PY verify_defect_counterexample.py  # twelve-good profile without disjoint-defect pairs
