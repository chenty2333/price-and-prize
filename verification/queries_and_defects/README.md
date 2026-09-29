# Logarithmic queries and the twelve-good disjoint-defect counterexample

**Two agents: query algorithm (Theorem 1).** `ordinal_market.py` represents
the reversal path implicitly. It finds a market-EF1 state with at most
3 + 2⌈log₂⌈m/2⌉⌉ value queries (sign bisection plus one repair query), for
goods, chores, and common-sign mixed items. `exact1_circle.py` implements the
Exact1 moving-knife algorithm of Goldberg et al. used in the related-work
comparison.

```sh
python test_algorithms.py   # all ranking profiles m ≤ 8; 2,000 each of goods,
                            # chores, mixed; 2,000 Exact1 instances (test_counts.json)
python verify_task2.py      # small examples: heterogeneous allocator functions can jump
```

**Three agents: disjoint-defect pairs need not exist.** A twelve-good profile
has 1,080 subjectively SD-EF1 allocations, 58 of them near-block-balanced,
and no pair whose market defects lie at disjoint triple boundaries. It still
has a two-allocation cover of another form (`unrestricted_cover_12.json`).

```sh
python verify_twelve.py        # enumerates all 3^12 allocations; checks the cover
python verify_no_disjoint.py   # separate full enumeration
```

Both use only the standard library. `search_min2.py` (NumPy, Numba) is the
targeted search that found the profile. `check_unrestricted.py` found the
cover, and `refine_no_disjoint.py` tests its neighbours. `prior/` holds helper
modules these scripts import. The logs and JSON files are their recorded
outputs.
