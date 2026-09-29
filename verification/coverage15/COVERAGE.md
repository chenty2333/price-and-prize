# Exact finite coverage: what is and is not certified

The current theorem concerns **market-balanced SD-EF1 existence with at most one
forward surplus unit**, not own-class-balanced existence beyond twelve goods.
The latter is false at thirteen goods. `strong_counterexample13.json` and the
technical appendix give an explicit profile, an analytic impossibility proof, and a
fair unrestricted allocation.

A full fifteen-good verification is certified only by the aggregate
`coverage15/verification.json` with `status: PASS`, all shard results present,
and represented orbit sizes summing to 1,401,400. A passing single partition,
a random test, a stopped pilot run, or a passing proper subset of shards does
not establish the finite theorem. No full eighteen-good run is reported.

## 1. Normalization

Relabel goods by their positions in agent 1's ranking. Agent 1 then has the
identity order. Market balance depends only on the unordered partition into
triples; the dummy is protected regardless of block order or within-block order.
There are `m! / (6^(m/3) (m/3)!)` such partitions: 15,400 at twelve goods,
1,401,400 at fifteen, and 190,590,400 at eighteen.

The enumerator takes the least unassigned label and each unordered pair of
larger remaining labels to form its block, and recurses. This generates every
partition exactly once. Shard ranges refer to this enumerator's indices.

## 2. Four sufficient candidate families

Every family consists of complete market allocations that pass **every original
agent-1 prefix inequality**. Every full bundle therefore has q=m/3 goods.

* `own`: one agent-1 good in every ranking triple.
* `shift1`: at most one surplus unit, with nonnegative block-end heights.
* `own_first`: `own`, and all three agents occur once in the first ranking triple.
* `shift1_first`: `shift1`, with that same first-triple condition.

For counts c_j in agent 1's ranking triples, the surplus is
`sum(max(c_j-1,0))`. Height at block endpoint k is `sum_{j<k}(c_j-1)`.
Every fair allocation has nonnegative height: at a 3k-good prefix, if agent 1
has s goods, the total is at most `3s+2`, forcing s>=k. Thus this height test
adds no unjustified fairness restriction.

Candidates are not sampled or truncated. The program enumerates all 6^q market
allocations and keeps exactly the members of the selected family.

## 3. Exact covering subfamilies

A ranking-prefix state has prefix set T and the candidate family L that has
survived **all preceding** agent-2 prefix tests. The next valid-candidate mask
depends only on T plus the next good, while L contains the earlier-order history.

A subfamily K of L is a covering core at T when every ordering of the remaining
goods is acceptable to agent 2 in at least one allocation in K. Different
continuations may use different allocations.

There are three certified operations:

1. If A in L already gives agent 2 at least q-1 goods, {A} covers every
   continuation. All opposing bundles have size q.
2. For each next good g, recursively obtain a core K_g from the surviving
   child candidates. The union of these cores covers at the parent.
3. A previously proved core K at T can be reused whenever K is a subset of
   the current L. Adding candidates never harms universal coverage.

Core learning is an induction on the number of unranked goods. It makes no
assumption that the whole allocation is currently fair before its final
selection; only the prefix tests explicitly encountered in this **ranking
coverage** recursion are required. It does not discard partial allocations in
an allocation-building induction.

The C++ program stores full integer candidate bitsets and tests exact subset
inclusion. It never equates states by hash alone. It may discard a stored
superset of a new covering core. That changes speed and counters, not truth.
An empty surviving family yields a concrete uncovered ranking prefix.

## 4. Endpoint symmetries and the fallback hierarchy

H permutes the first two labels and the last three labels independently;
|H|=12. Both `own` and `shift1` are H-equivariant. Permuting the first two
labels changes only the always-safe single-good prefix. At the beginning of
the last triple, agent 1 already has at least q-1 goods by the height lemma,
so all orders within the last triple are safe.

G permutes the entire first triple and the last triple; |G|=36.
`own_first` and `shift1_first` are G-equivariant because all prefixes inside
the first triple are then safe. **The first-triple condition is essential.**
The negative control `(1,0,0,1,2,2)` becomes agent-1-unfair after exchanging
labels 0 and 2, although it is initially fair and own-balanced.

For each G representative, the verifier tries `own_first`, then `shift1_first`.
If both fail, it explicitly splits that G orbit into at most three H orbits
and tries `own`, then `shift1`, for each. Successful coverage at any sufficient
family proves the desired one-shift existence for every partition in its orbit.
A failure of `own_first` alone does not show that ordinary own balance fails.

## 5. Canonical representatives and exact orbit weights

For a block B, record its fixed labels and the numbers a_B,b_B of mutable head
and tail labels. The multiset of these signatures is a complete orbit invariant:
match blocks with equal signatures and then their mutable labels to construct
a group element. Assigning mutable labels in sorted signature order gives a
canonical representative lying in the orbit.

The stabilizer has size

    product_B(a_B! b_B!) * product_(empty-fixed identical types)(multiplicity!)

Blocks with fixed labels cannot exchange places; identical empty-fixed blocks
can. Dividing 12 or 36 by the stabilizer gives the exact orbit size.

For fallback, all 36 group elements are enumerated explicitly, H-canonicalized,
and deduplicated by their complete block vectors. The sum of H orbit sizes is
checked against the parent G orbit size. Independent Python tests compare this
signature/weight argument against explicit group actions on 5,000 partitions.

One shard contains canonical representatives whose enumerator indices lie in
its half-open range. Their orbits can contain partitions with indices outside
that range. Thus `partitions_represented` is an orbit-weight sum, not the
number of raw indices in the shard. Across all disjoint shards, it must sum to
the entire domain. At fifteen goods there are 82,670 G orbits. An independent double count gives,
with N_m the number of triple partitions and m>=6,

    |G-orbits| = (N_m + (6(m-2)+4)N_(m-3)
                  + (9(m-4)(m-5)+12(m-5)+4)N_(m-6)) / 36.

The six group-element cycle types are the identity, one transposition, one
3-cycle, two transpositions, a transposition and a 3-cycle, and two 3-cycles;
their multiplicities are 1,6,4,9,12,4. Invariant blocks give the corresponding
fixed-partition counts N_m, (m-2)N_(m-3), N_(m-3),
(m-4)(m-5)N_(m-6), (m-5)N_(m-6), N_(m-6).
Averaging fixed-point counts counts orbits, by double-counting pairs (group
element, fixed partition). This also gives 1,090 at m=12 and 9,956,100 at
m=18. The eighteen-good number is a mathematical count, not an executed
coverage result.

## 6. Counter meanings and independent checks

`recursive_calls` counts entries to the recursive routine. `learned_cores`
counts newly certified core insertions. `retained_cores` sums the final stored
antichain sizes over attempted families. `subsumption_hits` counts successful
reuse. `safe_prunes` includes permanent-acceptability child checks that avoid
recursion. `coverage_problems` counts each attempted candidate family, so it
can exceed the number of G representatives.

These are not the previous verifier's full-state memo counts. The independent
`verify_python.py` deliberately retains the original full `(T,L)` state,
without learned cores or group quotienting. Its default ten complete
partition-coverage problems include the own-balanced failure and its
one-shift repair. The optional `--cpp` comparison expects the separately
compiled `verify_reference.cpp` program. Its larger candidate families can
require appreciable memory; no claim is made that every possible partition is
practical in this reference Python implementation.

`python_mask_self_test.json` covers eighteen complete small problems with every
prefix-mask calculation crosschecked by direct counts. Each fifteen-good
attempt additionally checks 128 masks directly. `symmetry_checks.json` records
5,000 explicit orbit tests and 483,840 candidate-membership invariance checks.
The full-state reference and fresh brute-force checker do not rely on learned
core logic. No proof-assistant certification is asserted.

## 7. Reproduction

    python run_verification.py --m 15 --workers 4 --output-dir coverage15
    c++ -O3 -std=c++17 verify_reference.cpp -o verify_reference
    python verify_python.py --cpp ./verify_reference
    python verify_python.py --self-test --output python_mask_self_test.json
    python test_symmetry.py

The primary runner compiles `verify_cover.cpp` with GCC-compatible C++17,
`-O3 -march=native`, records its SHA-256, and rejects incomplete coverage.
Only exact arithmetic is used. Runtime measurements are elapsed times of
actual executions, not future estimates. The sum of shard elapsed times is
not aggregate CPU time because the processes can contend with one another
and with the separately documented checks.
