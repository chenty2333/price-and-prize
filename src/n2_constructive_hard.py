"""Stress-test n2_constructive.solve on the instances that matter most:
every hard profile for m = 7, 8 (exhaustive; no partition is SD-EF1 for all
three rankings) and the Lemma-B-adversarial profiles, each with many market
valuations, including 0/1 prefix valuations (the extreme rays of the cone of
nonincreasing v)."""
import itertools
import json

import numpy as np

import fd
from n2_constructive import random_v, solve

rng = np.random.default_rng(7)
profiles = []
for m in (7, 8):
    perms = list(itertools.permutations(range(m)))
    Mal = fd.market_sd_ef1_allocations(2, m)
    b0 = np.array([np.packbits(fd.sd_ef1_pair(Mal, 0, 1, list(p))) for p in perms])
    b1 = np.array([np.packbits(fd.sd_ef1_pair(Mal, 1, 0, list(p))) for p in perms])
    for i in range(len(perms)):
        for k in np.nonzero(~((b0[i][None, :] & b1).any(axis=1)))[0]:
            profiles.append((m, list(perms[i]), list(perms[k])))
for rec in json.load(open("../results/n2_lemmaB_adversarial.json")):
    for s in rec["zeros"]:
        profiles.append((rec["m"], s[0], s[1]))
count = 0
for m, s1, s2 in profiles:
    vs = [np.r_[np.ones(k), np.zeros(m - k)] for k in range(1, m + 1)] + [random_v(m, rng) for _ in range(3)]
    for v in vs:
        owner, _ = solve(m, s1, s2, v)
        x = np.array([[owner[g] for g in range(m)]], dtype=np.int8)
        assert fd.subjective_sd_ef1(x, [s1, s2])[0] and fd.market_ef1(x, v, 2)[0]
        count += 1
print(f"hard/adversarial profiles: {len(profiles)}; (profile, v) pairs verified: {count}")
