# Capacity check: can the exact-encodable architecture learn (a+b) mod 7?

Output of `run_rung3_spike.py`. A self-contained p-way version of the threshold-gate architecture (gate attention + additive linear scores + LeakyReLU MLP — the exactly-encodable pieces), hand backprop numerically gradient-checked. NO solver, NO editing — this checks that the architecture can compute modular addition, and estimates the size against the exact frontier (~48 continuous noise variables = L x d_model).

**Gradient check:** worst |numeric - analytic| = 3.4e-10 (backprop trusted).

## Fitting the whole input space (what certification needs)

All 49 pairs of mod-7 addition; a model that gets them all right is a correct `(a+b) mod 7` computer over its entire input space, which is what a removal/preservation certificate would talk about.

| d_model | L | noise vars | best-of-3 train acc |
|---|---|---|---|
| 8 | 3 | 24 | 0.184 |
| 12 | 3 | 36 | 0.143 |
| 16 | 3 | 48 | 1.000 |

## Smaller prime for headroom (mod 5, 25 pairs)

| d_model | L | noise vars | best-of-3 train acc |
|---|---|---|---|
| 6 | 3 | 18 | 1.000 |
| 8 | 3 | 24 | 0.280 |
| 12 | 3 | 36 | 1.000 |

## Grokking-style generalization (train on 70% of pairs, d_model 12)

seed 0: train 0.12 / test 0.20; seed 1: train 1.00 / test 0.60; seed 2: train 0.15 / test 0.13.

## Verdict

**Feasible, with headroom at a smaller prime.** The exact-encodable architecture computes modular addition correctly over the whole input space: mod-7 fits at 48 noise variables (right at the exact frontier ~48), and **mod-5 fits at as few as 18 noise variables (d_model 6, L 3)** — well under the frontier, so there is room for a two-skill, editable model (the modular-adder subject uses mod 5). Learning is seed-sensitive (best-of-3, non-monotonic in size), as for the threshold-gate transformer. Grokking-style generalization is weak here, but the certified claim only needs the function correct on every pair (which it is), not delayed generalization.

Total time 93.1s on a laptop CPU (no solver).