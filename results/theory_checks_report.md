# Theory checks: machine-checked witnesses for P2-P4

Output of `run_theory_checks.py`. The propositions (the paper's framework section) are proved on paper; this report shows their checkable content executed against the actual code, so the theory and the implementation cannot silently drift apart. (P1 is true by the meaning of 'for all' and has no separate executable content.)

**Verdict: ALL CHECKS PASSED** (total time 38.4s).

## P2 certified radius — PASSED

The certified radius of skill B's preservation (tidy model, circuit ablated) came back as **0.099609** using **12 solver calls**, within the promised budget of 12. A fresh proof at the radius succeeded and a fresh attempt at 0.100195 was refuted — the returned number really is the tipping point, pinned to the promised precision (gap under 0.001).

## P3 illusion vs any grid — PASSED

For each test grid below, the proof's recipe built an edit whose surviving pocket sits between that grid's lines. Every grid approved the edit (zero violations at every point, with skill B's preservation *proved*, not just tested), and the solver refuted the removal with a survivor exactly where the construction predicted.

| grid | test points | fooled? | survivor found at x0 | predicted sliver | pocket (fraction of region) |
|---|---|---|---|---|---|
| 10x10 | 100 | yes | 0.627677 | 0.622222 ± 0.005556 | 2.78e-02 |
| 50x50 | 2500 | yes | 0.605099 | 0.604082 ± 0.001020 | 5.10e-03 |
| 100x100 | 10000 | yes | 0.601516 | 0.602020 ± 0.000505 | 2.53e-03 |
| 500x500 | 250000 | yes | 0.600501 | 0.600401 ± 0.000100 | 5.01e-04 |

## Corollary 1 adaptive tester defeated — PASSED

A deterministic ADAPTIVE tester (coarse grid, then 300 total queries zooming wherever head A looked most alive) approved a genuine edit — rightly. The proof's gadget was then placed in the widest gap between the x0 values it chose to query (center 0.975000, half-width 1.25e-02, bump height 12.9 vs a rigorous logit cap of 5.5). Re-run on the gadgeted model, the tester made the IDENTICAL sequence of queries (True) and approved again — while the parent model's correctness and skill B's preservation still PROVE, and the solver refutes the removal with a survivor at x0 = 0.975000, inside the gadget. Adaptivity did not help: a deterministic black-box tester's queries can be predicted by replaying it, exactly as Corollary 1 argues. (Randomized testers are outside this witness, as the proposition's scope remark states.)

## P4 steering algebra — PASSED

Subject: a messy trained model (tidiness off), circuit = neuron [1]. Three sub-checks:

**(a) The leftover identity.** steering(x) = ablation(x) + leftover(x), where the leftover is the formula from P4(a), held at 200,000 random inputs:

| dose | max gap between the two sides |
|---|---|
| 2.0 | 1.78e-14 |
| 8.0 | 0.00e+00 |
| 32.0 | 0.00e+00 |

**(b) The equivalence threshold.** At the corner-formula dose 5.9083, steering and ablation agreed at every sampled input with gap exactly 0.0, and the solver returned identical verdicts on all three certificates: True.

**(d) The collateral formula.** For the realistic diff-of-means vector: on every input where the push flips no neuron's on/off status, skill B's logit must shift by exactly the sum of W2[B,j]*v_j over the active neurons — a constant per activation pattern:

| dose | pattern-stable inputs | activation patterns seen | max gap vs formula |
|---|---|---|---|
| 0.5 | 156071 | 11 | 2.31e-14 |
| 1.0 | 120294 | 7 | 1.69e-14 |
| 2.0 | 69931 | 6 | 1.96e-14 |
| 4.0 | 17410 | 1 | 1.79e-14 |
| 8.0 | 0 | 0 | — |
