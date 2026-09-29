# The margin-shrink curve: how sharp can each claim be?

Output of `run_margin_curve.py`. Every certificate is stated with a MARGIN — a strip around the 0.5 decision boundary the claim stays out of, because the skill itself is ambiguous there. This report answers 'why margin 0.1, and what happens as it shrinks?': for each claim, the **tipping margin** below is the sharpest version of that claim that still PROVES (found by bisection of proofs; margins pinned to ±0.0005). '0' means the claim proves arbitrarily close to the boundary; larger numbers mean the proof breaks further out.

Read contribution-first: the tipping margin is also the **closest certifiable threshold**. The specification uses x ≥ 0.6 (margin 0.1), but every tipping margin below is how far we can push that edge toward the true 0.5 rule — the closest certifiable threshold is 0.5 ± (tipping margin). A tipping margin of 0.0012 means we can certify down to x ≥ 0.5012; a margin of 0 means the certificate reaches the true rule itself. So the certified boundary is not an arbitrary 0.6 offset — it tracks the model's OWN decision boundary, stopping only where the skill fades to a coin-flip and the margin genuinely vanishes. That the gap is the model's, not the prover's, is the honest content: the certified region is exactly the region where the skill is decisively present.

Note: for these one-sided regions, shrinking the margin is the same operation as the certified-radius inflation, so preservation tipping margins line up with (0.1 − certified radius) from the robustness report — one number, two readings. New here: the unedited model's own sharpness, removal tipping margins, and the dose sweep.

Total proving time: 91.1s (225 solver proofs).

## Subject 1 — the tidy model (skill A circuit: neurons [3])

**The unedited model's own sharpness** (correctness claims):

| claim | provable down to margin |
|---|---|
| A high | 0 (holds all the way to the boundary) |
| B high | 0 (holds all the way to the boundary) |
| B low | 0.0003 |

**After each edit** (removal of A + preservation of B):

| edit | removal provable down to | B-high provable down to | B-low provable down to |
|---|---|---|---|
| ablation | 0 (holds all the way to the boundary) | 0 (holds all the way to the boundary) | 0.0003 |
| weight edit | 0 (holds all the way to the boundary) | 0 (holds all the way to the boundary) | 0.0003 |
| steering targeted (dose 4) | 0 (holds all the way to the boundary) | 0 (holds all the way to the boundary) | 0.0003 |
| steering diff-of-means (dose 4) | 0 (holds all the way to the boundary) | 0 (holds all the way to the boundary) | 0.0088 |
| steering diff-of-means (dose 16) | 0 (holds all the way to the boundary) | 0 (holds all the way to the boundary) | 0.0343 |

## Subject 2 — the messy model (tidiness off) (skill A circuit: neurons [1])

**The unedited model's own sharpness** (correctness claims):

| claim | provable down to margin |
|---|---|
| A high | 0.0012 |
| B high | 0.0038 |
| B low | 0 (holds all the way to the boundary) |

**After each edit** (removal of A + preservation of B):

| edit | removal provable down to | B-high provable down to | B-low provable down to |
|---|---|---|---|
| ablation | 0 (holds all the way to the boundary) | 0.0038 | 0.0038 |
| weight edit | 0 (holds all the way to the boundary) | 0.0038 | 0 (holds all the way to the boundary) |
| steering targeted (dose 8) | 0 (holds all the way to the boundary) | 0.0038 | 0.0038 |
| steering diff-of-means (dose 8) | 0 (holds all the way to the boundary) | 0.0311 | 0.0076 |
| steering diff-of-means (dose 32) | 0 (holds all the way to the boundary) | 0.1061 | 0 (holds all the way to the boundary) |

## Dose vs margin (targeted steering, tidy model)

Proposition P4(c) says the minimal certifying dose can only grow as the region grows — so a weaker dose could, in general, certify removal only at a larger margin. The proved curve:

| dose | removal provable down to margin |
|---|---|
| 1 | — (not provable even at margin 0.3) |
| 2 | — (not provable even at margin 0.3) |
| 2.25 | — (not provable even at margin 0.3) |
| 2.5 | — (not provable even at margin 0.3) |
| 2.75 | — (not provable even at margin 0.3) |
| 3 | 0 (holds all the way to the boundary) |
| 4 | 0 (holds all the way to the boundary) |
| 8 | 0 (holds all the way to the boundary) |

What the step shape means: on this task the input where the skill fires HARDEST (x0 = 1, the far corner) lies inside the region at every margin, so a dose either beats that worst case — and then certifies clear down to the boundary — or fails at every margin. The margin-dependence P4(c) allows would only show on a task whose strongest activations sit near the boundary; here the theory's inequality is simply slack. The threshold itself (between dose 2.75 and 3) is the empirical face of P4(b)'s corner formula.

## How to read this, in one paragraph

The margin is part of the skill's *definition*, not a fudge factor: behavior inside the ambiguous strip is out of spec. The table shows nothing is hidden by that choice — margin 0.1 was simply a round number above the models' natural sharpness (worst case observed: 0.0038). The correctness and preservation claims stay provable essentially down to the boundary for the unedited model and for the surgical edits; the realistic steering recipe's collateral damage appears as a premature tipping margin (at high dose on the messy model its preservation claim is unprovable even at the spec margin 0.1 — the same break the robustness report catches); and removal, once real, proves all the way to the boundary. The proof machinery localizes exactly where each claim runs out.