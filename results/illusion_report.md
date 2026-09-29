# The intervention illusion

Output of `run_illusion.py`. The question: can an edit LOOK successful under ordinary testing while the solver PROVES the skill survives? Answer below, two ways.

## Route B — the constructed example (guaranteed)

- We built a 7-neuron model by hand. Before the edit, skill A provably works (proved: True).
- The edit switches off the two main skill-A neurons — the correct, obvious circuit. A hidden backup pathway survives in a sliver of inputs ~0.0006 wide around x0 = 0.815.
- **Coarse test (225 points): skill A looks GONE** (passed: True).
- **Fine grid (40401 points): 0 survivors seen** — even 40,000+ test points miss the sliver.
- Random test (300 points): 0 survivors seen. (Honest footnote: at this pocket size a 300-point random test misses the sliver with probability only 55% — this pass is seed luck. The grids are dodged by construction, and the recipe behind this model — proposition P3 — shrinks the pocket below any target test size; the in-principle claim rests on those, not on this coin flip.)
- **Solver: removal NOT proved** — counterexample at x0=0.815300, x1=0.000000 (numeric check: edited head A logit = 0.0003 > 0, skill alive).
- The surviving pocket covers only **0.200%** of the region — small enough to hide from tests, impossible to hide from the solver.
- Skill B, meanwhile, is PROVED preserved (True) — so every test a practitioner would run says this edit is a clean success.

## Route A — searched in ordinarily-trained models

- Trained 200 models across a sweep of tidiness penalties and hidden-layer sizes; 233 candidate edits passed the practitioner's full test battery.
- Of those, 233 were PROVED genuinely removed (the test's verdict was right), and **0 were illusions** (the test approved an edit the solver refuted).

_No naturally-arising illusion in this batch of seeds — the constructed example (Route B) still demonstrates the blind spot; widen the search (more seeds, other edit choices) to find a trained one._

## What this means, in one sentence

An edit can pass every test a careful practitioner would run — coarse grid, fine grid, random samples, collateral checks — while the skill it was supposed to remove provably survives in a pocket of inputs the tests never touch; only a proof over the WHOLE region (or any white-box method — the point is opening the model, not sampling it) can tell the difference.
