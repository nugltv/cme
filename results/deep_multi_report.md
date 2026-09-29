# The forced multi-neuron edit: constrained circuit search on the deep model

Output of `run_deep_multi.py`. Same subject, regions, margins, tests and proof machinery as `deep_report.md`; the only change is WHICH neurons the circuit search may use. The free search collapses onto a single bottleneck neuron, so the free-search certified edit touches one neuron. Here the search is constrained so that cannot happen, and every resulting edit is certified the usual way.

Subject: training seed 1 (1 earlier seed(s) skipped, same scan rule as run_deep.py).

## The circuits the three regimes found

- **unrestricted** (the free search's pick (run_deep.py's baseline)): [(2, 6)] — 0 neuron(s) in layer 1, 1 in layer 2; found by greedy.
- **layer 1 only** (bottleneck layer off-limits; must dismantle the XOR machinery): [(1, 1), (1, 3), (1, 10), (1, 11)] — 4 neuron(s) in layer 1, 0 in layer 2; found by exhaustive small-subset search (greedy missed it — the XOR pieces cancel in pairs, so no single knockout looks helpful).
- **detour** (the unrestricted circuit's neurons banned; second-best pathway): [(1, 1), (1, 3), (1, 10), (1, 11)] — 4 neuron(s) in layer 1, 0 in layer 2; found by exhaustive small-subset search (greedy missed it — the XOR pieces cancel in pairs, so no single knockout looks helpful).

(The detour regime converged on the SAME layer-1 circuit: with the bottleneck banned, the only other certifiable handle on skill A is the XOR machinery itself. This subject offers exactly two clean edits — the funnel, or the pieces feeding it.)

## Certificates per regime and edit

| regime | edit | passes tests? | removal proved? | removal radius | preservation proved? | preservation radius | note |
|---|---|---|---|---|---|---|---|
| — | control (no edit) | — | must fail | refuted at eps=0 | (unedited) | 0.094 | yardstick: B's natural margin |
| unrestricted | ablation | yes | yes | >= 0.5 (whole cube) | yes | 0.090 | 1 neuron(s) switched off |
| unrestricted | weight edit | yes | yes | >= 0.5 (whole cube) | yes | 0.094 | wires from the layer-2 circuit neurons to head A cut |
| unrestricted | steering targeted (dose 32) | yes | yes | >= 0.5 (whole cube) | yes | 0.090 | circuit neurons pushed down; smallest dose that passes the tests |
| layer 1 only | ablation | yes | yes | >= 0.5 (whole cube) | yes | 0.074 | 4 neuron(s) switched off |
| layer 1 only | steering targeted (dose 8) | yes | yes | >= 0.5 (whole cube) | yes | 0.076 | circuit neurons pushed down; smallest dose that passes the tests |
| detour | ablation | yes | yes | >= 0.5 (whole cube) | yes | 0.074 | 4 neuron(s) switched off |
| detour | steering targeted (dose 8) | yes | yes | >= 0.5 (whole cube) | yes | 0.076 | circuit neurons pushed down; smallest dose that passes the tests |

Total time 2261.7s on a laptop CPU.

## Why this matters

If the constrained regimes certify, certified edits are not limited to one neuron: the same certificates hold for an edit spread over several neurons (and, in the detour regime, over whichever pathway remains). If a regime finds nothing test-passing, that is reported as-is — it would mean this subject funnels skill A so tightly that only the bottleneck admits a clean edit, which is itself a checkable claim about the model, not a weakness of the method.