# Beyond the toy: certified edits on a two-hidden-layer XOR model

Output of `run_deep.py`. Model: 3 inputs -> 12 ReLU -> 8 ReLU -> 2 heads. Skill A = 'exactly one of x0, x1 above 0.5' (an XOR — no single neuron CAN hold it, so the circuit is distributed by necessity); skill B = 'x2 above 0.5'. Skill A's region is the union of two boxes, proved separately and combined (proposition P1). All the usual rules apply: exact fractions in the solver, margin 0.1, every number below is a proof.

Trained with seed 1; control certificates (all PROVED): the unedited model provably computes the XOR on all four of its boxes and skill B on both of its.

Seeds skipped before this subject: 1 (0 under-trained, 1 with no ablation of up to 10 neurons that even LOOKS like removal on tests — e.g. models whose head-A resting bias is positive cannot be silenced into saying LOW; an honest scan reports these rather than hiding them).

Skill A's circuit (knock-out search): [(2, 6)] — 0 neuron(s) in layer 1, 1 in layer 2.

| edit | passes tests? | removal proved? | removal radius | preservation proved? | preservation radius | note |
|---|---|---|---|---|---|---|
| control (no edit) | — | must fail | refuted at eps=0 | (unedited) | 0.094 | yardstick: B's natural margin |
| ablation | yes | yes | >= 0.5 (whole cube) | yes | 0.090 | neurons [(2, 6)] switched off |
| weight edit | yes | yes | >= 0.5 (whole cube) | yes | 0.094 | wires from the layer-2 circuit neurons to head A cut |
| steering targeted (dose 32) | yes | yes | >= 0.5 (whole cube) | yes | 0.090 | circuit neurons in both layers pushed down; smallest dose that passes the tests |
| steering diff-of-means @L2 (dose 32) | yes | yes | >= 0.5 (whole cube) | yes | 0.076 | smallest dose that passes the tests |
| steering diff-of-means @L2 (dose 128) | NO | yes | >= 0.5 (whole cube) | REFUTED | refuted at eps=0 | 4x the minimal dose; FAILS the ordinary tests |

- diff-of-means steering at layer 1: NO dose up to 64 passes the ordinary tests — every dose either leaves skill A visibly alive or visibly breaks skill B. Where you steer matters.

Total time 2759.5s on a laptop CPU (training + testing + all proofs). Depth is not free for the solver: single proofs still take seconds, but the radii bisections dominate and the full run costs tens of minutes, versus ~80s for the shallow equivalent.

## Why this matters

Depth and a genuinely distributed skill change nothing about what the certificates MEAN — but they void the one-hidden-layer algebra (proposition P4) that EXPLAINED the shallow results, so whatever pattern appears in this table is independent empirical evidence, not a corollary.