# The stronger removal claim: head A provably no longer listens to x0

Output of `run_independence.py`. The ordinary removal certificate says the edited head A stays at or below zero on skill A's region. This one says more: over the WHOLE input square, changing x0 alone — any two inputs identical except for x0 — moves head A's output by at most the certified influence below. Influence 0 means the head provably IGNORES x0: nothing sub-threshold is left listening. Proved with a two-copy (siamese) encoding; the influence numbers are bisected ceilings, like certified radii (tolerance 0.001).

## Tidy model (skill A's circuit: neurons [3])

| edit | certified influence of x0 on head A | on head B (yardstick: B never used x0) | note |
|---|---|---|---|
| control (no edit) | <= 22.0012 | <= 0.0062 | yardstick: the working skill MUST listen to x0 |
| ablation | <= 0.0007 | <= 0.0069 | neurons [3] switched off |
| weight edit | <= 0.0007 | <= 0.0062 | wires from neurons [3] to head A cut |
| steering targeted (dose 4) | <= 0.0006 | <= 0.0069 | smallest dose that passes the tests |
| steering diff-of-means (dose 4) | <= 0.0006 | <= 0.0069 | smallest dose that passes the tests |

(Note nothing reaches exactly 0: tiny stray weights from x0 to other neurons survive training even in the tidy model, and the certificate prices them honestly instead of rounding to zero.)

## Messy model (entangled; test-passing ablation: neurons [1])

| edit | certified influence of x0 on head A | on head B (yardstick) | note |
|---|---|---|---|
| control (no edit) | <= 62.8486 | <= 3.4153 | yardstick: the working skill MUST listen to x0 |
| ablation | <= 10.3222 | <= 3.7518 | neurons [1] switched off |
| weight edit | <= 10.3222 | <= 3.4153 | wires from neurons [1] to head A cut |
| steering targeted (dose 8) | <= 10.3226 | <= 3.7520 | smallest dose that passes the tests |
| steering diff-of-means (dose 8) | <= 8.7092 | <= 5.8700 | smallest dose that passes the tests |

(Read this table with care — it is the experiment's most instructive finding. Every edit above certifies REMOVAL on skill A's region, yet head A demonstrably still LISTENS to x0 elsewhere: most of that movement happens deep on the LOW side of zero, which removal does not forbid. On an entangled model, silencing a behavior in its region is far from deafening the head — and this certificate is the first number that states the difference as a proof. The diff-of-means row adds collateral from a new angle: after that edit, head B listens to x0 MORE than it did before the edit.)

## The acid test: the constructed illusion edit

| edit | certified influence of x0 on head A | note |
|---|---|---|
| illusion edit (constructed) | <= 0.6004 | passes 40,401 test points; the leftover tent pathway still reads x0 |

The refutation is a concrete PAIR: x = (0.814700, 0.0000) and y = (0.815900, 0.0000) — identical except for x0 — that the supposedly-removed head still tells apart (its output moves by 0.3000). An edit that passed 40,401 test points cannot pass this certificate: the leftover pathway reads x0, so the influence cannot reach zero.

## How to read the numbers

- The CONTROL rows must show large influence on head A — the unedited skill IS an x0-listener; that is the yardstick.
- After a genuine edit the certified influence collapses by orders of magnitude. Where it is exactly 0, the removal is airtight in the strongest sense: no leftover pathway from x0 to head A exists at any strength, anywhere.
- Where it is small but nonzero, the number IS the leftover: a proved ceiling on everything x0 can still do to the head — the quantity the ordinary removal claim leaves unstated, stated as a number.
- Head B's influence from x0 should be small in the control and unharmed by clean edits; steering that drags it around is collateral damage seen from a new angle.

Total time 1079.4s on a laptop CPU. The corresponding finding is §V-C of the paper.