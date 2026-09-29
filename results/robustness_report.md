# Certified wiggle room: robustness of the edit

Output of `run_robustness.py`. For each kind of edit removing the same skill from the same model, we prove how much input NUDGING the edit's effect withstands. 'Removal radius' = the largest wiggle room for which 'skill A stays gone' still proves; 'preservation radius' = the same for 'skill B still works'. The control row (no edit) is the yardstick: its preservation radius is B's natural safety margin, and any edit that shrinks it is doing collateral damage.

Wiggle room capped at 0.6 (the whole input square). Radii pinned to ±0.001; every probe is a full solver proof. Total proving time: 73.4s.

## Subject 1 — the tidy model (skill A = neuron(s) [3])

| edit | passes tests? | removal radius | preservation radius | note |
|---|---|---|---|---|
| control (no edit) | ❌ | **refuted at ε=0** (survivor at (0.6, 0.0)) | 0.100 | yardstick: removal must fail; preservation radius = B's natural margin |
| ablation | ✅ | **≥ 0.6** (whole input square — nothing left to nudge to) | 0.100 | neurons [3] switched off |
| weight edit | ✅ | **≥ 0.6** (whole input square — nothing left to nudge to) | 0.100 | wires from neurons [3] to head A cut |
| steering targeted (dose 4) | ✅ | **≥ 0.6** (whole input square — nothing left to nudge to) | 0.100 | smallest dose that passes the tests |
| steering targeted (dose 16) | ✅ | **≥ 0.6** (whole input square — nothing left to nudge to) | 0.100 | 4x the minimal dose |
| steering diff-of-means (dose 4) | ✅ | **≥ 0.6** (whole input square — nothing left to nudge to) | 0.091 | smallest dose that passes the tests; no neuron chosen by hand |
| steering diff-of-means (dose 16) | ✅ | **≥ 0.6** (whole input square — nothing left to nudge to) | 0.066 | 4x the minimal dose |

## Subject 2 — the messy model (tidiness off; practitioner's circuit = neurons [1])

Skill A is smeared across several neurons here, so the edit types can genuinely differ — this is the honest comparison.

| edit | passes tests? | removal radius | preservation radius | note |
|---|---|---|---|---|
| control (no edit) | ❌ | **refuted at ε=0** (survivor at (0.9656, 0.026)) | 0.096 | yardstick: removal must fail; preservation radius = B's natural margin |
| ablation | ✅ | **≥ 0.6** (whole input square — nothing left to nudge to) | 0.096 | neurons [1] switched off |
| weight edit | ✅ | **≥ 0.6** (whole input square — nothing left to nudge to) | 0.096 | wires from neurons [1] to head A cut |
| steering targeted (dose 8) | ✅ | **≥ 0.6** (whole input square — nothing left to nudge to) | 0.096 | smallest dose that passes the tests |
| steering targeted (dose 32) | ✅ | **≥ 0.6** (whole input square — nothing left to nudge to) | 0.096 | 4x the minimal dose |
| steering diff-of-means (dose 8) | ✅ | **≥ 0.6** (whole input square — nothing left to nudge to) | 0.069 | smallest dose that passes the tests; no neuron chosen by hand |
| steering diff-of-means (dose 32) | ❌ | **≥ 0.6** (whole input square — nothing left to nudge to) | - | 4x the minimal dose |

## How to read this, in one paragraph

Every number in the table is a PROOF, not a measurement: a removal radius of 0.25 means the solver verified that no input within 0.25 of the approved region — nudged in any direction, any combination of coordinates — makes the removed skill fire again; and it found a concrete input at 0.251 that does. Testing cannot produce this table even in principle: it cannot check infinitely many nudges of infinitely many inputs. This is the certified analogue of asking 'can a jailbreak-style nudge bring the skill back, and how big must it be?' — for input nudges, with the weights frozen as edited. Attacks that change the weights (fine-tuning recovery) are outside what any of these certificates promise; see the paper's limitations section for the full list of what the radius does and does not cover.
