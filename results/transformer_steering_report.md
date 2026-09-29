# Steering as certified rows (the edit comparison, on the transformer)

Output of `run_transformer_steering.py`. Each row is a certified removal radius and a certified preservation radius for one edit, over ALL sequences x all embedding noise (hull relaxation; a proof certifies the discrete claim by P1), from the same exact-rational prover as `run_transformer.py`. This is the transformer version of the toy models' edit comparison, with steering reported as certified numbers.

Held-out accuracy: skill A 1.0000, skill B 1.0000.

| edit | passes tests? | removal proved? | removal radius | preservation proved? | preservation radius | note |
|---|---|---|---|---|---|---|
| ablation [head [0], MLP [1, 5, 2]] | yes | yes | 0.016 | yes | 0.013 | the certified edit |
| steering diff-of-means (dose 1) | no | no | refuted at eps=0 | yes | 0.028 | residual nudged by 1x the diff-of-means vector |
| steering diff-of-means (dose 2) | no | no | refuted at eps=0 | yes | 0.022 | residual nudged by 2x the diff-of-means vector |
| steering diff-of-means (dose 4) | no | no | refuted at eps=0 | no | refuted at eps=0 | residual nudged by 4x the diff-of-means vector |
| steering diff-of-means (dose 8) | no | no | refuted at eps=0 | no | refuted at eps=0 | residual nudged by 8x the diff-of-means vector |
| steering diff-of-means (dose 16) | no | yes | >= 0.05 (cap) | no | refuted at eps=0 | residual nudged by 16x the diff-of-means vector |
| steering diff-of-means (dose 32) | no | yes | >= 0.05 (cap) | no | refuted at eps=0 | residual nudged by 32x the diff-of-means vector |

## Reading the table

Ablation removes skill A and preserves skill B, both certified. Steering traces the **dose -> collateral** trade-off P4 proves for the toy models: at low dose skill B is intact but skill A is not removed (removal radius refuted at eps 0); as the dose climbs, removal eventually certifies — but skill B's preservation radius collapses in step. No single dose both removes A and preserves B, which is the quantified reason steering is not deployable here.

Total time 3852.0s on a laptop CPU. Radii bisected to 0.005; cap 0.05; per-query timeout 120s ('unknown' reported, never hidden — a timed-out probe is treated as a failure, so every reported radius is a proved lower bound).