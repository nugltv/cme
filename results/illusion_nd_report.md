# The illusion hunt in 4 and 5 input dimensions

Output of `run_illusion_nd.py`. The 2-input search found no naturally-trained illusion; this re-runs the hunt on models with 4 and 5 inputs, where test points cover far less of the space and hidden pockets have room to exist.

- Models trained: 160 (4D and 5D, four tidiness settings, 20 seeds each).
- Edits that passed the practitioner's ENTIRE test battery (grid + 1000 random inputs for removal, grids + random inputs for preservation): 198.
- Of those: 195 proved genuinely removed, 0 unresolved (solver timeout), and **3 ILLUSIONS** — edits every test approved but the solver refuted.

| dims | tidiness (l1) | seed | edit (neurons off) | survivor input | pocket size | tests it fooled |
|---|---|---|---|---|---|---|
| 4 | 0.0005 | 8 | [11, 16, 21] | [1.0, 0.510669, 0.0, 0.159972] | < 0.00015% (95%-confidence ceiling; 0 hits in 2,000,000 darts) | 625-pt grid + 1000 random |
| 4 | 0.001 | 18 | [1, 18] | [1.0, 0.000641, 1.0, 0.855361] | < 0.00015% (95%-confidence ceiling; 0 hits in 2,000,000 darts) | 625-pt grid + 1000 random |
| 5 | 0.0 | 17 | [16, 21] | [1.0, 0.7438, 0.697112, 0.070066, 0.196238] | < 0.00015% (95%-confidence ceiling; 0 hits in 2,000,000 darts) | 3125-pt grid + 1000 random |

Every survivor above was re-checked numerically on the ordinary (float) model: the edited head A really does say HIGH at the listed input, so the skill genuinely survives there.

## What this means

In higher dimensions the intervention illusion arises in ORDINARILY TRAINED models: a diligent tester (thousands of test points, grid and random, plus collateral checks) approves an edit that provably did not remove the skill. This is the naturally-occurring companion to the constructed 2-input example in `illusion_report.md` — together they show the blind spot is both unavoidable in principle and real in practice.
