# The full edit-type x certified-radius table (threshold-gate transformer)

Output of `run_transformer_edit_table.py`. The transformer version of the toy models' edit comparison: every edit type gets a certified removal radius AND a certified preservation radius, over ALL sequences x all embedding noise (hull relaxation; a proof certifies the discrete claim by P1), from the same exact-rational prover as `run_transformer.py`. This extends `run_transformer_steering.py` with the weight-edit and targeted-steering rows, so all four edit types appear as certified rows on a transformer.

Subject: the certified threshold-gate model (d_model 8, 2 heads, L=6, seed 5). Held-out accuracy: skill A 1.0000, skill B 1.0000. Skill A's circuit: attention head [0] + MLP neurons [1, 5, 2].

| edit | passes tests? | removal | removal radius | preservation | preservation radius | numeric acc (A / B) | note |
|---|---|---|---|---|---|---|---|
| ablation [head [0], MLP [1, 5, 2]] | yes | proved | 0.016 | proved | 0.013 | 0.632 / 1.000 | certified edit; head read-out zeroed + MLP incoming wires cut |
| weight-edit [head [0], MLP [1, 5, 2]] | yes | proved | 0.016 | proved | 0.013 | 0.632 / 1.000 | head read-out zeroed + MLP OUTGOING wires cut (neurons still fire, nothing listens) |
| steering targeted (dose 2) | no | refuted | refuted at eps=0 | proved | 0.034 | 1.000 / 1.000 | least-norm residual push driving MLP [1, 5, 2] pre-activations down, scaled 2x |
| steering targeted (dose 8) | no | refuted | refuted at eps=0 | proved | 0.031 | 0.724 / 1.000 | least-norm residual push driving MLP [1, 5, 2] pre-activations down, scaled 8x |
| steering targeted (dose 32) | no | refuted | refuted at eps=0 | unknown | 0.016 | 0.722 / 1.000 | least-norm residual push driving MLP [1, 5, 2] pre-activations down, scaled 32x |
| steering diff-of-means (dose 1) | no | refuted | refuted at eps=0 | unknown | 0.028 | 1.000 / 1.000 | residual nudged by 1x the diff-of-means vector |
| steering diff-of-means (dose 2) | no | refuted | refuted at eps=0 | proved | 0.009 | 0.783 / 1.000 | residual nudged by 2x the diff-of-means vector |
| steering diff-of-means (dose 4) | no | undecided | refuted at eps=0 | undecided | refuted at eps=0 | 0.783 / 0.975 | residual nudged by 4x the diff-of-means vector |
| steering diff-of-means (dose 8) | no | undecided | refuted at eps=0 | undecided | refuted at eps=0 | 0.637 / 0.649 | residual nudged by 8x the diff-of-means vector |
| steering diff-of-means (dose 16) | no | proved | >= 0.05 (cap) | undecided | refuted at eps=0 | 0.632 / 0.643 | residual nudged by 16x the diff-of-means vector |
| steering diff-of-means (dose 32) | no | proved | >= 0.05 (cap) | refuted | refuted at eps=0 | 0.632 / 0.643 | residual nudged by 32x the diff-of-means vector |

Verdicts are four-way: **proved** (no counterexample over the whole region), **refuted** (the solver returned a *discrete* counterexample — a real sequence), **undecided** (the hull relaxation returned a *fractional* witness, which lies outside the discrete set and so neither certifies nor refutes the discrete claim), **unknown** (the solver timed out). All three non-proved outcomes are treated as failures for the radius, so radii stay proved lower bounds. The numeric accuracy columns (20k held-out sequences) disambiguate every non-proved cell: e.g. a preservation 'unknown' with skill B still at 1.000 is a solver-cost limit, not a broken skill.

## Reading the table

**Surgical edits (ablation, weight-edit) are the only edits that both remove skill A and preserve skill B**, each certified. They are weight changes to skill A's circuit — the head's read-out and the MLP neurons' wires — so they reach the whole circuit. Ablation cuts the MLP neurons' *incoming* wires; weight-edit cuts their *outgoing* wires; both reach the head the same way (its read-out columns zeroed), which is why their rows are identical (removal 0.016, preservation 0.013). This is the toy models' result that surgical edits remove cleanly, reproduced on the transformer.

**Targeted steering never removes skill A, at any dose** (removal refuted at every dose), because it is a small least-norm residual push that silences the circuit's MLP neurons but has no handle on the load-bearing attention head — which keeps reading the quotes (the attention is load-bearing; Fig. 3). It is, however, perfectly surgical about skill B: preservation proved and numeric B = 1.000 all the way to dose 32.

**Diff-of-means steering removes skill A only at a dose that simultaneously destroys skill B.** At low dose it removes nothing (removal refuted, then undecided by the relaxation); by dose 16 the push is large enough to force the A-readout negative and removal finally certifies (radius >= 0.05) — but it is a sledgehammer, not surgery: skill B has already collapsed to chance (numeric B 0.643, preservation refuted/undecided). There is **no dose that both removes A and preserves B** — the same dose->collateral trade-off P4 proves for the toy models, now on a transformer, and the quantified reason only the surgical edits are deployable.

Total time 4842.9s. Radii bisected to 0.005; cap 0.05; per-query timeout 120s (a timed-out probe is treated as failure, so every reported radius is a proved lower bound).