# The edit-type comparison on the known-formula subject

Output of `run_rung3_edit_table.py`. The edit comparison of the toy models, carried to the known-formula model (two independent modular adders on disjoint positions). 'Removed' here is the STRONG claim: the edited output is exactly independent of the summands (a1,b1) — the model no longer reads the numbers it should add. Numeric columns are exhaustive over all 625 sequences; the removal claim is anchored by the exact p-way two-copy Z3 certificate of `verify_rung3.py` over ALL sequences x continuous embedding noise.

Subject: mod 5, d_model 8, 2 heads, seed 0. Per-skill accuracy: A 1.0000, B 1.0000. Skill A's circuit: [('head', 0)] (a single attention head). Chance = 0.200.

| edit | reaches the head? | summand-independence (numeric) | skill A acc | skill B acc | note |
|---|---|---|---|---|---|
| ablation = weight-edit [head [0]] | yes | 1.000 | 0.200 | 1.000 | head read-out zeroed; for a head there is no incoming/outgoing distinction, so ablation and weight-edit coincide |
| steering diff-of-means (dose 1) | no | 0.000 | 1.000 | 1.000 | residual pushed 1x along the summand diff-of-means axis (cannot reach the head) |
| steering diff-of-means (dose 2) | no | 0.000 | 0.920 | 0.880 | residual pushed 2x along the summand diff-of-means axis (cannot reach the head) |
| steering diff-of-means (dose 4) | no | 0.000 | 0.280 | 0.520 | residual pushed 4x along the summand diff-of-means axis (cannot reach the head) |
| steering diff-of-means (dose 8) | no | 0.000 | 0.160 | 0.200 | residual pushed 8x along the summand diff-of-means axis (cannot reach the head) |
| steering diff-of-means (dose 16) | no | 0.000 | 0.040 | 0.240 | residual pushed 16x along the summand diff-of-means axis (cannot reach the head) |
| steering diff-of-means (dose 32) | no | 0.000 | 0.240 | 0.160 | residual pushed 32x along the summand diff-of-means axis (cannot reach the head) |

## The exact certificates (the two anchors)

- **Surgical edit** (ablation = weight-edit): summand-independence **proved**, certified radius **>= 0.05 (cap)** — proved exactly independent of the summands over all sequences and a continuous embedding-noise ball.
- **Unedited control**: **refuted (a summand-only change moves an output logit by more than the bound)** — correctly refuted; the intact model does read the summands (it computes the sum), so the certificate returns a witness, exactly as it should.
- **Steering** (diff-of-means, dose 8): **refuted (a summand-only change moves an output logit by more than the bound)** — refuted, just like the control: no residual offset makes the p-way output independent of the summands.

## Reading the table

**For a head-only circuit, ablation and weight-edit coincide.** An MLP neuron has separable incoming and outgoing wires (the gate transformer uses that to give ablation and weight-edit distinct rows); an attention head does not — the only weight change that stops it reaching the residual is zeroing its read-out columns. So the surgical family collapses to one certified row here, which is itself an honest finding at this scale: the whole skill lives in one head.

**Steering cannot achieve the strong removal, at any dose.** This is the toy models' surgical-vs-steering gap in its starkest form. The certified removal here is exact summand-independence; a residual steering vector adds a constant before the MLP, which shifts the operating point but leaves the attention head reading (a1,b1). The dose sweep shows numeric summand-independence never reaching 1.0 while skill B's accuracy erodes as the dose climbs — the same dose->collateral trade-off, now against a removal target steering provably cannot meet.

Total time 238.6s. Certified radius bisected to 0.005; cap 0.05; per-query timeout 180s (a timed-out probe is treated as failure, so the reported radius is a proved lower bound).