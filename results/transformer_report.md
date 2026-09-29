# A certified edit on a (small) threshold-gate transformer

Output of `run_transformer.py`. Subject: one decoder block (2 threshold-gate attention heads with additive scores, LeakyReLU MLP of width 8, d_model 8), sequence length 6, trained on two sequence skills sharing the trunk — skill A: 'is there an unclosed quote?' (Q1), skill B: 'is there an unclosed bracket?' (Q2); both are global counting properties (no input coordinate IS the label). This is the small config at the frontier where exact input-side certification is tractable — see the measured size ladder in results/rung2_size_ladder.log. Every certificate below quantifies over ALL token sequences in its class AND all embedding-space noise up to the stated epsilon, in one solver query; weights enter as exact rationals (quantized to the 2^-12 grid — the quantized model is the one certified), claims carry a 1e-6 slack, and the clean token space (3,125 sequences per task) is additionally brute-forced numerically as a cross-check.

Held-out accuracy: skill A 1.0000, skill B 1.0000. Encoding validated against the float forward at 40 random (sequence, noise) points: max gap 1.4e-14.

Control certificates (unedited model, eps = 0.005): ALL PROVED — the model provably answers both questions correctly for every sequence and every perturbation. Natural preservation radius 0.034; control removal refuted at eps=0 (correctly refuted: the skill is present).

Skill A's circuit (greedy, removal objective): [('head', 0), ('mlp', 1), ('mlp', 5), ('mlp', 2)].

| edit | passes tests? | removal proved? | removal radius | preservation proved? | preservation radius | note |
|---|---|---|---|---|---|---|
| ablation (circuit) | yes | yes | 0.016 | yes | 0.013 | atoms [('head', 0), ('mlp', 1), ('mlp', 5), ('mlp', 2)] switched off |

Total time 2241.5s on a laptop CPU. Radii are bisected to 0.005; the radius cap is 0.05 (embedding entries are O(0.5), so 0.05 is already a sizeable perturbation of every token at once). Natural-radius bisection depends on the solver finishing each near-boundary probe within the timeout; a probe that times out is treated as a failure (sound: the reported radius is always a proved lower bound), so natural radii can vary slightly run-to-run while the edit radii here are stable.

## Why this matters

This certifies an edit on a transformer: the certified object has content-based attention and an MLP, and the attention is load-bearing (switching off the heads alone, or the MLP alone, leaves skill A firing: results/rung2_circuit_search.log). The removed skill is a sequence property no single input coordinate encodes; the quantifier is continuous and input-side (embedding noise ahead of the whole computation, where Somani's certificates perturb the final residual on traced inputs); and the toy models' edit comparison is re-asked on this subject. Exact input-side certification requires threshold-gate attention (soft attention's application is bilinear in the noise); the gate model trains to 100% on both skills.