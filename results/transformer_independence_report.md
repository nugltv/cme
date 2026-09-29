# The stronger removal claim — does the readout still listen to quotes?

Output of `run_transformer_independence.py`. The transformer analogue of the toy-model certified influence (`results/independence_report.md`): a two-copy / siamese encoding proves whether the skill-A readout's output can move when only the quote content changes (the task token, the embedding noise, and every non-quote token weight held equal between the two copies). Certified over the hull relaxation (a superset region — a proof certifies the discrete claim by P1), sharing the same validated forward pass as the removal prover.

Held-out accuracy: skill A 1.0000, skill B 1.0000. Two-copy encoding validated against the float forward: max gap 1.4e-14.

## The finding

- **Unedited readout: strongly depends on quotes.** A quote-only change can move the skill-A logit by more than **64** (the largest bound the solver refuted) — the readout genuinely reads the open/close balance, as it must to do the skill.
- **Edited readout (circuit ablated): certified influence = 0 (exactly independent).** Ablating skill A's circuit ([head [0], MLP [1, 5, 2]]) does not merely push the logit below zero — it provably makes the readout **ignore the quote content entirely**: no arrangement of « and », sub-threshold or not, moves it.
- **This independence is robust to embedding noise: certified radius >= 0.05 (cap).** The readout stays exactly quote-independent under every embedding perturbation up to that bound — a *larger* radius than the removal certificate's own (0.016), because the edit severs the quote pathway structurally rather than just clamping its sign.

## Why this matters

This closes the gap the removal certificate alone leaves open: 'the logit stays nonpositive' is weaker than 'the readout no longer listens'. On the messy toy model the strengthened claim only dropped influence 63 -> 10 (removal without deafness). Here the edit achieves **exact** independence (influence 0) and keeps it under noise — the strongest form of the removal claim, now on a real transformer with load-bearing attention.

Total time 176.4s on a laptop CPU. Bisections to 0.005; per-query timeout 180s ('unknown' is reported, never hidden).