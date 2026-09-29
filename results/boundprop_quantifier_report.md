# Tightening the quantifier — discrete-token closure + the reach of the continuous ball

Output of `run_boundprop_quantifier.py`. It addresses the gap between a continuous embedding ball and the adversary's discrete token-substitution move set, on the M1 softmax+LayerNorm subject, in two halves.

Model: the M1 subject (d_model 64, 4 heads, 2 layers). Accuracy A 1.000→0.500, B 1.000→1.000 after the position-scoped attention knockout.

## Half 1 — a genuine discrete-token guarantee for removal

The edit firewalls the quote positions, so the quote content is irrelevant to the readout. We therefore certify removal not just over the TQ-positive sequences (the original M1 claim) but over the **entire TQ class** — every sequence whose task token is TQ, across all quote AND all bracket content. That set is *closed under any in-vocabulary rewriting of the content positions*, so certifying it is exactly a discrete token-substitution neighborhood guarantee: an attacker may rewrite the content to anything in-vocabulary and skill A stays removed — each sequence still carrying a continuous embedding ball on top.

| removal claim | # sequences | certified radius (sound) | PGD break |
|---|---|---|---|
| removal over ALL TQ sequences (discrete closure) | 64 | 0.0091 | 0.03 |
| removal over TQ-positive only (original M1) | 32 | 0.0091 | 0.03 |

The whole-TQ-class radius certifies the strictly larger discrete set, so the removal guarantee now reads: *for every sequence with task token TQ (any quote content, any bracket content) and every embedding perturbation up to the certified radius, the skill-A readout stays ≤ 0.*

## Half 2 — the reach of the continuous ball

Smallest L∞ distance between two distinct content-token embeddings (the smallest in-vocabulary single-character swap, same metric as the certified ball): **2.940** (median 3.496, over 6 token pairs). Certified removal radius over the whole TQ class: **0.0091**. A token swap is **~324×** the certified radius.

So the *continuous* embedding ball is orders of magnitude too small to reach a discrete token substitution — the radius is a genuine continuous-robustness number, NOT a disguised discrete-robustness claim. The two guarantees are complementary and neither is oversold: **discrete closure** over the content vocabulary comes from enumerating the class (half 1); **continuous robustness** comes from the ball (the radius). This is the quantified version of the paper's §VI scope note.

Total time 585.0s.