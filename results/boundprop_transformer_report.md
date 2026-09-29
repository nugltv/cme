# M1: a certified edit on a softmax+LayerNorm transformer, past the exact-Z3 frontier

Output of `run_boundprop_transformer.py` (M1). The same kind of certified edit as on the threshold-gate transformer — remove skill A (quote), preserve skill B (bracket) over an embedding-space region, quantified over **every** sequence of each class — now on a **standard softmax + LayerNorm** transformer that the exact-Z3 pipeline cannot encode at all, at **448 noise variables** (L=7 x d_model=64) vs the exact frontier's ~48. Certificates are SOUND bound propagation (auto_LiRPA CROWN), validated against exact Z3 in M0; each is bracketed from above by a PGD embedding-space attack.

Model: d_model 64, 4 heads, 2 layers, d_mlp 128. Per-skill accuracy A 1.0000 / B 1.0000. Skill-A circuit ablated: attention knockout of quote positions [0, 1, 2] in all 2 layers. After the edit: skill A 1.000 -> 0.500 (chance 0.500), skill B 1.000 -> 1.000.

| claim | # sequences | certified radius (auto_LiRPA, sound) | PGD first break (upper bracket) |
|---|---|---|---|
| removal (A readout <= 0 over all TQ-positive seqs) | 32 | 0.0091 | 0.03 |
| preservation (B > 0 over all TB-positive seqs) | 32 | 0.0079 | 0.05 |
| preservation (B <= 0 over all TB-negative seqs) | 32 | 0.0098 | 0.1 |

Read each row as **certified ≤ true ≤ PGD-break**: the certified radius is a *sound lower bound* (nothing in that embedding ball breaks the claim, for any sequence of the class); the PGD break is an *empirical upper bound* (an attack succeeds there). A wide gap is bound-propagation looseness through softmax/LayerNorm — expected, and the reason the exact pipeline is used where it reaches; a PGD break *below* the certified radius would signal an unsound bug (none here).

## Design & scope (stated plainly)

- **The edit is a position-scoped attention knockout, not a head/MLP component circuit.** A from-scratch *standard softmax* transformer does not organise two skills into separable heads/neurons even when they read disjoint positions — greedy component ablation finds no single atom that removes skill A while preserving skill B. The skills ARE separable at the input: skill A reads only the quote positions, skill B only the bracket positions. So the mechanistic edit is a path-patch — zero the value contribution of the quote positions in every block, firewalling quote content out of the readout. This is why removal drives skill A exactly to chance (0.500) while skill B stays at 1.000.
- **Disjoint positions are a deliberate choice** (as on the modular adders), not the shared-position setup of the threshold-gate task. It is what makes a clean, certifiable separation exist inside a standard architecture.
- **Radii are over balls of radius ≥ 1e-4, not the exact point.** CROWN's softmax/LayerNorm relaxation is degenerate exactly at eps=0 (an internal convex-concave assert), but tight at any positive eps; we probe the smallest positive ball, which is a strictly stronger claim than the point — so this only ever under-claims.

## Why this matters

Exactness is not what limits the guarantee to small models: the certified removal + preservation claims hold on a standard-architecture transformer well past the exact frontier, over every sequence of each class × a continuous embedding ball, by a sound method whose pipeline was checked to *equal* the exact Z3 radius where both apply (M0). The price is incompleteness — hence the PGD bracket.

Total time 540.7s. Certified radii bisected to 0.0005, cap 0.05. Removal certified radius 0.0091; preservation (worse of B's two claims) 0.0079.