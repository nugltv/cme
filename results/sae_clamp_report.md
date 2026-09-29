# SAE feature-clamp: a certified edit on the sparse-autoencoder surface

Output of `run_sae_clamp.py`. Sparse-autoencoder feature-clamps are the edit type practitioners most associate with steering/unlearning. This shows the clamp is **exactly certifiable** like every other edit: an SAE is piecewise-linear (ReLU encoder + linear decoder), so clamping a feature — applied as the additive correction `a_edited = a - f_k · Wd[:,k]` — encodes exactly in Z3 rationals, and removal + preservation are proved over a whole input region with a certified radius.

Toy model accuracy: skill A 0.9992, skill B 1.0000. SAE reconstruction MSE 0.00007. Clamped feature set: [12, 13, 28] (3 features — the smallest top-n set that removes skill A numerically, since the concept splits across several SAE features). After the clamp, numeric skill A accuracy falls to 0.4983 while skill B holds at 0.9885.

## The certificates (SAE-clamp, over the whole region)

| claim | verdict | grid cross-check |
|---|---|---|
| removal: skill A gone over A-region | PROVED | 0 violations (agree) |
| preservation: skill B HIGH over B-high | PROVED | 0 violations (agree) |
| preservation: skill B LOW over B-low | PROVED | 0 violations (agree) |

Control (unedited skill A over the A-region, want positive): **holds — skill A is present before the edit** — confirming the clamp, not the region, removes the skill.

## Certified input-perturbation radii — SAE-clamp vs ablation

| edit | removal radius | preservation radius |
|---|---|---|
| SAE feature-clamp (feature [12, 13, 28]) | >= 0.5 (cap) | 0.064 |
| ablation [[3]] (reference) | >= 0.5 (cap) | 0.100 |

## What this establishes

**The SAE feature-clamp certifies removal and preservation over the whole region, exactly** — it is not a special case the method can't reach, but one more piecewise-linear edit. Every proof agrees with the brute-force grid. The comparison row shows how its certified radii sit relative to a surgical ablation of the same skill on the same model. Either way, the SAE-clamp is handled by the identical exact-fraction machinery (`verify.prove_forall` / `certified_radius` via a `logits_fn` hook), so the edit taxonomy now spans ablation, weight-edit, steering, and the SAE feature-clamp.