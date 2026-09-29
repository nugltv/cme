# Noise-robust CERTIFIED preservation of skill B (modular adders)

Output of `run_rung3_preservation.py`. `run_rung3.py` certifies *removal* of skill A exactly over all sequences × continuous embedding noise, and checks *preservation* of skill B exhaustively over the clean sequences (no noise). Here skill B's preservation is certified **noise-robustly**, at the **certified** size (mod 5, d_model 8, 2 heads) — nothing shrunk.

Edit: ablate skill A's circuit [('head', 0)]. Skill B accuracy 1.000 → 1.000. Concrete single-sequence encoding validated against the float forward (max logit gap 1.1e-14).

## The claim, and why *this* claim

Preservation = the edited model still computes (a2+b2) mod p under SUB. We certify it as **argmax correctness**: for every SUB sequence and every embedding perturbation up to the certified radius, the correct class stays strictly greatest. (The stronger *exact-logit-equality* claim — the edit moves NO logit — times out at this size: the ablated head nudges SUB logits by an argmax-preserving sliver, so the solver hunts a hard witness and times out. Correctness is the right, lighter claim: only the noise is symbolic per sequence.)

## Results

- **eps = 0, exact rationals:** 625/625 SUB sequences certified correct — the same exhaustive coverage as before, but now an **exact-arithmetic proof**, not a float evaluation.
- **Noise-robust certified preservation radius: >= 0.002.** For every SUB sequence and every embedding perturbation up to this radius, skill B's answer is provably correct. Certified over an ascending grid (all 625 sequences re-checked at each eps):

| eps | all 625 certified? | time |
|---|---|---|
| 0.002 | yes | 278.2s |

So the adder subject certifies **both** sides of its claim over continuous noise: removal of skill A at radius ≥ 0.05 (the strong summand-independence claim, `run_rung3.py`), and preservation of skill B at radius >= 0.002 here. The preservation radius is smaller — a measured fact about the *exact solver*, not the model's robustness. Removal reaches ≥ 0.05 because its two-copy (siamese) encoding shares the noise between the copies, so most gate case-splits cancel; per-sequence correctness has no such cancellation, so under noise the piecewise-linear gates branch and a **minority of sequences have highly variable solve times** (the same exact-solver frontier as the threshold-gate size ladder, Table III). Individually those sequences still certify (in tenths of a second), but a single global radius over **all 625 at once** is what times out; we report the radius at which the whole space certifies reproducibly rather than inflating it.

Total time 597.4s on a laptop CPU.