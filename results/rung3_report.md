# A certified edit on a known-formula model: two modular adders

Output of `run_rung3.py`. Subject: two independent modular adders on disjoint positions (mod 5, `mod_arith_model.py` mode 'twoadd'), d_model 8, 2 heads, L=5 — 40 continuous noise variables, under the exact frontier of the threshold-gate transformer (~48). Sequence [a1, b1, a2, b2, TASK]: skill A = (a1+b1) mod p (positions 0,1, to REMOVE), skill B = (a2+b2) mod p (positions 2,3, to PRESERVE). The disjoint inputs are the point — skills sharing the same tokens/positions are entangled and cannot be separately edited; disjoint positions give the threshold-gate transformer's kind of separability, with arithmetic.

**Training.** Both skills learned exactly: A (a1+b1) 1.0000, B (a2+b2) 1.0000, hardened + quantized to 2^-12.

**Skill A's circuit (removal objective):** [('head', 0)].

**The edit (ablation), numerically:**

- Skill A accuracy 1.000 -> **0.200** (chance 0.200) — removed.
- Skill B accuracy 1.000 -> **1.000** — preserved.
- **Strong removal fact:** for **100%** of distractor pairs (a2,b2), the edited skill-A output is constant over all 5×5 summand pairs (a1,b1) — i.e. the edited model is **independent of the summands**: it does not merely lose accuracy, it provably (numerically here) no longer *reads* the numbers it is supposed to add, so it cannot compute the sum. This is the claim the exact certificate below proves over all sequences and continuous embedding noise, via the two-copy/siamese encoding.

## The exact certificate (all sequences × continuous embedding noise)

The p-way two-copy encoding was validated against the float forward (max gap 4.0e-15). Then, over the hull relaxation (a superset region — a proof certifies the discrete claim by P1), with weights as exact rationals:

- **Removal — PROVED.** For every sequence and every embedding perturbation up to the certified radius **>= 0.05 (cap)**, changing the summands (a1,b1) moves **no** output logit at all: the edited model is *exactly* independent of the numbers it should add, so it provably cannot compute (a1+b1) mod 5. This is the strong 'no longer reads the operands' removal, against a known formula.
- **Control — correctly refuted.** The *unedited* model's summand-independence is refuted (a summand-only change does move an output) — it genuinely reads the summands, as it must to do the skill. The certificate distinguishes the two.
- **Preservation — certified.** Over the ENTIRE clean input space — all 625 SUB sequences, not a sample — the edited model still computes (a2+b2) mod 5 at 100% accuracy, identical to the un-edited model on 100% of them. This is now certified two ways by the companion `run_rung3_preservation.py`: an **exact-rational** proof of correct classification over all 625 clean sequences (eps=0), and a **noise-robust** certificate — correct for every SUB sequence and every embedding perturbation up to radius ≥ 0.002 (`results/rung3_preservation_report.md`). So the adder certifies BOTH sides over continuous noise: removal ≥ 0.05, preservation ≥ 0.002. Preservation's smaller radius is an exact-solver frontier, not model fragility — the argmax-correctness claim lacks removal's two-copy cancellation, so a minority of sequences branch under noise.)

## Why this matters

Skill removal is now against a KNOWN FORMULA: 'the model no longer computes (a1+b1) mod p' is exact and checkable, not a threshold on an arbitrary property. And the removal is the strong kind — provable independence of the operands, the transformer analogue of the toy model's 'the head no longer listens' claim (results/independence_report.md), here meaning 'the adder no longer reads its summands'. And it is now proved exactly — over every sequence and a continuous cloud of embedding perturbations, not a test set — on a model with attention and a known-formula skill.

Total time 224.0s on a laptop CPU.