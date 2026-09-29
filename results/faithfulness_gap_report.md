# The faithfulness gap: a descriptive certificate certifies the illusion edit; the region certificate refutes it

Output of `run_faithfulness_gap.py`. The positioning argument of the paper's §II, made a result rather than a claim: the nearest published certificates are *descriptive* — Somani perturbs the final residual at traced inputs; Hadad et al. (Def. 2) quantify over patch values at reference inputs — both **continuous over an internal quantity at a finite set of inputs**. This work differs on one axis: it quantifies over a **continuous input region**. Here that axis decides the verdict, on the constructed intervention illusion (paper Fig. 1).

## The two certificates on the same edit

| certificate | quantifier | verdict on the illusion edit |
|---|---|---|
| descriptive family (Somani / Hadad-style) | continuous internal perturbation, at 40,626 reference inputs | **certifies removed** — head A ≤ -0.3000 < 0 at every reference input, and provably stays removed under internal perturbation up to δ* = 0.3000 |
| **region certificate (this work)** | continuous over ALL inputs in the region | **REFUTES** — returns a surviving input x = (0.8153, 0.0000) where head A = +0.000 > 0 (skill A still fires) |

The surviving input sits **0.0007** from the nearest reference input in x0 — in a gap the finite reference set never covered. The descriptive certificate is not *wrong*: it correctly certifies everything it quantifies over (and does so robustly, over a continuous internal ball). It is *blind* — it never ranges over the input where the skill survives.

## What the comparison shows

The two certificates differ only in their specification: what is quantified over. This edit shows the specification is load-bearing: a faithfulness/robustness certificate of the neighbors' kind certifies it as a clean, robust removal, while the behavioral-over-a-region certificate refutes it with a concrete surviving input. The difference is not the solver or the encoding; it is *what is quantified over*. This is Proposition 3 made concrete against the descriptive-certificate family: no protocol that checks a finite set of reference inputs — however robustly, over however large an internal perturbation — can certify a removal, because the surviving behavior can always hide at an input the set omits. Only quantifying over the input region closes that gap.

The intervention illusion here is the hand-constructed one (`run_illusion.py`, paper Fig. 1a,b); the same gap appears in the naturally-trained illusions of `run_illusion_nd.py`.