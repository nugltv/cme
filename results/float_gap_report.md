# The float gap: what the certificates are about, exactly

Output of `run_float_gap.py`. The solver proves things about the IDEAL network (weights as exact fractions, real-number arithmetic); the code runs float64. This report shows the gap between the two on a rigged zero-margin claim, bounds it rigorously, and then shows every real certificate clears the gap by orders of magnitude — so the proofs apply to the program that actually runs. Plain-language story: the paper's float-gap appendix.

## [1] A zero-margin claim where solver and float grid legitimately disagree

Head A is built to be exactly 0 on the whole region, so 'head A <= 0' is TRUE and the solver proves it. The float program wobbles: 9170 of 40401 grid points land a hair above zero (worst: 1.11e-16); at 100,000 random inputs, 24.1% do. With a rounding-noise tolerance of 1e-12 the grid reports 0 violations. Both answers are correct — they describe two functions that differ by rounding noise. This is why the grid cross-check convention now carries a tolerance.

## [2] The gap is bounded

A conservative worst-case bound on |float64 forward − ideal forward| over the whole input domain, from standard floating-point error analysis: **4.66e-15** (adversarial model), **1.10e-13** (toy model).

## [3] Every real certificate carries over

Each toy-model certificate re-proved with slack **1e-09** — the logit must clear zero by that much, 9,062x the error bound:

| certificate (with slack) | proved? |
|---|---|
| removal: edited head A <= -slack on A's HIGH region | ✅ |
| preservation: edited head B >= slack on B's HIGH region | ✅ |
| preservation: edited head B <= -slack on B's LOW region | ✅ |
| control: unedited head A >= slack on A's HIGH region | ✅ |

Since the ideal network clears zero by 1e-09 everywhere and the float program stays within 1.10e-13 of it, the float program satisfies the plain claims (> 0 / <= 0) at every point of every region. Total time 11.5s.

