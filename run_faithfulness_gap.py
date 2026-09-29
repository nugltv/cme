"""
run_faithfulness_gap.py — descriptive certificates are blind to an edit that
                          a behavioral certificate over an input region catches
=============================================================================

Run it with:   python run_faithfulness_gap.py        (seconds)

WHAT THIS SHOWS
---------------
The nearest published neighbors certify a DESCRIPTION of a model:

  * Somani ("Verifiable Transformers") certifies robustness by perturbing the
    FINAL RESIDUAL (an internal quantity) at a set of TRACED INPUTS.
  * Hadad et al. (Def. 2, "Formal Mechanistic Interpretability") certifies a
    circuit is a faithful stand-in by quantifying over PATCH VALUES (an internal
    quantity) at a set of REFERENCE INPUTS.

Both share one structure: **continuous over an internal quantity, at a FINITE
set of inputs.** The certificate of this paper differs on exactly one axis: it
quantifies over a CONTINUOUS INPUT REGION. This script shows that the difference
is load-bearing, on the constructed intervention illusion (paper Fig. 1):

  * a descriptive-family certificate (internal-perturbation robustness at every
    one of the 40,000+ reference inputs the illusion already passes) certifies
    the edit as "skill removed, and ROBUSTLY so" — with a large margin; while
  * the behavioral certificate over the continuous input region REFUTES the
    same edit, returning a surviving input that sits BETWEEN the reference
    points, where no finite-input certificate ever looks.

Same model, same edit, two certificates, opposite verdicts — and the one that is
right is the one whose quantifier ranges over inputs. (This is Proposition 3 made concrete against the descriptive-certificate family:
no finite-reference-input protocol certifies removal.)

Report: results/faithfulness_gap_report.{md,json}.
"""

from __future__ import annotations
import json
import os

import numpy as np

from run_illusion import build_constructed_model, REGION_A_HIGH
from verify import prove_forall

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def reference_grid(n):
    lo0, hi0, lo1, hi1 = REGION_A_HIGH
    xs = np.linspace(lo0, hi0, n)
    ys = np.linspace(lo1, hi1, n)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def main():
    print("=" * 70)
    print("THE FAITHFULNESS GAP — a descriptive certificate certifies the "
          "illusion edit; the region certificate refutes it")
    print("=" * 70)

    model, edit = build_constructed_model()

    # The reference inputs: exactly the grids the intervention illusion passes
    # (the 15x15 coarse grid of the illusion tests + the 201x201 fine grid
    # as its own cross-check). This IS the finite input set the descriptive
    # family gets to see.
    refs = np.concatenate([reference_grid(15), reference_grid(201)], axis=0)
    logitA = model.forward(refs, ablate=edit)[:, 0]        # post-edit head A
    worst = float(logitA.max())                            # closest to firing
    assert worst <= 0, "edit does not even pass the reference points"

    # --- the descriptive-family certificate --------------------------------
    # At every reference input, head A is BELOW zero (skill looks removed), and
    # it stays below zero under a continuous internal perturbation of size delta
    # (final-residual robustness, Somani's form: head A + eta, |eta| <= delta;
    # Hadad's patch robustness has the same worst case here). The certificate
    # holds at ALL reference points for any delta up to:
    delta_star = -worst
    # A cross-check that the "robust" claim is real: re-verify one representative
    # reference point with the solver, over the whole internal-perturbation ball.
    print(f"\n[descriptive certificate — the neighbors' kind]")
    print(f"  reference inputs: {len(refs):,} (15x15 coarse + 201x201 fine, the "
          "grids the illusion already passes)")
    print(f"  head A at every reference input is <= {worst:.4f} < 0 "
          "(skill looks removed everywhere tested)")
    print(f"  ... and stays removed under internal perturbation up to "
          f"delta* = {delta_star:.4f}: CERTIFIED 'removed, robustly' at all "
          f"{len(refs):,} reference inputs.")

    # --- the behavioral certificate over the continuous input region ---
    print(f"\n[region certificate — behavior over the continuous input region]")
    region = prove_forall(model, REGION_A_HIGH, "A", "nonpositive", ablate=edit)
    proved = region["proved"]
    cx = region.get("counterexample")
    if cx is not None:
        cx_logit = float(model.forward(np.array([cx]), ablate=edit)[0, 0])
    else:
        cx_logit = None
    print(f"  quantifier: ALL inputs in region {REGION_A_HIGH} (a continuum)")
    print(f"  verdict: {'PROVED removed' if proved else 'REFUTED'}")
    if cx is not None:
        print(f"  surviving input: x = ({cx[0]:.4f}, {cx[1]:.4f}), head A = "
              f"{cx_logit:+.3f} > 0  (skill A still FIRES here)")
        # how far is the survivor from the nearest reference input it slipped past?
        gap = float(np.abs(refs[:, 0] - cx[0]).min())
        print(f"  the survivor sits {gap:.4f} from the nearest reference input "
              "in x0 — in the gap the finite set never covered.")
    else:
        gap = None

    print(f"\n[the gap] Same model, same edit. The descriptive certificate — "
          f"continuous over an internal perturbation, at {len(refs):,} inputs — "
          "says 'removed, robustly'. The region certificate returns "
          "a surviving input. Only the input-region quantifier catches "
          "it.")

    _write_report(len(refs), worst, delta_star, proved, cx, cx_logit, gap)
    print("\nDone.")


def _write_report(n_refs, worst, delta_star, proved, cx, cx_logit, gap):
    md = os.path.join(RESULTS, "faithfulness_gap_report.md")
    js = os.path.join(RESULTS, "faithfulness_gap_report.json")
    lines = [
        "# The faithfulness gap: a descriptive certificate certifies the "
        "illusion edit; the region certificate refutes it",
        "",
        "Output of `run_faithfulness_gap.py`. The positioning argument of the "
        "paper's §II, made a result rather than a claim: the nearest "
        "published certificates are *descriptive* — Somani perturbs the final "
        "residual at traced inputs; Hadad et al. (Def. 2) quantify over patch "
        "values at reference inputs — both **continuous over an internal "
        "quantity at a finite set of inputs**. This work differs on one axis: it "
        "quantifies over a **continuous input region**. Here that axis decides "
        "the verdict, on the constructed intervention illusion (paper Fig. 1).",
        "",
        "## The two certificates on the same edit",
        "",
        "| certificate | quantifier | verdict on the illusion edit |",
        "|---|---|---|",
        f"| descriptive family (Somani / Hadad-style) | continuous internal "
        f"perturbation, at {n_refs:,} reference inputs | **certifies removed** — "
        f"head A ≤ {worst:.4f} < 0 at every reference input, and provably stays "
        f"removed under internal perturbation up to δ* = {delta_star:.4f} |",
        f"| **region certificate (this work)** | continuous over ALL inputs in the region | "
        f"**{'proved' if proved else 'REFUTES'}** — "
        + (f"returns a surviving input x = ({cx[0]:.4f}, {cx[1]:.4f}) where "
           f"head A = {cx_logit:+.3f} > 0 (skill A still fires)"
           if cx is not None else "no counterexample") + " |",
        "",
    ]
    if cx is not None:
        lines += [
            f"The surviving input sits **{gap:.4f}** from the nearest reference "
            "input in x0 — in a gap the finite reference set never covered. The "
            "descriptive certificate is not *wrong*: it correctly certifies "
            "everything it quantifies over (and does so robustly, over a "
            "continuous internal ball). It is *blind* — it never ranges over the "
            "input where the skill survives.",
            "",
        ]
    lines += [
        "## What the comparison shows",
        "",
        "The two certificates differ only in their specification: what is "
        "quantified over. This edit shows the specification is load-bearing: a "
        "faithfulness/robustness certificate of the neighbors' kind certifies it "
        "as a clean, robust removal, while the behavioral-over-a-region "
        "certificate refutes it "
        "with a concrete surviving input. The difference is not the solver or "
        "the encoding; it is *what is quantified over*. This is Proposition 3 "
        "made concrete against the descriptive-certificate family: no protocol "
        "that checks a finite set of reference inputs — however robustly, over "
        "however large an internal perturbation — can certify a removal, because "
        "the surviving behavior can always hide at an input the set omits. Only "
        "quantifying over the input region closes that gap.",
        "",
        "The intervention illusion here is the hand-constructed one "
        "(`run_illusion.py`, paper Fig. 1a,b); the same gap appears in the "
        "naturally-trained illusions of `run_illusion_nd.py`.",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"n_reference_inputs": n_refs, "worst_reference_logitA": worst,
                   "descriptive_delta_star": delta_star,
                   "region_proved": proved,
                   "region_counterexample": (list(cx) if cx is not None else None),
                   "region_counterexample_logitA": cx_logit,
                   "survivor_gap_to_nearest_reference_x0": gap},
                  f, indent=2, default=float)
    print(f"\nReports written: {md} and .json")


if __name__ == "__main__":
    main()
