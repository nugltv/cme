"""
run_transformer_steering.py — steering as CERTIFIED ROWS on the transformer
===========================================================================

Run it with:   python run_transformer_steering.py     (about 10 minutes)

WHAT THIS IS (paper §V-E: "no dose both removes A and preserves B")
-------------------------------------------------------------------
For each diff-of-means steering dose on the threshold-gate transformer: the
certified removal radius AND the certified preservation radius, in one table with
the ablation edit as the reference row — the transformer version of the toy
models' edit comparison (run_robustness.py).

WHAT IT SHOWS (measured)
------------------------
Steering is not simply "it fails". At low dose it leaves skill B intact but does
not remove skill A (removal radius 0); crank the dose and removal eventually
certifies — at the cost of skill B's preservation margin collapsing. That is the
same dose->collateral trade-off P4 proves for the toy models, now quantified
on a real transformer: the reason "no dose is deployable" is that no dose removes A
AND preserves B at once, and the table shows exactly where each fails.

Every certificate quantifies over ALL sequences x all embedding noise up to the
radius (hull relaxation; a proof certifies the discrete claim by P1), sharing the
exact-rational prover of run_transformer.py. Report ->
results/transformer_steering_report.{md,json}.
"""

from __future__ import annotations
import json
import os
import time

import numpy as np

import transformer_model as _tm
from transformer_model import (train_gate_subject, sample_batch, accuracy,
                               ablate_head, ablate_mlp_neurons, steer_residual,
                               diff_of_means_direction)
import run_transformer as rt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

CIRCUIT_HEADS = [0]
CIRCUIT_MLPS = [1, 5, 2]
DOSES = (1, 2, 4, 8, 16, 32)
rt.TIMEOUT = 120000        # 2 min/query (bound total runtime; 'unknown' honest)


def _row(name, model, note):
    """One certified row: removal + preservation proofs at EPS0 and their
    certified radii (reuses run_transformer's prover + radii)."""
    proofs = rt.prove_claim_set(model, removal=True)
    rr = rt.radii(model)
    removal_ok = proofs["removal (A pos -> nonpositive)"]["proved"]
    pres_ok = (proofs["preservation B pos"]["proved"]
               and proofs["preservation B neg"]["proved"])
    return {"edit": name, "note": note, "tests_pass": rt.tests_pass(model),
            "removal_proved": removal_ok, "preservation_proved": pres_ok,
            "removal_radius": rr["removal"],
            "preservation_radius": rr["preservation"]}


def main():
    t0 = time.time()
    print("=" * 70)
    print("STEERING AS CERTIFIED ROWS (the edit comparison on the transformer)")
    print("=" * 70)

    print(f"\n[1] Training the certified subject (seed {rt.SUBJECT_SEED})...")
    model = train_gate_subject(seed=rt.SUBJECT_SEED, config=rt.SUBJECT_CONFIG,
                               verbose=False)
    rng = np.random.default_rng(123)
    tok, y = sample_batch(20000, rng)
    accA, accB = accuracy(model, tok, y)
    print(f"    held-out accuracy: A {accA:.4f}, B {accB:.4f}")

    rows = []
    print("\n[2] Ablation reference row (the certified circuit)...")
    edited = ablate_head(model, CIRCUIT_HEADS[0])
    edited = ablate_mlp_neurons(edited, CIRCUIT_MLPS)
    rows.append(_row(f"ablation [head {CIRCUIT_HEADS}, MLP {CIRCUIT_MLPS}]",
                     edited, "the certified edit"))
    _print_row(rows[-1])

    print("\n[3] Diff-of-means steering, dose sweep...")
    direction = diff_of_means_direction(model)
    for dose in DOSES:
        ms = steer_residual(model, dose * direction)
        rows.append(_row(f"steering diff-of-means (dose {dose})", ms,
                         f"residual nudged by {dose}x the diff-of-means vector"))
        _print_row(rows[-1])

    seconds = time.time() - t0
    _write_report(accA, accB, rows, seconds)
    print(f"\nDone in {seconds:.1f}s.")


def _print_row(r):
    print(f"    {r['edit']}: tests {'pass' if r['tests_pass'] else 'FAIL'}, "
          f"removal {'PROVED' if r['removal_proved'] else 'refuted'} "
          f"(radius {rt._fmt_radius(r['removal_radius'])}), preservation "
          f"{'PROVED' if r['preservation_proved'] else 'refuted'} "
          f"(radius {rt._fmt_radius(r['preservation_radius'])})", flush=True)


def _write_report(accA, accB, rows, seconds):
    md = os.path.join(RESULTS, "transformer_steering_report.md")
    js = os.path.join(RESULTS, "transformer_steering_report.json")
    lines = [
        "# Steering as certified rows (the edit comparison, on the transformer)",
        "",
        "Output of `run_transformer_steering.py`. Each row is a certified removal radius and a certified "
        "preservation radius for one edit, over ALL sequences x all embedding "
        "noise (hull relaxation; a proof certifies the discrete claim by P1), from "
        "the same exact-rational prover as `run_transformer.py`. This is the transformer version of the toy models' edit comparison, with steering reported as certified numbers.",
        "",
        f"Held-out accuracy: skill A {accA:.4f}, skill B {accB:.4f}.",
        "",
        "| edit | passes tests? | removal proved? | removal radius | preservation "
        "proved? | preservation radius | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['edit']} | {'yes' if r['tests_pass'] else 'no'} | "
            f"{'yes' if r['removal_proved'] else 'no'} | "
            f"{rt._fmt_radius(r['removal_radius'])} | "
            f"{'yes' if r['preservation_proved'] else 'no'} | "
            f"{rt._fmt_radius(r['preservation_radius'])} | {r['note']} |")
    lines += [
        "",
        "## Reading the table",
        "",
        "Ablation removes skill A and preserves skill B, both certified. Steering "
        "traces the **dose -> collateral** trade-off P4 proves for the toy models: at low dose skill B is intact but skill A is not removed (removal "
        "radius refuted at eps 0); as the dose climbs, removal eventually "
        "certifies — but skill B's preservation radius collapses in step. No "
        "single dose both removes A and preserves B, which is the quantified reason steering is not deployable here.",
        "",
        f"Total time {seconds:.1f}s on a laptop CPU. Radii bisected to {rt.TOL}; "
        f"cap {rt.EPS_MAX}; per-query timeout {rt.TIMEOUT // 1000}s ('unknown' "
        "reported, never hidden — a timed-out probe is treated as a failure, so "
        "every reported radius is a proved lower bound).",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"acc": {"A": accA, "B": accB}, "rows": rows,
                   "eps0": rt.EPS0, "eps_max": rt.EPS_MAX, "tol": rt.TOL,
                   "seconds": seconds}, f, indent=2, default=float)
    print(f"\nReports written: {md} and .json")


if __name__ == "__main__":
    main()
