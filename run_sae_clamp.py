"""
run_sae_clamp.py — certify an SAE FEATURE-CLAMP edit (paper §V-E)
=============================================================================

Run it with:   python run_sae_clamp.py          (about a minute)

WHAT THIS ADDS
--------------
The sparse-autoencoder feature-clamp is the edit practitioners most associate
with steering and unlearning. This script shows it is not outside the method: because an SAE is
piecewise-linear (a ReLU encoder + a linear decoder), clamping a feature is one
more exactly-certifiable edit. On the toy model we:

  1. train the two-skill model (skill A = "x0 > 0.5", skill B = "x1 > 0.5");
  2. train a sparse autoencoder on its hidden activations;
  3. find the SAE feature that stands for skill A and CLAMP it to zero, applied
     as the standard additive correction  a_edited = a - f_k * Wd[:, k];
  4. PROVE, over the whole input region, that the clamp removes skill A and
     preserves skill B — and bisect the certified input-perturbation radius —
     using the SAME exact-fraction Z3 prover as every other experiment
     (verify.prove_forall / certified_radius with a logits_fn hook), plus a
     brute-force grid cross-check.

We also run the surgical ABLATION edit on the same model as a reference row, so
the SAE-clamp's certified radii sit next to a known-good edit's. The takeaway:
the feature clamp verifies exactly like the other edit types — removal and preservation
as proofs over a region, with a certified radius.

Report: results/sae_clamp_report.{md,json}.
"""

from __future__ import annotations
import json
import os

import numpy as np

from tiny_model import (TinyMLP, train, find_skill_circuit, accuracy,
                        true_label_A, true_label_B)
from sae_model import (train_sae, features_for_skill, clamp_forward,
                       build_logits_sae_clamp)
from verify import prove_forall, certified_radius

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

REGION_A_HIGH = (0.6, 1.0, 0.0, 1.0)   # skill A should say HIGH here
REGION_B_HIGH = (0.0, 1.0, 0.6, 1.0)   # skill B should say HIGH here
REGION_B_LOW = (0.0, 1.0, 0.0, 0.4)    # skill B should say LOW here
F_FEATURES = 32
EPS_MAX = 0.5
TOL = 1e-3


def _grid(box, n=201):
    lo0, hi0, lo1, hi1 = box
    g0, g1 = np.meshgrid(np.linspace(lo0, hi0, n), np.linspace(lo1, hi1, n))
    return np.stack([g0.ravel(), g1.ravel()], axis=1)


def _grid_violations(model, sae, k, box, col, want, n=201):
    """Brute-force cross-check on the FLOAT clamp forward (clamp_forward), the
    numpy twin of the exact encoding. Counts points that violate the claim."""
    pts = _grid(box, n)
    vals = clamp_forward(model, sae, k, pts)[:, col]
    return int((vals > 0).sum()) if want == "nonpositive" else int((vals <= 0).sum())


def main():
    print("=" * 70)
    print("SAE FEATURE-CLAMP — a certified edit on the sparse-autoencoder surface")
    print("=" * 70)

    print("\n[1] Training the two-skill toy model...")
    model = train(TinyMLP(H=16, seed=0), verbose=False)
    Xt = np.random.default_rng(0).uniform(0, 1, size=(4000, 2))
    Yt = np.stack([true_label_A(Xt), true_label_B(Xt)], axis=1)
    acc = accuracy(model, Xt, Yt)
    print(f"    trained accuracy: skill A {acc[0]:.4f}, skill B {acc[1]:.4f}")

    print("\n[2] Training a sparse autoencoder on the hidden activations...")
    rng = np.random.default_rng(3)
    Xs = rng.uniform(0, 1, size=(6000, 2))
    acts = np.maximum(Xs @ model.W1.T + model.b1, 0.0)
    sae = train_sae(acts, F=F_FEATURES, seed=0, verbose=True)
    recon_mse = float(((sae.reconstruct(acts) - acts) ** 2).mean())
    print(f"    SAE features: {F_FEATURES}, reconstruction MSE: {recon_mse:.5f}")

    print("\n[3] Selecting skill A's SAE feature SET (concepts split across "
          "features)...")
    # A concept usually splits across several SAE features, so a single clamp
    # under-removes. Pick the SMALLEST top-n feature set that numerically removes
    # skill A over the A-region on a grid (this is the practitioner's "clamp the
    # concept's features" move), then certify exactly that set.
    on = Xs[:, 0] > 0.5
    gridA = _grid(REGION_A_HIGH, 121)
    k = None
    for n in range(1, 13):
        cand = features_for_skill(sae, acts, on, n=n)
        if clamp_forward(model, sae, cand, gridA)[:, 0].max() <= 0:
            k = cand
            break
    if k is None:
        k = features_for_skill(sae, acts, on, n=12)   # report the honest miss
        print(f"    no set up to 12 features removes A on the grid; reporting "
              f"the top-12 set {k}")
    else:
        print(f"    smallest removing feature set: {k} ({len(k)} of "
              f"{F_FEATURES} features)")
    logits_fn = build_logits_sae_clamp(model, sae, k)

    # numeric check the clamp actually breaks A / keeps B (accuracy), for the report
    def clamp_acc(X):
        pred = (clamp_forward(model, sae, k, X) > 0).astype(float)
        yA = true_label_A(X); yB = true_label_B(X)
        return float((pred[:, 0] == yA).mean()), float((pred[:, 1] == yB).mean())
    caccA, caccB = clamp_acc(Xt)
    print(f"    after clamp: skill A accuracy {caccA:.4f}, skill B {caccB:.4f}")

    print("\n[4] PROVING the clamp's effect over the whole region (exact Z3)...")
    claims = [
        ("removal: skill A gone over A-region", REGION_A_HIGH, "A", "nonpositive", 0),
        ("preservation: skill B HIGH over B-high", REGION_B_HIGH, "B", "positive", 1),
        ("preservation: skill B LOW over B-low", REGION_B_LOW, "B", "nonpositive", 1),
    ]
    results = {}
    for name, box, head, want, col in claims:
        pf = prove_forall(model, box, head, want, logits_fn=logits_fn)
        viol = _grid_violations(model, sae, k, box, col, want)
        agree = (pf["proved"] and viol == 0) or (not pf["proved"] and viol > 0)
        print(f"    {name}: {'PROVED' if pf['proved'] else pf['status']}; "
              f"grid violations {viol} ({'agree' if agree else 'DISAGREE'})")
        results[name] = {"proved": pf["proved"], "status": pf["status"],
                         "grid_violations": viol, "grid_agrees": agree,
                         "counterexample": pf.get("counterexample")}

    # control: on the UNEDITED model skill A fires (removal claim should FAIL) —
    # confirms the clamp, not the region, is doing the work.
    ctrl = prove_forall(model, REGION_A_HIGH, "A", "positive")
    print(f"    control (unedited skill A over A-region, want positive): "
          f"{'holds (A present)' if ctrl['proved'] else ctrl['status']}")

    print("\n[5] Certified input-perturbation radii (SAE-clamp vs ablation)...")
    rr = {}
    rem = certified_radius(model, REGION_A_HIGH, "A", "nonpositive",
                           eps_max=EPS_MAX, tol=TOL, logits_fn=logits_fn)
    presb_hi = certified_radius(model, REGION_B_HIGH, "B", "positive",
                                eps_max=EPS_MAX, tol=TOL, logits_fn=logits_fn)
    presb_lo = certified_radius(model, REGION_B_LOW, "B", "nonpositive",
                                eps_max=EPS_MAX, tol=TOL, logits_fn=logits_fn)
    rr["sae_clamp"] = {"removal": rem["radius"],
                       "preservation": _min_radius(presb_hi, presb_lo)}
    print(f"    SAE-clamp: removal radius {_fmt(rem['radius'])}, "
          f"preservation radius {_fmt(rr['sae_clamp']['preservation'])}")

    # ablation reference on the same model
    circuit_A = find_skill_circuit(model, "A")
    rem_ab = certified_radius(model, REGION_A_HIGH, "A", "nonpositive",
                              ablate=circuit_A, eps_max=EPS_MAX, tol=TOL)
    presb_hi_ab = certified_radius(model, REGION_B_HIGH, "B", "positive",
                                   ablate=circuit_A, eps_max=EPS_MAX, tol=TOL)
    presb_lo_ab = certified_radius(model, REGION_B_LOW, "B", "nonpositive",
                                   ablate=circuit_A, eps_max=EPS_MAX, tol=TOL)
    rr["ablation"] = {"removal": rem_ab["radius"],
                      "preservation": _min_radius(presb_hi_ab, presb_lo_ab),
                      "circuit": circuit_A}
    print(f"    ablation [{circuit_A}]: removal radius {_fmt(rem_ab['radius'])}, "
          f"preservation radius {_fmt(rr['ablation']['preservation'])}")

    _write_report(acc, recon_mse, k, caccA, caccB, results, ctrl, rr)
    print("\nDone.")


def _min_radius(a, b):
    if a["radius"] is None or b["radius"] is None:
        return None
    return min(a["radius"], b["radius"])


def _fmt(r):
    return "refuted at eps=0" if r is None else (f">= {EPS_MAX} (cap)"
                                                 if r >= EPS_MAX else f"{r:.3f}")


def _write_report(acc, recon_mse, k, caccA, caccB, results, ctrl, rr):
    md = os.path.join(RESULTS, "sae_clamp_report.md")
    js = os.path.join(RESULTS, "sae_clamp_report.json")
    all_proved = all(v["proved"] for v in results.values())
    all_agree = all(v["grid_agrees"] for v in results.values())
    lines = [
        "# SAE feature-clamp: a certified edit on the sparse-autoencoder surface",
        "",
        "Output of `run_sae_clamp.py`. Sparse-autoencoder feature-clamps are the "
        "edit type practitioners most associate with steering/unlearning. This "
        "shows the clamp is **exactly certifiable** like every other edit: an SAE "
        "is piecewise-linear (ReLU encoder + linear decoder), so clamping a "
        "feature — applied as the additive correction `a_edited = a - f_k · "
        "Wd[:,k]` — encodes exactly in Z3 rationals, and removal + preservation "
        "are proved over a whole input region with a certified radius.",
        "",
        f"Toy model accuracy: skill A {acc[0]:.4f}, skill B {acc[1]:.4f}. SAE "
        f"reconstruction MSE {recon_mse:.5f}. Clamped feature set: {k} "
        f"({len(k)} features — the smallest top-n set that removes skill A "
        "numerically, since the concept splits across several SAE features). "
        f"After the clamp, numeric skill A accuracy falls to {caccA:.4f} while "
        f"skill B holds at {caccB:.4f}.",
        "",
        "## The certificates (SAE-clamp, over the whole region)",
        "",
        "| claim | verdict | grid cross-check |",
        "|---|---|---|",
    ]
    for name, v in results.items():
        lines.append(f"| {name} | {'PROVED' if v['proved'] else v['status']} | "
                     f"{v['grid_violations']} violations "
                     f"({'agree' if v['grid_agrees'] else 'DISAGREE'}) |")
    lines += [
        "",
        f"Control (unedited skill A over the A-region, want positive): "
        f"**{'holds — skill A is present before the edit' if ctrl['proved'] else ctrl['status']}** "
        "— confirming the clamp, not the region, removes the skill.",
        "",
        "## Certified input-perturbation radii — SAE-clamp vs ablation",
        "",
        "| edit | removal radius | preservation radius |",
        "|---|---|---|",
        f"| SAE feature-clamp (feature {k}) | {_fmt(rr['sae_clamp']['removal'])} "
        f"| {_fmt(rr['sae_clamp']['preservation'])} |",
        f"| ablation [{rr['ablation']['circuit']}] (reference) | "
        f"{_fmt(rr['ablation']['removal'])} | "
        f"{_fmt(rr['ablation']['preservation'])} |",
        "",
        "## What this establishes",
        "",
        ("**The SAE feature-clamp certifies removal and preservation over the "
         "whole region, exactly** — it is not a special case the method can't "
         "reach, but one more piecewise-linear edit. "
         if all_proved else
         "**On this model the SAE feature-clamp did NOT certify every claim** "
         "(see the table) — an honest finding: the reconstruction leaks the "
         "skill-A feature into directions the clamp doesn't fully cancel. ")
        + ("Every proof agrees with the brute-force grid. "
           if all_agree else "Note a proof/grid disagreement above. ")
        + "The comparison row shows how its certified radii sit relative to a "
        "surgical ablation of the same skill on the same model. Either way, the "
        "SAE-clamp is handled by the identical exact-fraction machinery "
        "(`verify.prove_forall` / `certified_radius` via a `logits_fn` hook), so "
        "the edit taxonomy now spans ablation, weight-edit, steering, and the "
        "SAE feature-clamp.",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"accuracy": {"A": acc[0], "B": acc[1]},
                   "sae_recon_mse": recon_mse, "clamped_feature": k,
                   "clamp_accuracy": {"A": caccA, "B": caccB},
                   "claims": results, "control_holds": ctrl["proved"],
                   "radii": rr}, f, indent=2, default=float)
    print(f"\nReports written: {md} and .json")


if __name__ == "__main__":
    main()
