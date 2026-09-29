"""
run_float_gap.py
================

What, exactly, do the certificates talk about? (Paper App. B.)

The solver is fed the network's weights as EXACT fractions, so its proofs are
about the IDEAL network — the true real-number function those weights define.
But `model.forward` runs in float64 and picks up rounding noise of order 1e-16.
Those are two (very slightly) different functions. This script makes the gap
visible, shows when it matters and when it provably doesn't, and produces the
numbers reported in the paper's App. B:

  [1] THE GAP CAN BITE — on a claim with zero real margin. We build an
      adversarial model: head A computes
          ReLU(x0 - 0.5) + ReLU(x1 - 0.5) - ReLU(x0 + x1 - 1),
      which is EXACTLY ZERO everywhere on the region [0.6,1]^2 (the three
      terms cancel). So the claim "head A <= 0 on the region" is true, and
      Z3 correctly PROVES it. The float program, though, computes the three
      ReLUs with rounding and lands a hair above zero on a large fraction of
      inputs — so the brute-force grid reports thousands of "violations",
      every single one smaller than ~2e-16. The solver and the grid are BOTH
      right; they are answering questions about two different functions.
      (Hence the grid cross-check compares with a tolerance — see
      `grid_check(tol=...)` in verify.py.)

  [2] THE GAP IS BOUNDED — we compute a rigorous, conservative bound on how
      far the float64 forward pass can stray from the ideal network, using
      standard floating-point error analysis (each add/multiply is exact up
      to a relative 2^-53; propagate worst-case through the layers). For
      these models the bound comes out around 1e-14.

  [3] SO EVERY REAL CERTIFICATE CARRIES OVER — we re-prove the toy
      model's certificates with a SLACK: not just "logit <= 0" but "logit <= -1e-9",
      four orders of magnitude above the float error bound. All still prove.
      That closes the gap: if the ideal network clears zero by 1e-9 and the
      float program can differ by at most ~1e-14, then the float program's
      verdicts are pinned on the whole region too. The certificates are
      about the program you actually run, not just the ideal one.

See the paper's App. B.
Report: results/float_gap_report.md (+ .json).  Runtime: ~1 minute.
"""

from __future__ import annotations
import json, os, time
import numpy as np

from tiny_model import TinyMLP, train, find_skill_circuit
from verify import prove_forall, grid_check

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# The slack we demand of every re-proved certificate: the ideal network must
# clear zero by at least this much. Chosen to be astronomically above the
# float error bound (~1e-14) and astronomically below the models' real
# decision margins (~1e-1), so it changes nothing about what is provable.
GAMMA = 1e-9

REGION_A_HIGH = (0.6, 1.0, 0.0, 1.0)
REGION_B_HIGH = (0.0, 1.0, 0.6, 1.0)
REGION_B_LOW = (0.0, 1.0, 0.0, 0.4)


# ---------------------------------------------------------------------------
# [1] The adversarial model: exactly zero on the region, so the claim
#     "head A <= 0" is true (Z3 proves it) while the float program wobbles
#     to either side of zero.
# ---------------------------------------------------------------------------
def build_zero_margin_model() -> TinyMLP:
    m = TinyMLP(H=3, seed=0, d=2)
    m.W1 = np.array([[1.0, 0.0],    # ReLU(x0 - 0.5)
                     [0.0, 1.0],    # ReLU(x1 - 0.5)
                     [1.0, 1.0]])   # ReLU(x0 + x1 - 1)
    m.b1 = np.array([-0.5, -0.5, -1.0])
    m.W2 = np.array([[1.0, 1.0, -1.0],   # head A: the cancelling combination
                     [0.0, 0.0, 0.0]])   # head B: unused
    m.b2 = np.array([0.0, 0.0])
    return m


# ---------------------------------------------------------------------------
# [2] A rigorous bound on |float64 forward - ideal forward| over the domain.
#
# Standard floating-point error analysis (Higham-style): every float add or
# multiply returns the true result times (1 + delta) with |delta| <= u,
# u = 2^-53. Accumulated over a sum of n terms (in ANY order) plus the
# products feeding it, the error is at most gamma_{n+1} * (sum of the terms'
# magnitudes), where gamma_k = k*u / (1 - k*u). ReLU is exact in floats and
# shrinks differences (|ReLU(p) - ReLU(q)| <= |p - q|), so errors pass
# through it without growing. We propagate these worst cases layer by layer,
# bounding every intermediate magnitude over the whole input domain [0,1]^d.
# The result is deliberately conservative — a ceiling, not an estimate.
# ---------------------------------------------------------------------------
def float_error_bound(model: TinyMLP) -> float:
    u = 2.0 ** -53
    def gamma(k):
        return k * u / (1.0 - k * u)

    d = model.W1.shape[1]
    # Hidden pre-activation z_j = b1_j + sum_i W1_ji * x_i, with |x_i| <= 1:
    mag_z = np.abs(model.b1) + np.abs(model.W1).sum(axis=1)  # |z_j| ceiling
    err_hidden = gamma(d + 1) * mag_z                        # rounding in z_j
    mag_a = mag_z                                            # |ReLU(z)| <= |z|
    # Head logit = b2_h + sum_j W2_hj * a_j: its own rounding, plus the
    # hidden errors carried through the weights.
    H = model.H
    mag_head = np.abs(model.b2) + np.abs(model.W2) @ mag_a
    err_head = gamma(H + 1) * mag_head + np.abs(model.W2) @ err_hidden
    return float(err_head.max())


def main():
    t0 = time.time()
    print("=" * 70)
    print("THE FLOAT GAP — what the certificates are about, exactly")
    print("=" * 70)

    # ---------------- [1] the zero-margin adversarial model ----------------
    print("\n[1] A claim with ZERO real margin: ideal network exactly 0 on the "
          "region")
    zm = build_zero_margin_model()
    region = (0.6, 1.0, 0.6, 1.0)
    proof = prove_forall(zm, region, "A", "nonpositive")
    grid_strict = grid_check(zm, region, "A", "nonpositive", tol=0.0)
    grid_tol = grid_check(zm, region, "A", "nonpositive", tol=1e-12)

    rng = np.random.default_rng(0)
    pts = rng.uniform(0.6, 1.0, size=(100_000, 2))
    vals = zm.forward(pts)[:, 0]
    frac_pos = float((vals > 0).mean())
    max_val = float(vals.max())

    print(f"    solver on the ideal network : "
          f"{'PROVED' if proof['proved'] else 'NOT PROVED'} (head A <= 0)")
    print(f"    float grid, strict          : {grid_strict['violations_found']}"
          f" 'violations' / {grid_strict['grid_points']} points")
    print(f"    float grid, tol=1e-12       : {grid_tol['violations_found']}"
          f" violations / {grid_tol['grid_points']} points")
    print(f"    float forward at 100k random inputs: {frac_pos:.1%} land above "
          f"zero; the worst is {max_val:.2e} (pure rounding noise)")

    # ---------------- [2] the rigorous error bound ----------------
    print("\n[2] Rigorous ceiling on |float forward - ideal forward|:")
    bound_zm = float_error_bound(zm)
    print(f"    adversarial model : {bound_zm:.2e}")

    print("    (training the toy model to bound it too...)")
    model = train(TinyMLP(H=16, seed=0), verbose=False)
    bound_toy = float_error_bound(model)
    print(f"    toy model (tidy)  : {bound_toy:.2e}")
    assert max_val <= bound_zm, "observed float wobble exceeded the bound?!"
    assert bound_toy < GAMMA / 1000, "error bound not far below the slack?!"

    # ---------------- [3] margin transfer on the toy-model certificates ----
    print(f"\n[3] Re-proving the toy-model certificates with slack {GAMMA:.0e} "
          f"(≥ {GAMMA / bound_toy:,.0f}x the error ceiling):")
    circuit = find_skill_circuit(model, "A")
    claims = [
        ("removal: edited head A <= -slack on A's HIGH region",
         REGION_A_HIGH, "A", "nonpositive", circuit),
        ("preservation: edited head B >= slack on B's HIGH region",
         REGION_B_HIGH, "B", "positive", circuit),
        ("preservation: edited head B <= -slack on B's LOW region",
         REGION_B_LOW, "B", "nonpositive", circuit),
        ("control: unedited head A >= slack on A's HIGH region",
         REGION_A_HIGH, "A", "positive", None),
    ]
    rows = []
    for name, box, head, want, ablate in claims:
        r = prove_forall(model, box, head, want, ablate=ablate, slack=GAMMA)
        print(f"    [{'PROVED' if r['proved'] else 'NOT PROVED'}] {name}")
        rows.append({"claim": name, "proved": r["proved"], "status": r["status"]})
    all_ok = all(r["proved"] for r in rows)

    seconds = round(time.time() - t0, 1)
    print(f"\n{'ALL CERTIFICATES CARRY OVER TO THE FLOAT PROGRAM' if all_ok else 'SOMETHING FAILED — see above'}"
          f"  ({seconds}s)")

    # ---------------- report ----------------
    summary = {
        "zero_margin_demo": {
            "solver_proved": proof["proved"],
            "grid_strict_violations": grid_strict["violations_found"],
            "grid_tol_violations": grid_tol["violations_found"],
            "grid_points": grid_strict["grid_points"],
            "float_fraction_above_zero": frac_pos,
            "float_max_value": max_val,
        },
        "error_bounds": {"adversarial": bound_zm, "toy_model": bound_toy},
        "slack": GAMMA,
        "slack_certificates": rows,
        "all_carry_over": all_ok,
        "seconds": seconds,
    }
    with open(os.path.join(RESULTS, "float_gap_report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_markdown(summary)
    print("Report written: results/float_gap_report.md and .json")


def _write_markdown(s):
    z = s["zero_margin_demo"]
    lines = []
    lines.append("# The float gap: what the certificates are about, exactly\n")
    lines.append(
        "Output of `run_float_gap.py`. The solver proves things about the IDEAL "
        "network (weights as exact fractions, real-number arithmetic); the code "
        "runs float64. This report shows the gap between the two on a rigged "
        "zero-margin claim, bounds it rigorously, and then shows every real "
        "certificate clears the gap by orders of magnitude — so the proofs "
        "apply to the program that actually runs. Plain-language story: "
        "the paper's float-gap appendix.\n")
    lines.append("## [1] A zero-margin claim where solver and float grid "
                 "legitimately disagree\n")
    lines.append(
        f"Head A is built to be exactly 0 on the whole region, so 'head A <= 0' "
        f"is TRUE and the solver proves it. The float program wobbles: "
        f"{z['grid_strict_violations']} of {z['grid_points']} grid points land "
        f"a hair above zero (worst: {z['float_max_value']:.2e}); at 100,000 "
        f"random inputs, {z['float_fraction_above_zero']:.1%} do. With a "
        f"rounding-noise tolerance of 1e-12 the grid reports "
        f"{z['grid_tol_violations']} violations. Both answers are correct — "
        f"they describe two functions that differ by rounding noise. This is "
        f"why the grid cross-check convention now carries a tolerance.\n")
    lines.append("## [2] The gap is bounded\n")
    lines.append(
        f"A conservative worst-case bound on |float64 forward − ideal forward| "
        f"over the whole input domain, from standard floating-point error "
        f"analysis: **{s['error_bounds']['adversarial']:.2e}** (adversarial "
        f"model), **{s['error_bounds']['toy_model']:.2e}** (toy model).\n")
    lines.append("## [3] Every real certificate carries over\n")
    ratio = s["slack"] / s["error_bounds"]["toy_model"]
    lines.append(
        f"Each toy-model certificate re-proved with slack **{s['slack']:.0e}** — "
        f"the logit must clear zero by that much, {ratio:,.0f}x the error "
        f"bound:\n")
    lines.append("| certificate (with slack) | proved? |")
    lines.append("|---|---|")
    for r in s["slack_certificates"]:
        lines.append(f"| {r['claim']} | {'✅' if r['proved'] else '❌'} |")
    lines.append(
        f"\nSince the ideal network clears zero by {s['slack']:.0e} everywhere "
        f"and the float program stays within {s['error_bounds']['toy_model']:.2e} "
        f"of it, the float program satisfies the plain claims (> 0 / <= 0) at "
        f"every point of every region. Total time {s['seconds']}s.\n")
    with open(os.path.join(RESULTS, "float_gap_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
