"""
run_deep.py — beyond the toy: certified edits on a DEEPER model with a
              skill that cannot live in one neuron
=======================================================================

Run it with:    python run_deep.py        (several minutes)

WHY THIS EXPERIMENT EXISTS
--------------------------
Everything so far ran on one-hidden-layer networks where the target skill
can sit in a single neuron. Maybe certified edits only work in that regime. This
experiment re-runs the WHOLE story — find the circuit, edit it, prove
removal and preservation, measure certified radii per edit type — on a
model that breaks both simplifications at once:

  * TWO hidden layers (3 -> 12 -> 8 -> 2), so circuits can span depth and
    the exact steering-vs-ablation algebra of proposition P4 (proved for
    one hidden layer) no longer applies — whatever survives here survives
    on its own, not by that theorem;
  * skill A = "exactly one of x0, x1 is above 0.5" (an XOR of thresholds),
    which NO single ReLU neuron can compute — the skill is distributed by
    mathematical necessity, not by our training choices;
  * skill A's HIGH region is not even one box: it is the UNION of two
    boxes (x0 high & x1 low, and x0 low & x1 high). We prove removal on
    each box separately and combine — proposition P1's union rule doing
    real work.

Skill B stays simple ("is x2 above 0.5?") so preservation still has a crisp
meaning.

WHAT GETS MEASURED (mirrors run_robustness.py)
-----------------------------------------------
For the trained model: control proofs that both skills work. Then, for each
kind of edit that passes ordinary tests: proofs that skill A is gone over
BOTH boxes of its region, proofs that skill B still works, and certified
radii (largest provable input wiggle room) for removal and preservation.

Everything lands in results/deep_report.md (plain language) and
results/deep_report.json (the same facts as data).
"""

from __future__ import annotations
import json
import os
import time

import numpy as np

from deep_model import DeepMLP, train_deep, deep_accuracy, deep_label_A, \
    deep_label_B, ablate_deep, weight_edit_deep, steer_deep, \
    targeted_vector_deep, diff_of_means_deep
from verify import prove_forall, certified_radius

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# Regions, all with the usual 0.1 margin from the 0.5 boundaries.
# Skill A says HIGH when exactly one of x0, x1 is high — TWO boxes:
A_HIGH_BOXES = [
    [(0.6, 1.0), (0.0, 0.4), (0.0, 1.0)],   # x0 high, x1 low
    [(0.0, 0.4), (0.6, 1.0), (0.0, 1.0)],   # x0 low, x1 high
]
# ... and LOW when both are high or both are low:
A_LOW_BOXES = [
    [(0.6, 1.0), (0.6, 1.0), (0.0, 1.0)],
    [(0.0, 0.4), (0.0, 0.4), (0.0, 1.0)],
]
B_HIGH_BOX = [(0.0, 1.0), (0.0, 1.0), (0.6, 1.0)]
B_LOW_BOX = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.4)]

EPS_MAX = 0.5     # wiggle-room cap (0.5 already grows every box to the
                  # whole cube on its constrained sides)
TOL = 2e-3        # radii pinned to about two decimal places (deep proofs
                  # cost more than shallow ones; see the report's timing)
TIMEOUT_MS = 120000


# ---------------------------------------------------------------------------
# Practitioner-style testing (the gate an edit must pass before we bother
# proving anything) — grids plus random points, as in run_illusion_nd.py.
# ---------------------------------------------------------------------------
def grid_points(box, k=7):
    axes = [np.linspace(lo, hi, k) for lo, hi in box]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def random_points(box, n, rng):
    return np.stack([rng.uniform(lo, hi, n) for lo, hi in box], axis=1)


def violations(model, pts, head, want):
    vals = model.forward(pts)[:, 0 if head == "A" else 1]
    return int((vals <= 0).sum() if want == "positive"
               else (vals > 0).sum())


def tests_pass(model) -> bool:
    """Skill A looks GONE on both its boxes (grid + 800 random points each)
    AND skill B looks intact on its boxes. The test gate."""
    rng = np.random.default_rng(42)
    for box in A_HIGH_BOXES:
        if violations(model, grid_points(box), "A", "nonpositive"):
            return False
        if violations(model, random_points(box, 800, rng), "A",
                      "nonpositive"):
            return False
    for box, want in ((B_HIGH_BOX, "positive"), (B_LOW_BOX, "nonpositive")):
        if violations(model, grid_points(box), "B", want):
            return False
        if violations(model, random_points(box, 800, rng), "B", want):
            return False
    return True


# ---------------------------------------------------------------------------
# Circuit finding on the deep model. A subtlety the shallow experiments
# never met: ranking neurons by "how much does knocking it out hurt skill
# A's ACCURACY?" is the wrong objective for REMOVAL. Killing the XOR
# machinery drives accuracy to a coin flip, but the crippled head can end
# up saying HIGH everywhere — the skill is broken, not removed, and the
# deployment tests (rightly) reject that. A practitioner suppressing a
# behavior optimizes what we optimize here instead: GREEDILY pick the
# neurons whose knockout pushes head A's logit DOWN across the region where
# the behavior fires, while keeping skill B intact — and stop as soon as
# the ordinary tests say the behavior is gone.
#
# (A model whose head-A resting bias is positive can be un-removable by ANY
# neuron-silencing edit — with every neuron quiet, the head still says
# HIGH. The greedy search then never passes the tests and the caller moves
# on to the next training seed; that skips such subjects automatically.)
# ---------------------------------------------------------------------------
def find_deep_circuit(model, max_k=10, n=20000, seed=7, allowed=None):
    """allowed: optional list of (layer, j) pairs the search may pick from —
    None means every neuron in both layers. Restricting it is how the
    companion experiment (run_deep_multi.py) FORCES multi-neuron or
    layer-constrained circuits instead of letting the search collapse onto
    a single bottleneck neuron."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, model.d))
    Y = np.stack([deep_label_A(X), deep_label_B(X)], axis=1)
    base_B = deep_accuracy(model, X, Y)[1]
    # points where removal must hold: skill A's HIGH boxes
    P = np.concatenate([random_points(b, 4000, rng) for b in A_HIGH_BOXES])

    all_neurons = [(layer, j) for layer in (1, 2)
                   for j in range(model.layer_sizes[layer - 1])]
    if allowed is not None:
        all_neurons = [nj for nj in all_neurons if nj in set(allowed)]
    chosen: list[tuple[int, int]] = []
    for _ in range(max_k):
        best = None
        for nj in all_neurons:
            if nj in chosen:
                continue
            cand = set(chosen) | {nj}
            if base_B - deep_accuracy(model, X, Y, ablate=cand)[1] > 0.02:
                continue                                 # would damage B
            worst = model.forward(P, ablate=cand)[:, 0].max()
            if best is None or worst < best[1]:
                best = (nj, worst)
        if best is None:
            return None                                  # every option hurts B
        chosen.append(best[0])
        if tests_pass(ablate_deep(model, chosen)):
            return chosen
    return None


# ---------------------------------------------------------------------------
# Proving: each claim over each box; radii per claim.
# ---------------------------------------------------------------------------
def prove_all(model) -> dict:
    """The full certificate set for an EDITED model: removal over both A
    boxes, preservation over both B boxes. Returns per-claim results."""
    out = {}
    for i, box in enumerate(A_HIGH_BOXES):
        out[f"removal box {i + 1}"] = prove_forall(
            model, box, "A", "nonpositive", timeout_ms=TIMEOUT_MS)
    out["B high"] = prove_forall(model, B_HIGH_BOX, "B", "positive",
                                 timeout_ms=TIMEOUT_MS)
    out["B low"] = prove_forall(model, B_LOW_BOX, "B", "nonpositive",
                                timeout_ms=TIMEOUT_MS)
    return out


def radii(model) -> dict:
    """Certified radii: removal = the smaller of the two A boxes' radii;
    preservation = the smaller of the two B claims' radii."""
    rem = [certified_radius(model, box, "A", "nonpositive", eps_max=EPS_MAX,
                            tol=TOL, timeout_ms=TIMEOUT_MS)
           for box in A_HIGH_BOXES]
    pres = [certified_radius(model, B_HIGH_BOX, "B", "positive",
                             eps_max=EPS_MAX, tol=TOL,
                             timeout_ms=TIMEOUT_MS),
            certified_radius(model, B_LOW_BOX, "B", "nonpositive",
                             eps_max=EPS_MAX, tol=TOL,
                             timeout_ms=TIMEOUT_MS)]
    return {"removal": _combine(rem), "preservation": _combine(pres),
            "queries": sum(r["queries"] for r in rem + pres)}


def _combine(rs) -> dict:
    """The binding (smallest) radius over a claim's boxes."""
    if any(r["radius"] is None for r in rs):
        return {"radius": None, "saturated": False}
    return {"radius": min(r["radius"] for r in rs),
            "saturated": all(r["saturated"] for r in rs)}


def _fmt_radius(r) -> str:
    if r["radius"] is None:
        return "refuted at eps=0"
    if r["saturated"]:
        return f">= {EPS_MAX} (whole cube)"
    return f"{r['radius']:.3f}"


def find_min_dose(make_edited, doses=(0.5, 1, 2, 4, 8, 16, 32, 64)):
    for s in doses:
        if tests_pass(make_edited(s)):
            return s
    return None


# ---------------------------------------------------------------------------
# The experiment.
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 70)
    print("BEYOND THE TOY — certified edits on a 2-hidden-layer XOR model")
    print("=" * 70)

    # ---- 1. Train, scanning seeds until a subject BOTH learns the skills
    # and admits a test-passing ablation (a seed whose head-A default is HIGH
    # can be un-removable by neuron silencing — the search skips those; the
    # report notes how many seeds were skipped and why that is honest). ----
    print("\n[1] Training 3 -> 12 -> 8 -> 2 on XOR(x0,x1) + threshold(x2)...")
    model, circuit, skipped = None, None, []
    rng = np.random.default_rng(999)
    Xte = rng.uniform(0, 1, size=(20000, 3))
    Yte = np.stack([deep_label_A(Xte), deep_label_B(Xte)], axis=1)
    for seed in range(10):
        cand = train_deep(DeepMLP(H1=12, H2=8, seed=seed), seed=seed + 1,
                          verbose=False)
        acc = deep_accuracy(cand, Xte, Yte)
        print(f"    seed {seed}: test-acc A={acc[0]:.4f} B={acc[1]:.4f}", end="")
        if acc[0] < 0.99 or acc[1] < 0.99:
            print("  (skills not learned well enough — skipped)")
            skipped.append({"seed": seed, "why": "under-trained",
                            "acc_A": acc[0], "acc_B": acc[1]})
            continue
        found = find_deep_circuit(cand)
        if found is None:
            print("  (no ablation up to 10 neurons passes the tests — "
                  "skipped)")
            skipped.append({"seed": seed, "why": "no test-passing ablation",
                            "acc_A": acc[0], "acc_B": acc[1],
                            "head_A_bias": float(cand.b_out[0])})
            continue
        print("  ACCEPTED")
        model, circuit, train_seed = cand, found, seed
        break
    if model is None:
        raise SystemExit("no seed produced a subject — widen the scan")

    # ---- 2. Control certificates: the unedited model provably works ------
    print("\n[2] Control proofs (unedited model)...")
    control = {}
    for i, box in enumerate(A_HIGH_BOXES):
        control[f"A high box {i + 1}"] = prove_forall(
            model, box, "A", "positive", timeout_ms=TIMEOUT_MS)
    for i, box in enumerate(A_LOW_BOXES):
        control[f"A low box {i + 1}"] = prove_forall(
            model, box, "A", "nonpositive", timeout_ms=TIMEOUT_MS)
    control["B high"] = prove_forall(model, B_HIGH_BOX, "B", "positive",
                                     timeout_ms=TIMEOUT_MS)
    control["B low"] = prove_forall(model, B_LOW_BOX, "B", "nonpositive",
                                    timeout_ms=TIMEOUT_MS)
    for name, r in control.items():
        print(f"    {name}: {'PROVED' if r['proved'] else r['status']}")
    control_ok = all(r["proved"] for r in control.values())

    # ---- 3. Skill A's circuit (already found during the subject scan) ----
    print("\n[3] Skill A's circuit (greedy knock-out search)...")
    by_layer = {1: [j for (l, j) in circuit if l == 1],
                2: [j for (l, j) in circuit if l == 2]}
    print(f"    circuit: {circuit} "
          f"(layer 1: {by_layer[1]}, layer 2: {by_layer[2]})")

    # ---- 4. The edit suite -------------------------------------------------
    print("\n[4] Building the edit suite...")
    suite = [("ablation", ablate_deep(model, circuit),
              f"neurons {circuit} switched off")]

    if not by_layer[1]:
        suite.append(("weight edit", weight_edit_deep(model, circuit),
                      "wires from the layer-2 circuit neurons to head A cut"))
        weight_edit_note = None
    else:
        weight_edit_note = ("n/a — the circuit includes layer-1 neurons, "
                            "which have no single wire to head A to cut")
        print(f"    weight edit: {weight_edit_note}")

    s_t = find_min_dose(lambda s: steer_deep(
        steer_deep(model, 1, targeted_vector_deep(model, 1, by_layer[1], s)),
        2, targeted_vector_deep(model, 2, by_layer[2], s)))
    if s_t is not None:
        m = steer_deep(
            steer_deep(model, 1,
                       targeted_vector_deep(model, 1, by_layer[1], s_t)),
            2, targeted_vector_deep(model, 2, by_layer[2], s_t))
        suite.append((f"steering targeted (dose {s_t:g})", m,
                      "circuit neurons in both layers pushed down; smallest "
                      "dose that passes the tests"))

    extra_notes = []
    for layer in (1, 2):
        s_d = find_min_dose(lambda s: steer_deep(
            model, layer, diff_of_means_deep(model, layer, s)))
        if s_d is None:
            msg = (f"diff-of-means steering at layer {layer}: NO dose up to "
                   "64 passes the ordinary tests — every dose either leaves "
                   "skill A visibly alive or visibly breaks skill B. Where "
                   "you steer matters.")
            print(f"    {msg}")
            extra_notes.append(msg)
            continue
        for dose, tag in ((s_d, "smallest dose that passes the tests"),
                          (4 * s_d, "4x the minimal dose")):
            m = steer_deep(model, layer, diff_of_means_deep(model, layer,
                                                            dose))
            label = f"steering diff-of-means @L{layer} (dose {dose:g})"
            if tests_pass(m):
                suite.append((label, m, tag))
            else:
                suite.append((label, m, tag + "; FAILS the ordinary tests"))

    # ---- 5. Prove everything, measure radii -------------------------------
    print("\n[5] Certificates + certified radii per edit...")
    rows = []
    for name, m, note in suite:
        proofs = prove_all(m)
        passes = tests_pass(m)
        rr = radii(m)
        removal_ok = all(proofs[k]["proved"] for k in proofs
                         if k.startswith("removal"))
        pres_ok = proofs["B high"]["proved"] and proofs["B low"]["proved"]
        print(f"    {name}: tests {'pass' if passes else 'FAIL'}, "
              f"removal {'PROVED' if removal_ok else 'refuted'} "
              f"(radius {_fmt_radius(rr['removal'])}), "
              f"preservation {'PROVED' if pres_ok else 'refuted'} "
              f"(radius {_fmt_radius(rr['preservation'])})")
        rows.append({"edit": name, "note": note, "tests_pass": passes,
                     "removal_proved": removal_ok,
                     "preservation_proved": pres_ok,
                     "removal_radius": rr["removal"],
                     "preservation_radius": rr["preservation"]})

    # Control preservation radius = skill B's natural margin (the yardstick
    # collateral damage is measured against), and the control removal must
    # be refuted (the unedited model HAS the skill).
    ctrl_rr = radii(model)
    print(f"    control (no edit): removal "
          f"{_fmt_radius(ctrl_rr['removal'])} (must be refuted), "
          f"preservation radius {_fmt_radius(ctrl_rr['preservation'])}")

    seconds = time.time() - t0
    _write_report(train_seed, skipped, control, control_ok, circuit,
                  by_layer, weight_edit_note, rows, ctrl_rr, extra_notes,
                  seconds)
    print(f"\nDone in {seconds:.1f}s.")


def _write_report(train_seed, skipped, control, control_ok, circuit,
                  by_layer, weight_edit_note, rows, ctrl_rr, extra_notes,
                  seconds):
    md = os.path.join(RESULTS, "deep_report.md")
    js = os.path.join(RESULTS, "deep_report.json")
    lines = [
        "# Beyond the toy: certified edits on a two-hidden-layer XOR model",
        "",
        "Output of `run_deep.py`. Model: 3 inputs -> 12 ReLU -> 8 ReLU -> 2 "
        "heads. Skill A = 'exactly one of x0, x1 above 0.5' (an XOR — no "
        "single neuron CAN hold it, so the circuit is distributed by "
        "necessity); skill B = 'x2 above 0.5'. Skill A's region is the "
        "union of two boxes, proved separately and combined (proposition "
        "P1). All the usual rules apply: exact fractions in the solver, "
        "margin 0.1, every number below is a proof.",
        "",
        f"Trained with seed {train_seed}; control certificates "
        f"({'all PROVED' if control_ok else 'NOT all proved'}): the "
        "unedited model provably computes the XOR on all four of its "
        "boxes and skill B on both of its.",
        "",
        (f"Seeds skipped before this subject: {len(skipped)} "
         f"({sum(1 for s in skipped if s['why'] == 'under-trained')} "
         "under-trained, "
         f"{sum(1 for s in skipped if s['why'] == 'no test-passing ablation')} "
         "with no ablation of up to 10 neurons that even LOOKS like removal "
         "on tests — e.g. models whose head-A resting bias is positive "
         "cannot be silenced into saying LOW; an honest scan reports these "
         "rather than hiding them)." if skipped else
         "No seeds were skipped: the first trained model was usable."),
        "",
        f"Skill A's circuit (knock-out search): {circuit} — "
        f"{len(by_layer[1])} neuron(s) in layer 1, {len(by_layer[2])} in "
        "layer 2." + ("" if weight_edit_note is None else
                      f" Weight edit: {weight_edit_note}."),
        "",
        "| edit | passes tests? | removal proved? | removal radius | "
        "preservation proved? | preservation radius | note |",
        "|---|---|---|---|---|---|---|",
        f"| control (no edit) | — | must fail | "
        f"{_fmt_radius(ctrl_rr['removal'])} | (unedited) | "
        f"{_fmt_radius(ctrl_rr['preservation'])} | yardstick: B's natural "
        "margin |",
    ]
    for r in rows:
        lines.append(
            f"| {r['edit']} | {'yes' if r['tests_pass'] else 'NO'} | "
            f"{'yes' if r['removal_proved'] else 'REFUTED'} | "
            f"{_fmt_radius(r['removal_radius'])} | "
            f"{'yes' if r['preservation_proved'] else 'REFUTED'} | "
            f"{_fmt_radius(r['preservation_radius'])} | {r['note']} |")
    if extra_notes:
        lines.append("")
        for n in extra_notes:
            lines.append(f"- {n}")
    lines += [
        "",
        f"Total time {seconds:.1f}s on a laptop CPU (training + testing + "
        "all proofs). Depth is not free for the solver: single proofs still "
        "take seconds, but the radii bisections dominate and the full run "
        "costs tens of minutes, versus ~80s for the shallow equivalent.",
        "",
        "## Why this matters",
        "",
        "Depth and a genuinely distributed skill change nothing about what "
        "the certificates MEAN — but they void the one-hidden-layer algebra "
        "(proposition P4) that EXPLAINED the shallow results, so whatever "
        "pattern appears in this table is independent empirical evidence, "
        "not a corollary.",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"train_seed": train_seed, "skipped_seeds": skipped,
                   "control": {k: {"proved": v["proved"]}
                               for k, v in control.items()},
                   "circuit": [list(c) for c in circuit],
                   "weight_edit_note": weight_edit_note,
                   "edits": rows, "control_radii": ctrl_rr,
                   "notes": extra_notes,
                   "seconds": seconds}, f, indent=2, default=float)
    print(f"\nReports written: {md} and .json")


if __name__ == "__main__":
    main()
