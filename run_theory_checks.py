"""
run_theory_checks.py — machine-checked witnesses for the propositions P1–P4
============================================================================

Run it with:    python run_theory_checks.py        (takes under a minute)

THE IDEA, IN PLAIN WORDS
------------------------
The paper's theory section makes four formal claims,
P1–P4, each proved with pencil and paper. Pencil-and-paper proofs can contain
slips, and they can silently drift out of sync with what the code actually
does. So this script EXECUTES the checkable content of each proposition
against the real code:

  P2 — the "certified radius" (largest provable wiggle room) really is found
       within the promised number of solver calls, and re-checking just below
       and just above the returned radius gives proof / refutation as claimed.
  P3 — "no finite test can certify removal": for SEVERAL different test grids
       (10x10 up to 500x500 points), the proof's recipe builds an edit that
       passes every single grid point while the solver refutes the removal —
       with the surviving input exactly where the proof predicted it.
  P4 — the steering-vs-ablation algebra: the exact "leftover" formula matches
       the real models to floating-point precision; above the predicted dose
       threshold, steering and ablation become literally the same function;
       and the collateral-damage formula for realistic steering predicts skill
       B's logit shift exactly.

(P1 says "a claim proved on a region holds on every sub-region and on unions
of regions" — that is true by the meaning of the words 'for all', and there is
nothing beyond that to execute; every other experiment in the repo uses it
implicitly.)

Everything is written to results/theory_checks_report.md (plain language) and
results/theory_checks_report.json (the same facts as data).
"""

from __future__ import annotations
import json
import math
import os
import time

import numpy as np

from tiny_model import TinyMLP, train, find_skill_circuit, accuracy, \
    true_label_A, true_label_B
from verify import prove_forall, grid_check, inflate_box, certified_radius
from edits import apply_ablation, apply_steering, targeted_suppression_vector, \
    diff_of_means_vector

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# Same spec regions as the other experiments (claims stay away from the fuzzy
# 0.5 boundary; see run_slice.py).
REGION_A_HIGH = (0.6, 1.0, 0.0, 1.0)
REGION_B_HIGH = (0.0, 1.0, 0.6, 1.0)
REGION_B_LOW = (0.0, 1.0, 0.0, 0.4)


# ---------------------------------------------------------------------------
# Check for P2: the certified radius comes back within the promised budget,
# and is genuinely the tipping point (proves at the radius, fails just past).
# ---------------------------------------------------------------------------
def check_P2() -> dict:
    print("\n[P2] Certified radius: query budget + tipping-point re-check")
    eps_max, tol = 0.6, 1e-3
    budget = 2 + math.ceil(math.log2(eps_max / tol))   # the P2(d) bound

    # Subject: the tidy base toy model with its circuit ablated; the claim is
    # skill B's preservation on its HIGH region — chosen because its radius is
    # a genuine interior tipping point (~0.10), not a saturated one.
    base = train(TinyMLP(H=16, seed=0), verbose=False)
    circuit = find_skill_circuit(base, "A")
    edited = apply_ablation(base, circuit)

    r = certified_radius(edited, REGION_B_HIGH, "B", "positive",
                         eps_max=eps_max, tol=tol)
    within_budget = r["queries"] <= budget
    bracket_tight = (r["first_failure"] is not None
                     and r["first_failure"] - r["radius"] <= tol + 1e-12)

    # Independent re-check of the bracket: one fresh proof at the returned
    # radius (must prove) and one just past it (must fail).
    at_radius = prove_forall(edited, inflate_box(REGION_B_HIGH, r["radius"]),
                             "B", "positive")
    past_radius = prove_forall(edited,
                               inflate_box(REGION_B_HIGH, r["first_failure"]),
                               "B", "positive")
    reproved = at_radius["proved"] and not past_radius["proved"]

    ok = within_budget and bracket_tight and reproved
    print(f"    radius = {r['radius']:.3f} in {r['queries']} solver calls "
          f"(bound: {budget}); re-proved at radius, re-refuted at "
          f"{r['first_failure']:.3f} -> {'OK' if ok else 'MISMATCH'}")
    return {"name": "P2 certified radius", "passed": bool(ok),
            "radius": r["radius"], "first_failure": r["first_failure"],
            "queries": r["queries"], "query_budget": budget,
            "reproved_at_radius": at_radius["proved"],
            "refuted_past_radius": not past_radius["proved"]}


# ---------------------------------------------------------------------------
# Check for P3: the proof's recipe defeats ANY given test grid. We take four
# different grids and, for each, build the "tent" edit with the proof's own
# parameter choices; the grid must approve it and the solver must refute it.
# ---------------------------------------------------------------------------
def build_tent_model(c: float, w: float, beta: float):
    """
    The P3 construction with explicit parameters (the hand-built model of
    run_illusion.py is the instance c=0.815, w=0.0006, beta=1000): a main
    skill-A pathway (neurons 0,1 — these are the edit), a three-neuron "tent"
    that is zero everywhere except a bump of height beta*w around c, a skill-B
    pathway (neurons 5,6), and a head-A hurdle of beta*w/2. After ablating the
    main pathway, head A > 0 exactly on the sliver |x0 - c| < w/2.
    """
    m = TinyMLP(H=7, seed=0)
    m.W1 = np.array([
        [20.0, 0.0], [-20.0, 0.0],            # main A pathway (the edit)
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],   # tent parts
        [0.0, 20.0], [0.0, -20.0],            # skill B pathway
    ])
    m.b1 = np.array([-10.0, 10.0, -(c - w), -c, -(c + w), -10.0, 10.0])
    m.W2 = np.array([
        [1.0, -1.0, beta, -2.0 * beta, beta, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
    ])
    m.b2 = np.array([-beta * w / 2.0, 0.0])   # the hurdle
    return m, [0, 1]


def check_P3() -> dict:
    print("\n[P3] For ANY finite test grid, an edit exists that fools it")
    rows, all_ok = [], True
    for n in (10, 50, 100, 500):
        # The proof's recipe, step by step: the n grid lines on x0 leave gaps;
        # centre the sliver in the first gap, make it a quarter of a gap wide.
        lo0, hi0 = REGION_A_HIGH[0], REGION_A_HIGH[1]
        gap = (hi0 - lo0) / (n - 1)
        c = lo0 + gap / 2.0
        w = gap / 4.0
        beta = 0.6 / w                        # fixes the bump height at 0.6
        model, edit = build_tent_model(c, w, beta)

        # The practitioner's full test battery at this grid size: removal
        # looks done, and skill B looks fine, at every single grid point.
        removal_grid = grid_check(model, REGION_A_HIGH, "A", "nonpositive",
                                  ablate=edit, n=n)
        b_high = grid_check(model, REGION_B_HIGH, "B", "positive",
                            ablate=edit, n=n)
        b_low = grid_check(model, REGION_B_LOW, "B", "nonpositive",
                           ablate=edit, n=n)
        grid_fooled = (removal_grid["violations_found"] == 0
                       and b_high["violations_found"] == 0
                       and b_low["violations_found"] == 0)

        # The proposition also promises the pre-edit model is legitimate and
        # preservation genuinely holds — as PROOFS, not test results.
        legit = prove_forall(model, REGION_A_HIGH, "A", "positive")["proved"]
        pres = (prove_forall(model, REGION_B_HIGH, "B", "positive",
                             ablate=edit)["proved"]
                and prove_forall(model, REGION_B_LOW, "B", "nonpositive",
                                 ablate=edit)["proved"])

        # ... and the solver must refute the removal, with the survivor inside
        # the predicted sliver (c - w/2, c + w/2).
        refutation = prove_forall(model, REGION_A_HIGH, "A", "nonpositive",
                                  ablate=edit)
        refuted = (not refutation["proved"]
                   and refutation["counterexample"] is not None)
        cx0 = refutation["counterexample"][0] if refuted else None
        survivor_where_predicted = (refuted
                                    and abs(cx0 - c) <= w / 2.0 + 1e-12)

        ok = grid_fooled and legit and pres and refuted \
            and survivor_where_predicted
        all_ok = all_ok and ok
        pocket = w / (hi0 - lo0)
        print(f"    {n}x{n} grid ({n * n} points): fooled={grid_fooled}, "
              f"preservation proved={pres}, solver survivor at "
              f"x0={cx0:.6f} (predicted {c:.6f} +/- {w / 2:.6f}) "
              f"-> {'OK' if ok else 'MISMATCH'}")
        rows.append({"grid": f"{n}x{n}", "grid_points": n * n,
                     "sliver_center": c, "sliver_width": w,
                     "pocket_fraction_of_region": pocket,
                     "grid_fooled": grid_fooled,
                     "pre_edit_model_proved_correct": legit,
                     "preservation_proved": pres,
                     "solver_refuted_removal": refuted,
                     "survivor_x0": cx0,
                     "survivor_where_predicted": survivor_where_predicted,
                     "passed": bool(ok)})
    return {"name": "P3 illusion vs any grid", "passed": bool(all_ok),
            "grids": rows}


# ---------------------------------------------------------------------------
# Check for COROLLARY 1: even an ADAPTIVE tester — one that picks each next
# test point by looking at earlier answers — is defeated, provided it is
# deterministic and black-box. The fixed-grid check for P3 does not cover
# adaptive testers; this is the machine witness for them.
#
# The witness follows the proof exactly:
#   1. build a genuinely edited model g0 (the tidy base toy model, circuit
#      ablated — removal PROVED, nothing hidden);
#   2. run a deterministic adaptive tester against it: coarse grid first,
#      then repeated zooming toward wherever head A's logit is highest
#      (the closest to firing). It approves — rightly, this edit is real;
#   3. read the tester's TRANSCRIPT, place the proof's tent gadget in the
#      largest gap between the x0 values it chose to query, with the bump
#      high enough (beta*w/2 > a rigorous cap on |g0_A|) that the skill
#      genuinely fires inside the gap;
#   4. re-run the SAME tester on the gadgeted model: because the tent is
#      zero at every point the tester visits, it sees identical answers,
#      makes identical choices, and approves again — fooled;
#   5. the solver refutes the removal, with the survivor inside the gap,
#      and skill B's preservation still PROVES (the gadget is invisible to
#      every check the tester could run).
# ---------------------------------------------------------------------------
def _adaptive_tester(query_fn, region=REGION_A_HIGH, coarse=9, budget=300):
    """A deterministic, adaptive, black-box tester for 'skill A removed on
    the region': approve iff head A's logit stays <= 0 at every point it
    queries. Adaptivity: after a coarse grid, it repeatedly bisects the x0
    interval whose endpoints have shown the HIGHEST logits (the most alive
    corner of the model), at the x1 where that maximum was seen. Decisions
    quantize logits to 9 decimals so they depend on real signal, not
    float dust. Returns (approved, transcript)."""
    lo0, hi0, lo1, hi1 = region
    transcript = []                      # [(x0, x1, quantized logit), ...]

    def ask(x0, x1):
        v = round(query_fn(x0, x1), 9)
        transcript.append((x0, x1, v))
        return v

    seen = {}                            # x0 -> (best logit at x0, its x1)
    for x0 in np.linspace(lo0, hi0, coarse):
        for x1 in np.linspace(lo1, hi1, coarse):
            v = ask(float(x0), float(x1))
            if v > 0:
                return False, transcript
            if x0 not in seen or v > seen[x0][0]:
                seen[float(x0)] = (v, float(x1))
    while len(transcript) < budget:
        xs = sorted(seen)                # zoom where the model looks most alive
        pairs = [(max(seen[a][0], seen[b][0]), a, b)
                 for a, b in zip(xs, xs[1:])]
        _, a, b = max(pairs)             # deterministic: ties break on (a, b)
        mid = (a + b) / 2.0
        x1 = seen[a][1] if seen[a][0] >= seen[b][0] else seen[b][1]
        v = ask(mid, x1)
        if v > 0:
            return False, transcript
        seen[mid] = (v, x1)
    return True, transcript


def _add_tent(base, c, w, beta):
    """Append the three tent neurons to a model, feeding head A with the
    (beta, -2*beta, beta) pattern and head B with nothing — the Corollary 1
    gadget (no bias hurdle)."""
    m = TinyMLP(H=base.H + 3, seed=0)
    m.W1 = np.vstack([base.W1, [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    m.b1 = np.concatenate([base.b1, [-(c - w), -c, -(c + w)]])
    m.W2 = np.hstack([base.W2,
                      [[beta, -2.0 * beta, beta], [0.0, 0.0, 0.0]]])
    m.b2 = base.b2.copy()
    return m


def check_corollary1() -> dict:
    print("\n[Corollary 1] Even an adaptive (deterministic, black-box) tester "
          "is defeated")
    from verify import logit_magnitude_cap
    base = train(TinyMLP(H=16, seed=0), verbose=False)
    circuit = find_skill_circuit(base, "A")
    g0 = apply_ablation(base, circuit)   # a GENUINE edit, proved removed

    approved0, t0_transcript = _adaptive_tester(
        lambda x0, x1: float(g0.forward(np.array([[x0, x1]]))[0, 0]))
    print(f"    tester vs the genuine edit: "
          f"{'APPROVES' if approved0 else 'rejects'} "
          f"after {len(t0_transcript)} adaptive queries")

    # The gadget, placed by reading the transcript (the proof's step).
    lo0, hi0 = REGION_A_HIGH[0], REGION_A_HIGH[1]
    xs = sorted({q[0] for q in t0_transcript} | {lo0, hi0})
    gaps = [(b - a, a, b) for a, b in zip(xs, xs[1:])]
    gap, a, b = max(gaps)
    c, w = (a + b) / 2.0, gap / 4.0      # support (c-w, c+w) inside the gap
    G = logit_magnitude_cap(g0, "A")     # rigorous cap on |g0_A| anywhere
    beta = 2.0 * (G + 1.0) / w           # bump height beta*w = 2(G+1) > 2G
    g = _add_tent(g0, c, w, beta)
    f_parent = _add_tent(base, c, w, beta)
    print(f"    gadget placed in the widest un-queried gap: "
          f"c={c:.6f}, w={w:.2e} (bump height {beta * w:.1f} vs "
          f"logit cap {G:.1f})")

    # The gadgeted model is a legitimate subject: its unedited parent still
    # proves the skill, and the edit still proves preservation.
    legit = prove_forall(f_parent, REGION_A_HIGH, "A", "positive")["proved"]
    pres = (prove_forall(g, REGION_B_HIGH, "B", "positive")["proved"]
            and prove_forall(g, REGION_B_LOW, "B", "nonpositive")["proved"])

    # The same tester, re-run against the gadgeted model.
    approved1, t1_transcript = _adaptive_tester(
        lambda x0, x1: float(g.forward(np.array([[x0, x1]]))[0, 0]))
    same_queries = [q[:2] for q in t0_transcript] == \
        [q[:2] for q in t1_transcript]
    print(f"    tester vs the gadgeted model: "
          f"{'APPROVES' if approved1 else 'rejects'}, transcript "
          f"{'IDENTICAL' if same_queries else 'DIVERGED'} "
          f"({len(t1_transcript)} queries)")

    # The solver is not fooled.
    ref = prove_forall(g, REGION_A_HIGH, "A", "nonpositive")
    refuted = not ref["proved"] and ref["counterexample"] is not None
    cx0 = ref["counterexample"][0] if refuted else None
    in_gap = refuted and (c - w) < cx0 < (c + w)
    if refuted:
        print(f"    solver refutes the removal: survivor at x0={cx0:.6f} "
              f"(inside the tent support {c - w:.6f}..{c + w:.6f}: "
              f"{in_gap})")

    passed = (approved0 and approved1 and same_queries and legit and pres
              and refuted and in_gap)
    print(f"    -> {'OK' if passed else 'MISMATCH'}")
    return {"name": "Corollary 1 adaptive tester defeated",
            "passed": bool(passed),
            "queries": len(t0_transcript),
            "tester_approves_genuine_edit": bool(approved0),
            "tester_approves_gadgeted_model": bool(approved1),
            "transcripts_identical": bool(same_queries),
            "gadget": {"center": c, "halfwidth": w,
                       "bump_height": beta * w, "logit_cap": G},
            "parent_model_proved_correct": bool(legit),
            "preservation_proved": bool(pres),
            "solver_refuted_removal": bool(refuted),
            "survivor_x0": cx0, "survivor_inside_gadget": bool(in_gap)}


# ---------------------------------------------------------------------------
# Checks for P4: the steering-vs-ablation algebra, on a real trained model.
# ---------------------------------------------------------------------------
def _most_A_critical_neuron(model, n=8000, seed=7) -> int:
    """The single neuron whose removal hurts skill A most (the practitioner's
    top pick on a messy model)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    Y = np.stack([true_label_A(X), true_label_B(X)], axis=1)
    base_acc = accuracy(model, X, Y)[0]
    drops = [(base_acc - accuracy(model, X, Y, ablate=[j])[0], j)
             for j in range(model.H)]
    return max(drops)[1]


def check_P4() -> dict:
    print("\n[P4] Steering-vs-ablation algebra on a messy trained model")
    # A messy model (tidiness off), like the entangled toy model — the algebra is
    # claimed for EVERY one-hidden-layer model, so any trained one will do.
    base = train(TinyMLP(H=16, seed=4), l1=0.0, seed=5, verbose=False)
    circuit = [_most_A_critical_neuron(base)]
    abl = apply_ablation(base, circuit)

    rng = np.random.default_rng(123)
    X = rng.uniform(0.0, 1.0, size=(200_000, 2))
    pre = X @ base.W1.T + base.b1              # unedited pre-activations

    # --- P4(a): the exact leftover identity, at several doses ---------------
    # steer_s(x) = abl(x) + sum over circuit of W2[:,j] * ReLU(pre_j(x) - s)
    identity_rows, identity_ok = [], True
    for s in (2.0, 8.0, 32.0):
        steer = apply_steering(base, targeted_suppression_vector(base, circuit, s))
        leftover = np.maximum(pre[:, circuit] - s, 0.0) @ base.W2[:, circuit].T
        gap = float(np.abs(steer.forward(X) - (abl.forward(X) + leftover)).max())
        ok = gap < 1e-9
        identity_ok = identity_ok and ok
        identity_rows.append({"dose": s, "max_abs_gap": gap, "passed": bool(ok)})
        print(f"    (a) dose {s:>4}: max |steer - (ablation + leftover)| = "
              f"{gap:.2e} over 200,000 inputs -> {'OK' if ok else 'MISMATCH'}")

    # --- P4(b): above the corner-formula threshold, steer == ablation -------
    # M_j over [0,1]^2 = b_j + sum_i max(0, W1[j,i]) (max of a linear function
    # over a box sits at a corner).
    M = base.b1[circuit] + np.maximum(base.W1[circuit], 0.0).sum(axis=1)
    s_eq = float(M.max()) + 1e-9
    steer_eq = apply_steering(base, targeted_suppression_vector(base, circuit, s_eq))
    num_gap = float(np.abs(steer_eq.forward(X) - abl.forward(X)).max())
    # And the solver must give the two edits identical verdicts.
    verdicts_match = all(
        prove_forall(steer_eq, box, head, want)["proved"]
        == prove_forall(abl, box, head, want)["proved"]
        for box, head, want in [(REGION_A_HIGH, "A", "nonpositive"),
                                (REGION_B_HIGH, "B", "positive"),
                                (REGION_B_LOW, "B", "nonpositive")])
    thresh_ok = num_gap == 0.0 and verdicts_match
    print(f"    (b) threshold dose {s_eq:.4f}: max |steer - ablation| = "
          f"{num_gap} (exactly zero), solver verdicts identical: "
          f"{verdicts_match} -> {'OK' if thresh_ok else 'MISMATCH'}")

    # --- P4(d): the collateral formula for diff-of-means steering -----------
    # An input is "pattern-stable" if every steered neuron keeps its status
    # under the push: active before AND after, or inactive before AND after.
    # On such inputs P4(d) says skill B's logit shifts by EXACTLY the constant
    # sum of W2[B,j]*v_j over the neurons active in both — a value that
    # depends only on which neurons are active, not on the input itself.
    collateral_rows, collateral_ok, any_stable = [], True, False
    for strength in (0.5, 1.0, 2.0, 4.0, 8.0):
        v = diff_of_means_vector(base, strength)
        steer = apply_steering(base, v)
        touched = np.abs(v) > 1e-12
        active_before = pre > 1e-9
        active_after = (pre + v) > 1e-9
        inactive_both = (pre <= -1e-9) & ((pre + v) <= -1e-9)
        stable = (((active_before & active_after) | inactive_both)
                  | ~touched).all(axis=1)
        n_stable = int(stable.sum())
        row = {"dose": strength, "stable_points": n_stable}
        if n_stable > 0:
            any_stable = True
            # predicted shift per input: sum W2[B,j]*v_j over neurons active
            # in both (v_j = 0 elsewhere contributes nothing anyway)
            act = (active_before & active_after)[stable]
            predicted = (act * v) @ base.W2[1]
            shift = (steer.forward(X[stable]) - base.forward(X[stable]))[:, 1]
            gap = float(np.abs(shift - predicted).max())
            n_patterns = len({tuple(r) for r in act[:, touched]})
            ok = gap < 1e-9
            collateral_ok = collateral_ok and ok
            row.update({"max_abs_gap": gap, "patterns_seen": n_patterns,
                        "passed": bool(ok)})
            print(f"    (d) dose {strength:>4}: {n_stable} pattern-stable "
                  f"inputs across {n_patterns} activation patterns; B's "
                  f"logit shift matches the formula to {gap:.2e} "
                  f"-> {'OK' if ok else 'MISMATCH'}")
        else:
            row.update({"max_abs_gap": None, "patterns_seen": 0,
                        "passed": None})
            print(f"    (d) dose {strength:>4}: no pattern-stable inputs "
                  f"(every input flips some neuron) — nothing to check")
        collateral_rows.append(row)
    collateral_ok = collateral_ok and any_stable

    passed = identity_ok and thresh_ok and collateral_ok
    return {"name": "P4 steering algebra", "passed": bool(passed),
            "circuit": circuit,
            "a_leftover_identity": identity_rows,
            "b_threshold": {"dose": s_eq, "max_abs_gap": num_gap,
                            "solver_verdicts_match": verdicts_match,
                            "passed": bool(thresh_ok)},
            "d_collateral": collateral_rows}


# ---------------------------------------------------------------------------
# Report writing.
# ---------------------------------------------------------------------------
def _write_report(checks: list[dict], seconds: float):
    md = os.path.join(RESULTS, "theory_checks_report.md")
    js = os.path.join(RESULTS, "theory_checks_report.json")
    verdict = "ALL CHECKS PASSED" if all(c["passed"] for c in checks) \
        else "SOME CHECKS FAILED"

    lines = [
        "# Theory checks: machine-checked witnesses for P2-P4",
        "",
        "Output of `run_theory_checks.py`. The propositions "
        "("
        "the paper's framework section) are proved on paper; this "
        "report shows their checkable content executed against the actual "
        "code, so the theory and the implementation cannot silently drift "
        "apart. (P1 is true by the meaning of 'for all' and has no separate "
        "executable content.)",
        "",
        f"**Verdict: {verdict}** (total time {seconds:.1f}s).",
        "",
    ]
    for c in checks:
        mark = "PASSED" if c["passed"] else "FAILED"
        lines.append(f"## {c['name']} — {mark}")
        lines.append("")
        if c["name"].startswith("P2"):
            lines += [
                f"The certified radius of skill B's preservation (tidy model, "
                f"circuit ablated) came back as **{c['radius']:.6f}** using "
                f"**{c['queries']} solver calls**, within the promised budget "
                f"of {c['query_budget']}. A fresh proof at the radius "
                f"succeeded and a fresh attempt at {c['first_failure']:.6f} "
                f"was refuted — the returned number really is the tipping "
                f"point, pinned to the promised precision (gap under 0.001).",
                "",
            ]
        elif c["name"].startswith("P3"):
            lines += [
                "For each test grid below, the proof's recipe built an edit "
                "whose surviving pocket sits between that grid's lines. Every "
                "grid approved the edit (zero violations at every point, "
                "with skill B's preservation *proved*, not just tested), and "
                "the solver refuted the removal with a survivor exactly "
                "where the construction predicted.",
                "",
                "| grid | test points | fooled? | survivor found at x0 | "
                "predicted sliver | pocket (fraction of region) |",
                "|---|---|---|---|---|---|",
            ]
            for g in c["grids"]:
                lines.append(
                    f"| {g['grid']} | {g['grid_points']} | "
                    f"{'yes' if g['grid_fooled'] else 'NO'} | "
                    f"{g['survivor_x0']:.6f} | "
                    f"{g['sliver_center']:.6f} ± {g['sliver_width'] / 2:.6f} | "
                    f"{g['pocket_fraction_of_region']:.2e} |")
            lines.append("")
        elif c["name"].startswith("Corollary 1"):
            gd = c["gadget"]
            lines += [
                "A deterministic ADAPTIVE tester (coarse grid, then "
                f"{c['queries']} total queries zooming wherever head A "
                "looked most alive) approved a genuine edit — rightly. The "
                "proof's gadget was then placed in the widest gap between "
                f"the x0 values it chose to query (center {gd['center']:.6f}, "
                f"half-width {gd['halfwidth']:.2e}, bump height "
                f"{gd['bump_height']:.1f} vs a rigorous logit cap of "
                f"{gd['logit_cap']:.1f}). Re-run on the gadgeted model, the "
                "tester made the IDENTICAL sequence of queries "
                f"({c['transcripts_identical']}) and approved again — while "
                "the parent model's correctness and skill B's preservation "
                "still PROVE, and the solver refutes the removal with a "
                f"survivor at x0 = {c['survivor_x0']:.6f}, inside the "
                "gadget. Adaptivity did not help: a deterministic black-box "
                "tester's queries can be predicted by replaying it, exactly "
                "as Corollary 1 argues. (Randomized testers are outside "
                "this witness, as the proposition's scope remark states.)",
                "",
            ]
        elif c["name"].startswith("P4"):
            a = c["a_leftover_identity"]
            b = c["b_threshold"]
            lines += [
                f"Subject: a messy trained model (tidiness off), circuit = "
                f"neuron {c['circuit']}. Three sub-checks:",
                "",
                "**(a) The leftover identity.** steering(x) = ablation(x) + "
                "leftover(x), where the leftover is the formula from P4(a), "
                "held at 200,000 random inputs:",
                "",
                "| dose | max gap between the two sides |",
                "|---|---|",
            ]
            for r in a:
                lines.append(f"| {r['dose']} | {r['max_abs_gap']:.2e} |")
            lines += [
                "",
                f"**(b) The equivalence threshold.** At the corner-formula "
                f"dose {b['dose']:.4f}, steering and ablation agreed at every "
                f"sampled input with gap exactly {b['max_abs_gap']}, and the "
                f"solver returned identical verdicts on all three "
                f"certificates: {b['solver_verdicts_match']}.",
                "",
                "**(d) The collateral formula.** For the realistic "
                "diff-of-means vector: on every input where the push flips "
                "no neuron's on/off status, skill B's logit must shift by "
                "exactly the sum of W2[B,j]*v_j over the active neurons — a "
                "constant per activation pattern:",
                "",
                "| dose | pattern-stable inputs | activation patterns seen "
                "| max gap vs formula |",
                "|---|---|---|---|",
            ]
            for r in c["d_collateral"]:
                gap = "—" if r["max_abs_gap"] is None \
                    else f"{r['max_abs_gap']:.2e}"
                lines.append(f"| {r['dose']} | {r['stable_points']} | "
                             f"{r['patterns_seen']} | {gap} |")
            lines.append("")

    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"verdict": verdict, "seconds": seconds, "checks": checks},
                  f, indent=2)
    print(f"\nReports written: {md} and .json")


def main():
    t0 = time.time()
    print("=" * 70)
    print("THEORY CHECKS — executing the checkable content of P2-P4")
    print("=" * 70)
    checks = [check_P2(), check_P3(), check_corollary1(), check_P4()]
    seconds = time.time() - t0
    verdict = "ALL CHECKS PASSED" if all(c["passed"] for c in checks) \
        else "SOME CHECKS FAILED"
    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}  ({seconds:.1f}s)")
    print("=" * 70)
    _write_report(checks, seconds)
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
