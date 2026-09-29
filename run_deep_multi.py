"""
run_deep_multi.py — the FORCED multi-neuron edit (paper Table II, "Deep MLP")
=======================================================================

Run it with:    python run_deep_multi.py        (tens of minutes)

WHY THIS EXPERIMENT EXISTS
--------------------------
The deep-model experiment (run_deep.py) was built to answer "does the story
survive depth and a distributed skill?" — and it did, with one catch: the
greedy circuit search, left free, found a single layer-2 bottleneck neuron,
so the certified edit touched ONE neuron. That does not demonstrate a
certified edit that is genuinely spread over several neurons.

This experiment forces the issue, on the SAME trained subject:

  * regime "unrestricted"  — the search may pick any neuron (the run_deep.py
    baseline, re-proved here so the table is self-contained);
  * regime "layer 1 only"  — the bottleneck layer is off-limits; the search
    must dismantle the XOR machinery itself, which no single layer-1 neuron
    can hold, so a test-passing circuit here is multi-neuron by necessity;
  * regime "detour"        — every neuron is allowed EXCEPT the ones the
    unrestricted search chose; if the model routes skill A through more
    than one pathway, this finds the second-best one.

For every test-passing edit in every regime we prove the usual full set:
removal over BOTH boxes of skill A's region (union rule, P1), preservation
of skill B, and certified radii.

Reports: results/deep_multi_report.md (+ .json).
"""

from __future__ import annotations
import itertools
import json
import os
import time

import numpy as np

from deep_model import DeepMLP, train_deep, deep_accuracy, deep_label_A, \
    deep_label_B, ablate_deep, weight_edit_deep, steer_deep, \
    targeted_vector_deep
# The regions, the test gate, the circuit search, the proof and
# radii helpers are shared with run_deep.py — same rules, same numbers.
from run_deep import A_HIGH_BOXES, A_LOW_BOXES, B_HIGH_BOX, B_LOW_BOX, \
    tests_pass, find_deep_circuit, prove_all, radii, _fmt_radius, \
    find_min_dose, TIMEOUT_MS
from verify import prove_forall

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def find_circuit_exhaustive(model, allowed, max_k=4):
    """The greedy search is MYOPIC on XOR machinery: the XOR's pieces cancel
    in PAIRS, so knocking out one piece alone can look useless — or push
    head A the wrong way — while knocking out the right pair kills the
    skill. When greedy comes back empty we therefore search ALL subsets of the allowed neurons, smallest
    first, and return the first that passes the deployment tests."""
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(allowed, k):
            if tests_pass(ablate_deep(model, list(combo))):
                return list(combo)
    return None


def steer_targeted(model, by_layer, dose):
    """Push the circuit's neurons down by `dose`, layer by layer."""
    m = model
    for layer in (1, 2):
        if by_layer[layer]:
            m = steer_deep(m, layer,
                           targeted_vector_deep(m, layer, by_layer[layer],
                                                dose))
    return m


def split_by_layer(circuit):
    return {1: [j for (l, j) in circuit if l == 1],
            2: [j for (l, j) in circuit if l == 2]}


def main():
    t0 = time.time()
    print("=" * 70)
    print("THE FORCED MULTI-NEURON EDIT — same deep subject, constrained "
          "circuit search")
    print("=" * 70)

    # ---- 1. The subject: the same scan rule as run_deep.py, so we land on
    # the same trained model, then require the forced regimes on it. --------
    print("\n[1] Training and scanning seeds (same rule as run_deep.py)...")
    rng = np.random.default_rng(999)
    Xte = rng.uniform(0, 1, size=(20000, 3))
    Yte = np.stack([deep_label_A(Xte), deep_label_B(Xte)], axis=1)
    model, train_seed, circuit_u, skipped = None, None, None, []
    for seed in range(10):
        cand = train_deep(DeepMLP(H1=12, H2=8, seed=seed), seed=seed + 1,
                          verbose=False)
        acc = deep_accuracy(cand, Xte, Yte)
        print(f"    seed {seed}: test-acc A={acc[0]:.4f} B={acc[1]:.4f}",
              end="")
        if acc[0] < 0.99 or acc[1] < 0.99:
            print("  (under-trained — skipped)")
            skipped.append({"seed": seed, "why": "under-trained"})
            continue
        found = find_deep_circuit(cand)
        if found is None:
            print("  (no test-passing ablation — skipped)")
            skipped.append({"seed": seed, "why": "no test-passing ablation"})
            continue
        print("  ACCEPTED")
        model, train_seed, circuit_u = cand, seed, found
        break
    if model is None:
        raise SystemExit("no seed produced a subject — widen the scan")

    # ---- 2. The three circuit-search regimes on this one subject ----------
    print("\n[2] Circuit search per regime (greedy, then exhaustive small "
          "subsets if greedy is defeated by the XOR's pairwise "
          "cancellation)...")
    layer1_neurons = [(1, j) for j in range(model.layer_sizes[0])]
    non_baseline = [(l, j) for l in (1, 2)
                    for j in range(model.layer_sizes[l - 1])
                    if (l, j) not in set(circuit_u)]

    def find_forced(allowed, max_k):
        circ = find_deep_circuit(model, allowed=allowed)
        if circ is not None:
            return circ, "greedy"
        circ = find_circuit_exhaustive(model, allowed, max_k=max_k)
        if circ is not None:
            return circ, ("exhaustive small-subset search (greedy missed "
                          "it — the XOR pieces cancel in pairs, so no "
                          "single knockout looks helpful)")
        return None, "neither greedy nor exhaustive search"

    circ_l1, how_l1 = find_forced(layer1_neurons, max_k=5)
    circ_dt, how_dt = find_forced(non_baseline, max_k=4)
    regimes = [
        ("unrestricted", circuit_u,
         "the free search's pick (run_deep.py's baseline)", "greedy"),
        ("layer 1 only", circ_l1,
         "bottleneck layer off-limits; must dismantle the XOR machinery",
         how_l1),
        ("detour", circ_dt,
         "the unrestricted circuit's neurons banned; second-best pathway",
         how_dt),
    ]
    for name, circ, why, how in regimes:
        if circ is None:
            print(f"    {name}: NO test-passing circuit (negative result)")
        else:
            bl = split_by_layer(circ)
            print(f"    {name}: {circ} ({len(bl[1])} in layer 1, "
                  f"{len(bl[2])} in layer 2) — found by {how}")

    # ---- 3. Edit suite per regime ------------------------------------------
    print("\n[3] Edits + certificates + radii per regime...")
    rows = []
    for regime, circ, why, how in regimes:
        if circ is None:
            rows.append({"regime": regime, "circuit": None, "edit": "—",
                         "note": "no test-passing circuit found: " + why,
                         "skipped": True, "tests_pass": False,
                         "removal_proved": False,
                         "preservation_proved": False,
                         "removal_radius": None, "preservation_radius": None})
            continue
        bl = split_by_layer(circ)
        suite = [("ablation", ablate_deep(model, circ),
                  f"{len(circ)} neuron(s) switched off")]
        if not bl[1]:
            suite.append(("weight edit", weight_edit_deep(model, circ),
                          "wires from the layer-2 circuit neurons to head A "
                          "cut"))
        s_t = find_min_dose(lambda s: steer_targeted(model, bl, s))
        if s_t is not None:
            suite.append((f"steering targeted (dose {s_t:g})",
                          steer_targeted(model, bl, s_t),
                          "circuit neurons pushed down; smallest dose that "
                          "passes the tests"))
        else:
            suite.append(("steering targeted", None,
                          "NO dose up to 64 passes the tests"))

        for edit_name, m, note in suite:
            if m is None:
                rows.append({"regime": regime, "circuit": circ,
                             "edit": edit_name, "note": note,
                             "skipped": True, "tests_pass": False,
                             "removal_proved": False,
                             "preservation_proved": False,
                             "removal_radius": None,
                             "preservation_radius": None})
                print(f"    [{regime}] {edit_name}: {note}")
                continue
            proofs = prove_all(m)
            passes = tests_pass(m)
            rr = radii(m)
            removal_ok = all(proofs[k]["proved"] for k in proofs
                             if k.startswith("removal"))
            pres_ok = proofs["B high"]["proved"] and proofs["B low"]["proved"]
            print(f"    [{regime}] {edit_name}: tests "
                  f"{'pass' if passes else 'FAIL'}, removal "
                  f"{'PROVED' if removal_ok else 'refuted'} "
                  f"(radius {_fmt_radius(rr['removal'])}), preservation "
                  f"{'PROVED' if pres_ok else 'refuted'} "
                  f"(radius {_fmt_radius(rr['preservation'])})")
            rows.append({"regime": regime, "circuit": circ,
                         "edit": edit_name, "note": note,
                         "tests_pass": passes,
                         "removal_proved": removal_ok,
                         "preservation_proved": pres_ok,
                         "removal_radius": rr["removal"],
                         "preservation_radius": rr["preservation"]})

    # ---- 4. Yardstick: the unedited model's preservation radius -----------
    print("\n[4] Control (unedited) radii for the yardstick...")
    ctrl_rr = radii(model)
    print(f"    control: removal {_fmt_radius(ctrl_rr['removal'])} "
          f"(must be refuted), preservation "
          f"{_fmt_radius(ctrl_rr['preservation'])}")

    seconds = time.time() - t0
    _write_report(train_seed, skipped, regimes, rows, ctrl_rr, seconds)
    print(f"\nDone in {seconds:.1f}s.")


def _write_report(train_seed, skipped, regimes, rows, ctrl_rr, seconds):
    md = os.path.join(RESULTS, "deep_multi_report.md")
    js = os.path.join(RESULTS, "deep_multi_report.json")
    regime_desc = {name: (circ, why) for name, circ, why, _ in regimes}
    lines = [
        "# The forced multi-neuron edit: constrained circuit search on the "
        "deep model",
        "",
        "Output of `run_deep_multi.py`. Same subject, regions, margins, "
        "tests and proof machinery as `deep_report.md`; the only change is "
        "WHICH neurons the circuit search may use. The free search collapses onto a single bottleneck neuron, so the free-search certified edit touches one neuron. "
        "Here the search is constrained so that cannot happen, and every "
        "resulting edit is certified the usual way.",
        "",
        f"Subject: training seed {train_seed} "
        f"({len(skipped)} earlier seed(s) skipped, same scan rule as "
        "run_deep.py).",
        "",
        "## The circuits the three regimes found",
        "",
    ]
    for name, circ, why, how in regimes:
        if circ is None:
            lines.append(f"- **{name}** ({why}): no test-passing circuit (a negative result).")
        else:
            bl = split_by_layer(circ)
            lines.append(f"- **{name}** ({why}): {circ} — "
                         f"{len(bl[1])} neuron(s) in layer 1, "
                         f"{len(bl[2])} in layer 2; found by {how}.")
    l1_circ = regime_desc.get("layer 1 only", (None, None))[0]
    dt_circ = regime_desc.get("detour", (None, None))[0]
    if l1_circ is not None and l1_circ == dt_circ:
        lines += [
            "",
            "(The detour regime converged on the SAME layer-1 circuit: with "
            "the bottleneck banned, the only other certifiable handle on "
            "skill A is the XOR machinery itself. This subject offers "
            "exactly two clean edits — the funnel, or the pieces feeding "
            "it.)"]
    lines += [
        "",
        "## Certificates per regime and edit",
        "",
        "| regime | edit | passes tests? | removal proved? | removal radius "
        "| preservation proved? | preservation radius | note |",
        "|---|---|---|---|---|---|---|---|",
        f"| — | control (no edit) | — | must fail | "
        f"{_fmt_radius(ctrl_rr['removal'])} | (unedited) | "
        f"{_fmt_radius(ctrl_rr['preservation'])} | yardstick: B's natural "
        "margin |",
    ]
    for r in rows:
        if r.get("skipped"):
            lines.append(f"| {r['regime']} | {r['edit']} | — | — | — | — | "
                         f"— | {r['note']} |")
            continue
        lines.append(
            f"| {r['regime']} | {r['edit']} | "
            f"{'yes' if r['tests_pass'] else 'NO'} | "
            f"{'yes' if r['removal_proved'] else 'REFUTED'} | "
            f"{_fmt_radius(r['removal_radius'])} | "
            f"{'yes' if r['preservation_proved'] else 'REFUTED'} | "
            f"{_fmt_radius(r['preservation_radius'])} | {r['note']} |")
    lines += [
        "",
        f"Total time {seconds:.1f}s on a laptop CPU.",
        "",
        "## Why this matters",
        "",
        "If the constrained regimes certify, certified edits are not limited to one neuron: the same "
        "certificates hold for an edit spread over several neurons (and, in "
        "the detour regime, over whichever pathway remains). If a regime "
        "finds nothing test-passing, that is reported as-is — it would mean "
        "this subject funnels skill A so tightly that only the bottleneck "
        "admits a clean edit, which is itself a checkable claim about the "
        "model, not a weakness of the method.",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"train_seed": train_seed, "skipped_seeds": skipped,
                   "regimes": {name: {"circuit": None if c is None else
                                      [list(x) for x in c], "why": why,
                                      "found_by": how}
                               for name, c, why, how in regimes},
                   "rows": rows, "control_radii": ctrl_rr,
                   "seconds": seconds}, f, indent=2, default=float)
    print(f"\nReports written: {md} and .json")


if __name__ == "__main__":
    main()
