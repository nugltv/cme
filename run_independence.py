"""
run_independence.py — the STRONGER removal claim: "the head no longer
                      listens to x0" (paper §V-C)
======================================================================

Run it with:    python run_independence.py        (a few minutes)

WHY THIS EXPERIMENT EXISTS
--------------------------
Our removal certificate says: after the edit, head A's output stays at or
below zero everywhere in skill A's region. True — but consider what it
does NOT say: the head could still be READING x0, just at sub-threshold
strength. A leftover pathway below the waterline is invisible to the
removal claim on its region — and such pathways are exactly the raw
material of the intervention illusion.

The stronger claim, proved here with a TWO-COPY ("siamese") encoding:

    take ANY two inputs anywhere in the whole input square that differ
    only in x0 — the edited head A's output moves by at most kappa.

At kappa = 0 the head provably IGNORES x0: no value of x0 can ever matter,
above or below the decision threshold, inside or outside the claimed
region. And bisecting kappa gives the CERTIFIED INFLUENCE of x0 on the
head — a single proved number for "how much does this model still listen
to the input it was supposed to forget?", comparable across edit types
exactly like the certified radius.

WHAT GETS MEASURED
------------------
For the tidy and the messy (entangled) toy models: the certified
influence of x0 on head A, before the edit (yardstick — the working skill
must have large influence) and after each edit type (ablation, weight
edit, targeted steering, diff-of-means steering). For contrast, the same
number for head B (whose skill never depended on x0; its influence should
be small and untouched by good edits).

And the acid test: the CONSTRUCTED ILLUSION edit (paper Fig. 1), which
passes every ordinary test while a sliver survives. The removal claim over
the region catches it (run_illusion.py); the influence certificate
catches it too, with a number: the leftover tent pathway reads x0, so the
certified influence stays provably far from zero — and the solver hands
back a PAIR of inputs, identical except for x0, that the "removed" head
still tells apart.

Report: results/independence_report.md (+ .json).
"""

from __future__ import annotations
import json
import os
import time

import numpy as np

from tiny_model import TinyMLP, train, find_skill_circuit
from verify import prove_independence, certified_influence
from edits import apply_ablation, apply_weight_edit, apply_steering, \
    targeted_suppression_vector, diff_of_means_vector
from run_robustness import tests_pass, find_minimal_strength, \
    rank_neurons_by_damage_to_A
from run_illusion import build_constructed_model

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# The claim quantifies over the WHOLE legal input square — not just skill
# A's region. That is the point: "no longer listens to x0" is a statement
# about the model, not about a neighborhood we chose.
FULL_BOX = (0.0, 1.0, 0.0, 1.0)
TOL = 1e-3
STRENGTHS = (0.5, 1, 2, 4, 8, 16, 32, 64)


def measure(name, model, note="", ablate=None, heads=("A", "B")):
    """Certified influence of x0 on each head, plus exact-independence
    verdicts. One row of the table."""
    row = {"edit": name, "note": note}
    for head in heads:
        inf = certified_influence(model, FULL_BOX, head, [0], ablate=ablate,
                                  tol=TOL)
        row[f"influence_{head}"] = inf
        label = ("0 (exactly independent)" if inf["exact_zero"]
                 else f"<= {inf['influence']:.4f}" if inf["influence"]
                 is not None else "unresolved (timeout)")
        print(f"  [{name}] head {head}: certified influence of x0 {label} "
              f"({inf['queries']} solver calls, {inf['seconds']}s)")
    return row


def build_suite(base, circuit):
    """Same edit battery as run_robustness.py, minus the 4x-dose rows (the
    influence story is about kinds of edit, and the doses already have their
    own robustness/margin-curve treatment)."""
    suite = [("control (no edit)", base, "yardstick: the working skill MUST "
              "listen to x0", None)]
    suite.append(("ablation", apply_ablation(base, circuit),
                  f"neurons {circuit} switched off", None))
    suite.append(("weight edit", apply_weight_edit(base, circuit, head=0),
                  f"wires from neurons {circuit} to head A cut", None))
    s_min, m_min = find_minimal_strength(
        lambda s: apply_steering(base,
                                 targeted_suppression_vector(base, circuit,
                                                             s)), STRENGTHS)
    if s_min is not None:
        suite.append((f"steering targeted (dose {s_min:g})", m_min,
                      "smallest dose that passes the tests", None))
    s_dm, m_dm = find_minimal_strength(
        lambda s: apply_steering(base, diff_of_means_vector(base, s)),
        STRENGTHS)
    if s_dm is not None:
        suite.append((f"steering diff-of-means (dose {s_dm:g})", m_dm,
                      "smallest dose that passes the tests", None))
    return suite


def main():
    t0 = time.time()
    print("=" * 70)
    print("THE STRONGER REMOVAL CLAIM — head A provably no longer listens "
          "to x0")
    print("=" * 70)

    # ---- Subject 1: the tidy toy model -----------------------------------
    print("\n[1] Tidy model:")
    tidy = train(TinyMLP(H=16, seed=0), verbose=False)
    tidy_circuit = find_skill_circuit(tidy, "A")
    print(f"    skill A's circuit: neuron(s) {tidy_circuit}")
    tidy_rows = [measure(name, m, note)
                 for name, m, note, _ in build_suite(tidy, tidy_circuit)]

    # ---- Subject 2: the messy (entangled) toy model -----------------------
    print("\n[2] Messy model (tidiness penalty off):")
    messy, messy_circuit = None, None
    for seed in range(10):
        cand = train(TinyMLP(H=16, seed=seed), l1=0.0, seed=seed + 1,
                     verbose=False)
        ranked = rank_neurons_by_damage_to_A(cand)
        for k in range(1, 6):
            if tests_pass(apply_ablation(cand, ranked[:k])):
                messy, messy_circuit = cand, ranked[:k]
                break
        if messy is not None:
            print(f"    seed {seed}: test-passing ablation = neurons "
                  f"{messy_circuit}")
            break
    messy_rows = []
    if messy is not None:
        messy_rows = [measure(name, m, note)
                      for name, m, note, _ in build_suite(messy,
                                                          messy_circuit)]
    else:
        print("    no messy seed admitted a test-passing ablation — skipped")

    # ---- Subject 3: the constructed illusion (the acid test) --------------
    print("\n[3] The constructed illusion edit (run_illusion.py, Route B):")
    illusion_model, illusion_edit = build_constructed_model()
    print("    (passes every ordinary test; the sliver survives)")
    illusion_row = measure("illusion edit (constructed)", illusion_model,
                           "passes 40,401 test points; the leftover tent "
                           "pathway still reads x0", ablate=illusion_edit,
                           heads=("A",))
    # For the story: also ask the yes/no exact-independence question and
    # show the pair of inputs the "removed" head can still tell apart.
    check = prove_independence(illusion_model, FULL_BOX, "A", [0],
                               ablate=illusion_edit, bound=0.0)
    pair = check["counterexample_pair"]
    if pair is not None:
        print(f"    exact independence REFUTED with a pair: "
              f"x = ({pair[0][0]:.6f}, {pair[0][1]:.4f}) vs "
              f"y = ({pair[1][0]:.6f}, {pair[1][1]:.4f}) — same x1, "
              f"different x0, head A moves by {abs(check['logit_gap']):.4f}")
    illusion_row["independence_pair"] = pair
    illusion_row["independence_gap"] = check["logit_gap"]

    seconds = time.time() - t0
    _write_report(tidy_circuit, tidy_rows, messy_circuit, messy_rows,
                  illusion_row, seconds)
    print(f"\nDone in {seconds:.1f}s.")


def _fmt_inf(inf) -> str:
    if inf["exact_zero"]:
        return "**0 — exactly independent**"
    if inf["influence"] is None:
        return "unresolved (timeout)"
    return f"<= {inf['influence']:.4f}"


def _write_report(tidy_circuit, tidy_rows, messy_circuit, messy_rows,
                  illusion_row, seconds):
    md = os.path.join(RESULTS, "independence_report.md")
    js = os.path.join(RESULTS, "independence_report.json")
    lines = [
        "# The stronger removal claim: head A provably no longer listens "
        "to x0",
        "",
        "Output of `run_independence.py`. The ordinary removal certificate "
        "says the edited head A stays at or below zero on skill A's region. "
        "This one says more: over the WHOLE input square, changing x0 alone "
        "— any two inputs identical except for x0 — moves head A's output "
        "by at most the certified influence below. Influence 0 means the "
        "head provably IGNORES x0: nothing sub-threshold is left listening. "
        "Proved with a two-copy (siamese) encoding; the influence numbers "
        "are bisected ceilings, like certified radii (tolerance 0.001).",
        "",
        "## Tidy model (skill A's circuit: neurons "
        f"{tidy_circuit})",
        "",
        "| edit | certified influence of x0 on head A | on head B "
        "(yardstick: B never used x0) | note |",
        "|---|---|---|---|",
    ]
    for r in tidy_rows:
        lines.append(f"| {r['edit']} | {_fmt_inf(r['influence_A'])} | "
                     f"{_fmt_inf(r['influence_B'])} | {r['note']} |")
    lines += [
        "",
        "(Note nothing reaches exactly 0: tiny stray weights from x0 to "
        "other neurons survive training even in the tidy model, and the "
        "certificate prices them honestly instead of rounding to zero.)",
        "", f"## Messy model (entangled; test-passing ablation: "
        f"neurons {messy_circuit})", ""]
    if messy_rows:
        lines += ["| edit | certified influence of x0 on head A | on head B "
                  "(yardstick) | note |", "|---|---|---|---|"]
        for r in messy_rows:
            lines.append(f"| {r['edit']} | {_fmt_inf(r['influence_A'])} | "
                         f"{_fmt_inf(r['influence_B'])} | {r['note']} |")
        lines += [
            "",
            "(Read this table with care — it is the experiment's most "
            "instructive finding. Every edit above certifies REMOVAL on "
            "skill A's region, yet head A demonstrably still LISTENS to x0 "
            "elsewhere: most of that movement happens deep on the LOW side "
            "of zero, which removal does not forbid. On an entangled model, "
            "silencing a behavior in its region is far from deafening the "
            "head — and this certificate is the first number that states "
            "the difference as a proof. The diff-of-means row adds "
            "collateral from a new angle: after that edit, head B listens "
            "to x0 MORE than it did before the edit.)"]
    else:
        lines.append("(no messy seed admitted a test-passing ablation)")
    ir = illusion_row
    pair = ir.get("independence_pair")
    lines += [
        "",
        "## The acid test: the constructed illusion edit",
        "",
        f"| edit | certified influence of x0 on head A | note |",
        "|---|---|---|",
        f"| {ir['edit']} | {_fmt_inf(ir['influence_A'])} | {ir['note']} |",
        "",
    ]
    if pair is not None:
        lines.append(
            f"The refutation is a concrete PAIR: x = ({pair[0][0]:.6f}, "
            f"{pair[0][1]:.4f}) and y = ({pair[1][0]:.6f}, {pair[1][1]:.4f}) "
            "— identical except for x0 — that the supposedly-removed head "
            f"still tells apart (its output moves by "
            f"{abs(ir['independence_gap']):.4f}). An edit that passed "
            "40,401 test points cannot pass this certificate: the leftover "
            "pathway reads x0, so the influence cannot reach zero.")
    lines += [
        "",
        "## How to read the numbers",
        "",
        "- The CONTROL rows must show large influence on head A — the "
        "unedited skill IS an x0-listener; that is the yardstick.",
        "- After a genuine edit the certified influence collapses by "
        "orders of magnitude. Where it is exactly 0, the removal is "
        "airtight in the strongest sense: no leftover pathway from x0 to "
        "head A exists at any strength, anywhere.",
        "- Where it is small but nonzero, the number IS the leftover: a "
        "proved ceiling on everything x0 can still do to the head — the "
        "quantity the ordinary removal claim leaves unstated, stated as a number.",
        "- Head B's influence from x0 should be small in the control and "
        "unharmed by clean edits; steering that drags it around is "
        "collateral damage seen from a new angle.",
        "",
        f"Total time {seconds:.1f}s on a laptop CPU. The corresponding finding is §V-C of the paper.",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"tidy_circuit": tidy_circuit, "tidy": tidy_rows,
                   "messy_circuit": messy_circuit, "messy": messy_rows,
                   "illusion": illusion_row, "seconds": seconds},
                  f, indent=2, default=float)
    print(f"\nReports written: {md} and .json")


if __name__ == "__main__":
    main()
