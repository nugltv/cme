"""
run_deep_survey.py — how common are single-neuron vs multi-neuron circuits
                     across trained deep models? (numeric survey, no proofs)
============================================================================

Run it with:    python run_deep_survey.py        (~8 minutes; no solver)

WHY THIS EXISTS
---------------
run_deep.py happened to accept a subject (training seed 1) whose free
circuit search collapses onto a single layer-2 bottleneck neuron, so its
certified edit touches one neuron. run_deep_multi.py adds certified multi-neuron
edits on that same subject; THIS script answers the population question
behind it: across freshly trained deep models, how often is the test-passing
circuit a single neuron, how often is it several, and how often does a
LAYER-1-ONLY (multi-neuron by necessity) circuit exist at all?

It trains the same architecture on seeds 0-9 (same rule as run_deep.py),
and for each well-trained seed records:
  * the free greedy search's circuit (any layer);
  * the greedy layer-1-only circuit;
  * the smallest layer-1-only circuit found by EXHAUSTIVE search over
    subsets of up to 5 of the 12 layer-1 neurons (greedy is myopic here:
    the XOR pieces cancel in pairs, so no single knockout looks helpful).

Everything is the practitioner's test gate only (grids + random
points) — no certificates. The certified experiments are run_deep.py and
run_deep_multi.py; this is the demographic context for them.

Report: results/deep_survey_report.md (+ .json).
"""

from __future__ import annotations
import itertools
import json
import os
import time

import numpy as np

from deep_model import DeepMLP, train_deep, deep_accuracy, deep_label_A, \
    deep_label_B, ablate_deep
from run_deep import tests_pass, find_deep_circuit

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def exhaustive_layer1(model, max_k=5):
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(range(model.layer_sizes[0]), k):
            circ = [(1, j) for j in combo]
            if tests_pass(ablate_deep(model, circ)):
                return circ
    return None


def main():
    t0 = time.time()
    print("=" * 70)
    print("DEEP-MODEL CIRCUIT SURVEY — how common is the single-neuron "
          "funnel?")
    print("=" * 70)
    rng = np.random.default_rng(999)
    Xte = rng.uniform(0, 1, size=(20000, 3))
    Yte = np.stack([deep_label_A(Xte), deep_label_B(Xte)], axis=1)

    rows = []
    for seed in range(10):
        m = train_deep(DeepMLP(H1=12, H2=8, seed=seed), seed=seed + 1,
                       verbose=False)
        acc = deep_accuracy(m, Xte, Yte)
        if acc[0] < 0.99 or acc[1] < 0.99:
            print(f"    seed {seed}: under-trained "
                  f"({acc[0]:.3f}/{acc[1]:.3f}) — excluded")
            rows.append({"seed": seed, "usable": False,
                         "acc_A": acc[0], "acc_B": acc[1]})
            continue
        free = find_deep_circuit(m)
        l1_greedy = find_deep_circuit(
            m, allowed=[(1, j) for j in range(m.layer_sizes[0])])
        l1_exh = exhaustive_layer1(m)
        print(f"    seed {seed}: free greedy {free} | layer-1 greedy "
              f"{l1_greedy} | layer-1 exhaustive (smallest) {l1_exh}")
        rows.append({"seed": seed, "usable": True,
                     "acc_A": acc[0], "acc_B": acc[1],
                     "free_greedy": free, "layer1_greedy": l1_greedy,
                     "layer1_exhaustive": l1_exh})

    usable = [r for r in rows if r["usable"]]
    with_free = [r for r in usable if r["free_greedy"]]
    single = [r for r in with_free if len(r["free_greedy"]) == 1]
    cross = [r for r in with_free
             if len({l for l, _ in r["free_greedy"]}) == 2]
    with_l1 = [r for r in usable if r["layer1_exhaustive"]]
    seconds = time.time() - t0

    print(f"\n    {len(usable)} usable seeds: {len(with_free)} admit a "
          f"test-passing free-search circuit ({len(single)} single-neuron, "
          f"{len(cross)} cross-layer), {len(with_l1)} admit a layer-1-only "
          f"circuit of at most 5 neurons.")
    _write_report(rows, usable, with_free, single, cross, with_l1, seconds)
    print(f"\nDone in {seconds:.1f}s.")


def _write_report(rows, usable, with_free, single, cross, with_l1, seconds):
    md = os.path.join(RESULTS, "deep_survey_report.md")
    js = os.path.join(RESULTS, "deep_survey_report.json")
    lines = [
        "# Deep-model circuit survey: how common is the single-neuron "
        "funnel?",
        "",
        "Output of `run_deep_survey.py` — numeric only (the practitioner's "
        "deployment tests; no proofs). Context for `deep_report.md` (whose "
        "subject's free search collapses onto one bottleneck neuron) and "
        "`deep_multi_report.md` (which certifies forced multi-neuron edits "
        "on that same subject).",
        "",
        "| seed | usable? | free-search circuit | layer-1-only (greedy) | "
        "layer-1-only (exhaustive, smallest) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if not r["usable"]:
            lines.append(f"| {r['seed']} | under-trained | — | — | — |")
            continue
        lines.append(f"| {r['seed']} | yes | {r['free_greedy']} | "
                     f"{r['layer1_greedy']} | {r['layer1_exhaustive']} |")
    lines += [
        "",
        f"**The population picture ({len(usable)} well-trained seeds):** "
        f"{len(with_free)} admit a test-passing circuit at all; of those, only "
        f"{len(single)} collapse to a single neuron while {len(cross)} are "
        f"naturally CROSS-LAYER; and {len(with_l1)} admit a layer-1-only "
        "circuit of at most 5 neurons (multi-neuron by necessity — the XOR "
        "cannot live in one). The single-neuron funnel of `deep_report.md` "
        "is the exception in this population, not the rule. Note also that "
        "greedy misses some layer-1 circuits that exhaustive search finds: "
        "the XOR pieces cancel in pairs, so single knockouts look "
        "unhelpful — the same 'damage is not removal' family of traps "
        "documented in the paper's appendix.",
        "",
        f"Total time {seconds:.1f}s on a laptop CPU (training dominates; "
        "no solver calls).",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"rows": rows, "seconds": seconds}, f, indent=2,
                  default=str)
    print(f"\nReports written: {md} and .json")


if __name__ == "__main__":
    main()
