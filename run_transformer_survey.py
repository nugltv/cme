"""
run_transformer_survey.py — how common is a DISTRIBUTED (attention-load-bearing)
                            circuit across trained threshold-gate transformers?
==============================================================================

Run it with:   python run_transformer_survey.py        (~30 min; NO solver)

WHAT THIS IS (supplementary; context for paper Fig. 3)
------------------------------------------------------
The certified transformer (run_transformer.py, seed 5) has a genuinely
DISTRIBUTED circuit — one attention head PLUS three MLP neurons. Of the seeds in
results/rung2_circuit_search.log, it is 1 of 6 with a cross-component circuit;
the others give a single head, MLP-only, or no test-passing circuit at all. This
script measures how typical that structure is, the same way run_deep_survey.py
does for the deep model: across many freshly trained seeds, at the certified
configuration AND one sequence-length step up and down, how often is the
test-passing circuit

    * CROSS-COMPONENT  (>=1 head AND >=1 MLP neuron) — the certified seed's kind,
      the one where attention is genuinely load-bearing;
    * a SINGLE ATOM    (exactly one head, or exactly one MLP neuron);
    * MULTI, ONE FAMILY (several heads, or several MLP neurons, but not both) —
      distributed, but within one component type;
    * NONE             (no test-passing circuit exists).

Everything here is the practitioner's test gate only — the SAME numeric
test run_transformer.py uses (all sequences enumerated + sampled embedding
noise), NO certificates. The certified experiment is run_transformer.py; this
is population context for it. The model built per seed is the exact certified
artifact (train_gate_subject: train -> harden -> fine-tune -> quantize to
2^-12). The paper's claim is worded to what this population supports: a
distributed, attention-load-bearing circuit is POSSIBLE and certifiable on this
architecture.

Report: results/transformer_survey_report.md (+ .json).
"""

from __future__ import annotations
import json
import os
import time

import numpy as np

import transformer_model as _tm
from transformer_model import train_gate_subject, sample_batch, accuracy
import run_transformer as rt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# The certified architecture (run_transformer.SUBJECT_CONFIG), swept over seeds at
# three sequence lengths: the certified L=6, one step down (L=5), one step up
# (L=7). Circuit STRUCTURE is a numeric property, so the solver-tractability of
# each length is irrelevant here — we are asking what the trained model's
# circuit looks like, not proving anything about it.
CONFIG = dict(d_model=8, n_heads=2, d_head=4, d_mlp=8)
SWEEP = [
    ("L=5 (step down)", 5, range(8)),
    ("L=6 (certified)", 6, range(12)),
    ("L=7 (step up)", 7, range(8)),
]
EPS0 = 0.005          # the certified numeric working-noise gate (rt.EPS0 default)


def classify(circ) -> str:
    """Bucket a circuit by how distributed it is. `circ` is a list of atoms
    like [('head', 0), ('mlp', 1), ...] or None."""
    if not circ:
        return "none"
    heads = [a for a in circ if a[0] == "head"]
    mlps = [a for a in circ if a[0] == "mlp"]
    if heads and mlps:
        return "cross-component"                 # the certified seed's kind
    if len(circ) == 1:
        return "single head" if heads else "single mlp"
    return "multi heads" if heads else "multi mlp"


def main():
    t0 = time.time()
    print("=" * 70)
    print("CIRCUIT SURVEY — how common is a distributed, "
          "attention-load-bearing circuit?")
    print("=" * 70)
    rt.EPS0 = EPS0
    rng = np.random.default_rng(123)

    groups = []
    for label, L, seeds in SWEEP:
        _tm.set_seq_len(L)
        print(f"\n=== {label}: config {CONFIG}, seeds {list(seeds)} "
              f"(noise vars {L * CONFIG['d_model']}) ===", flush=True)
        rows = []
        for seed in seeds:
            m = train_gate_subject(seed=seed, config=CONFIG, verbose=False)
            tok, y = sample_batch(8000, rng)
            accA, accB = accuracy(m, tok, y)
            if min(accA, accB) < 0.99:
                print(f"  seed {seed}: undertrained "
                      f"(A {accA:.3f} B {accB:.3f}) — excluded", flush=True)
                rows.append({"seed": seed, "usable": False,
                             "acc_A": accA, "acc_B": accB})
                continue
            circ, _ = rt.find_circuit(m)
            cls = classify(circ)
            print(f"  seed {seed}: circuit={circ}  -> {cls}", flush=True)
            rows.append({"seed": seed, "usable": True,
                         "acc_A": accA, "acc_B": accB,
                         "circuit": circ, "class": cls})
        groups.append({"label": label, "L": L, "rows": rows})

    seconds = time.time() - t0
    _write_report(groups, seconds)
    print(f"\nDone in {seconds:.1f}s.")


def _bucket_counts(rows):
    usable = [r for r in rows if r["usable"]]
    counts = {}
    for r in usable:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    return len(usable), counts


def _write_report(groups, seconds):
    md = os.path.join(RESULTS, "transformer_survey_report.md")
    js = os.path.join(RESULTS, "transformer_survey_report.json")
    lines = [
        "# Circuit survey: how common is a distributed, attention-load-bearing circuit?",
        "",
        "Output of `run_transformer_survey.py` — numeric only (the same "
        "test gate `run_transformer.py` uses: all sequences enumerated + "
        "sampled embedding noise; NO certificates). Context for the certified transformer result (`transformer_report.md`, seed 5, whose circuit is cross-component): is that circuit typical, or the exception? Each model is the exact "
        "certified artifact: `train_gate_subject` (train → harden → "
        "fine-tune → quantize to 2^-12).",
        "",
        "**Circuit classes.** *cross-component* = ≥ 1 attention head AND "
        "≥ 1 MLP neuron (attention genuinely load-bearing — the "
        "certified seed's kind); *single head* / *single mlp* = exactly one atom; "
        "*multi heads* / *multi mlp* = several atoms of one type (distributed "
        "within a component); *none* = no test-passing circuit.",
        "",
    ]
    summary = []
    for g in groups:
        n_usable, counts = _bucket_counts(g["rows"])
        cross = counts.get("cross-component", 0)
        summary.append((g["label"], n_usable, cross, counts))
        lines += [
            f"## {g['label']}  (config {CONFIG})",
            "",
            "| seed | usable? | circuit | class |",
            "|---|---|---|---|",
        ]
        for r in g["rows"]:
            if not r["usable"]:
                lines.append(f"| {r['seed']} | under-trained "
                             f"(A {r['acc_A']:.3f}/B {r['acc_B']:.3f}) | — "
                             f"| — |")
                continue
            lines.append(f"| {r['seed']} | yes | {r['circuit']} | "
                         f"{r['class']} |")
        bucket_str = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        lines += [
            "",
            f"**{g['label']}: {n_usable} usable seeds** — {bucket_str}. "
            f"Cross-component (attention load-bearing) circuits: "
            f"**{cross} of {n_usable}**.",
            "",
        ]

    total_usable = sum(s[1] for s in summary)
    total_cross = sum(s[2] for s in summary)
    lines += [
        "## The population picture",
        "",
        f"Across all swept seeds and lengths: **{total_cross} of "
        f"{total_usable}** usable seeds produce a cross-component "
        "(attention-load-bearing) circuit of the kind the certified seed-5 "
        "result showcases. Per length: "
        + "; ".join(f"{lbl} {cross}/{n}" for lbl, n, cross, _ in summary)
        + ".",
        "",
        "Read this alongside `transformer_report.md`: the certified seed is a "
        "genuine, certified example of a distributed circuit — what this "
        "survey establishes is how *representative* that structure is, so the claim is worded to exactly what the population supports.",
        "",
        f"Total time {seconds:.1f}s on a laptop CPU (training dominates; no "
        "solver calls).",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"config": CONFIG, "groups": groups, "seconds": seconds,
                   "total_usable": total_usable, "total_cross": total_cross},
                  f, indent=2, default=str)
    print(f"\nReports written: {md} and .json")


if __name__ == "__main__":
    main()
