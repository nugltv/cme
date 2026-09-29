"""
run_transformer_size_survey.py — is a distributed (attention-load-bearing)
       circuit an ARCHITECTURE property, or a one-off? (supplementary)
=============================================================================

Run it with:   python run_transformer_size_survey.py    (~40-60 min; NO solver)

WHAT THIS IS
------------
`run_transformer_survey.py` sweeps SEEDS at the certified configuration and one
sequence-length step up/down, and finds cross-component circuits are the
exception at that fixed shape (1 of 28). This script asks whether that is a fact
about this ONE architecture, or whether the *frequency* of distributed circuits
moves with the architecture — width, number of heads, MLP size. It sweeps the
ARCHITECTURE ladder (not just seeds) and reports the cross-component frequency
per configuration.

Everything is the practitioner's test gate only — the SAME numeric test
`run_transformer.py` uses (all sequences enumerated + sampled embedding noise),
NO certificates. Each model is the exact certified artifact (`train_gate_subject`:
train -> harden -> fine-tune -> quantize to 2^-12). The certified experiment is
`run_transformer.py`; this is population context for it.

Report: results/transformer_size_survey_report.{md,json}.
"""

from __future__ import annotations
import json
import os
import time

import numpy as np

import transformer_model as _tm
from transformer_model import train_gate_subject, sample_batch, accuracy
import run_transformer as rt
from run_transformer_survey import classify

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# Fix sequence length at the certified L=6; vary the architecture. Each config
# keeps n_heads * d_head == d_model (the residual width the heads write back).
# The ladder moves one axis at a time off the certified shape, plus the two
# extreme head counts, so each column isolates one architectural lever.
L = 6
SEEDS = range(10)
EPS0 = 0.005          # the certified numeric working-noise gate (rt.EPS0 default)
LADDER = [
    ("certified         d8/h2/mlp8",  dict(d_model=8,  n_heads=2, d_head=4, d_mlp=8)),
    ("wider model     d12/h2/mlp8", dict(d_model=12, n_heads=2, d_head=6, d_mlp=8)),
    ("narrower model  d6/h2/mlp8",  dict(d_model=6,  n_heads=2, d_head=3, d_mlp=8)),
    ("one head        d8/h1/mlp8",  dict(d_model=8,  n_heads=1, d_head=8, d_mlp=8)),
    ("more heads      d8/h4/mlp8",  dict(d_model=8,  n_heads=4, d_head=2, d_mlp=8)),
    ("wider mlp       d8/h2/mlp16", dict(d_model=8,  n_heads=2, d_head=4, d_mlp=16)),
    ("narrow mlp      d8/h2/mlp4",  dict(d_model=8,  n_heads=2, d_head=4, d_mlp=4)),
]


def main():
    t0 = time.time()
    print("=" * 70)
    print("SIZE SURVEY — does distributed-circuit frequency move with "
          "the architecture?")
    print("=" * 70)
    rt.EPS0 = EPS0
    _tm.set_seq_len(L)
    rng = np.random.default_rng(123)

    groups = []
    for label, cfg in LADDER:
        print(f"\n=== {label}: config {cfg}, seeds {list(SEEDS)} "
              f"(noise vars {L * cfg['d_model']}) ===", flush=True)
        rows = []
        for seed in SEEDS:
            m = train_gate_subject(seed=seed, config=cfg, verbose=False)
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
        groups.append({"label": label, "config": cfg, "rows": rows})

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
    md = os.path.join(RESULTS, "transformer_size_survey_report.md")
    js = os.path.join(RESULTS, "transformer_size_survey_report.json")
    lines = [
        "# Size survey: is a distributed circuit an architecture property?",
        "",
        "Output of `run_transformer_size_survey.py` — numeric only (the same test gate `run_transformer.py` uses: all "
        "sequences enumerated + sampled embedding noise; NO certificates). This extends `run_transformer_survey.py` (a seed sweep at ONE shape) across the ARCHITECTURE ladder, to see whether the frequency of "
        "cross-component (attention-load-bearing) circuits is a property of the "
        "shape rather than a one-off. Each model is the exact certified artifact "
        "(`train_gate_subject`: train -> harden -> fine-tune -> quantize).",
        "",
        f"Sequence length fixed at L={L}; {len(list(SEEDS))} seeds per config; "
        "each config keeps n_heads x d_head = d_model.",
        "",
        "**Circuit classes.** *cross-component* = >= 1 attention head AND >= 1 "
        "MLP neuron (attention genuinely load-bearing — the certified seed's "
        "kind); *single head* / *single mlp* = exactly one atom; *multi heads* "
        "/ *multi mlp* = several atoms of one type; *none* = no test-passing "
        "circuit.",
        "",
        "| config | usable | cross-component | breakdown |",
        "|---|---|---|---|",
    ]
    summary = []
    for g in groups:
        n_usable, counts = _bucket_counts(g["rows"])
        cross = counts.get("cross-component", 0)
        summary.append((g["label"], n_usable, cross, counts))
        bucket_str = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        lines.append(f"| {g['label']} | {n_usable} | **{cross}** | "
                     f"{bucket_str} |")
    lines += ["", "## Per-config detail", ""]
    for g in groups:
        lines += [f"### {g['label']}  (config {g['config']})", "",
                  "| seed | usable? | circuit | class |", "|---|---|---|---|"]
        for r in g["rows"]:
            if not r["usable"]:
                lines.append(f"| {r['seed']} | under-trained "
                             f"(A {r['acc_A']:.3f}/B {r['acc_B']:.3f}) | — "
                             f"| — |")
                continue
            lines.append(f"| {r['seed']} | yes | {r['circuit']} | "
                         f"{r['class']} |")
        lines.append("")

    total_usable = sum(s[1] for s in summary)
    total_cross = sum(s[2] for s in summary)
    lines += [
        "## The population picture",
        "",
        f"Across the whole ladder: **{total_cross} of {total_usable}** usable "
        "seeds produce a cross-component (attention-load-bearing) circuit. Per "
        "config: " + "; ".join(f"{lbl.split()[0]} {cross}/{n}"
                               for lbl, n, cross, _ in summary) + ".",
        "",
        "The question this answers is whether the certified seed-5 distributed "
        "circuit is a lucky exception or a predictable consequence of the "
        "architecture. Compare the columns: if cross-component frequency rises "
        "with head count or residual width, the distributed structure is a "
        "shape property; if it stays rare everywhere, the supported wording is "
        "that a distributed, attention-load-bearing circuit CAN occur and be "
        "certified on this architecture (the seed-sweep finding, now shown "
        "robust across shapes). This is population context for the certified "
        "result in `transformer_report.md`.",
        "",
        f"Total time {seconds:.1f}s on a laptop CPU (training dominates; no "
        "solver calls).",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"L": L, "ladder": [c for _, c in LADDER], "groups": groups,
                   "seconds": seconds, "total_usable": total_usable,
                   "total_cross": total_cross}, f, indent=2, default=str)
    print(f"\nReports written: {md} and .json")


if __name__ == "__main__":
    main()
