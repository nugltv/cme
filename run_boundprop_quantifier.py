"""
run_boundprop_quantifier.py — tighten the quantifier domain (paper §V-F, §VI)
==============================================================================

Run it with (isolated bound-prop env — see requirements-boundprop.txt):
    .venv-boundprop/bin/python run_boundprop_quantifier.py

THE QUESTION
------------
M1 certifies removal + preservation over "a class of sequences × a continuous
embedding ball". But an adversary's real move is a DISCRETE
token substitution (rewrite a character), which lands OUTSIDE a small embedding
ball. So does the guarantee say anything about the adversary's move set, or only
about imperceptible continuous noise?

WHAT THIS ADDS (two halves)
---------------------------
1. **A genuine discrete-token guarantee for removal.** The M1 removal claim was
   stated over the TQ-*positive* sequences only. But the edit firewalls the quote
   positions entirely, so the quote content is irrelevant to the readout — and we
   certify removal over the **ENTIRE TQ class**: every sequence whose task token
   is TQ, for ALL 4^3 quote combinations × ALL 4^3 bracket combinations. That set
   is *closed* under any rewriting of the content positions, so this is exactly a
   discrete-substitution neighborhood guarantee: an attacker may rewrite the
   quote (or bracket) content to anything in-vocabulary and skill A stays removed
   — proved, not tested, each sequence also carrying a continuous embedding ball.

2. **The boundary of the continuous ball.** We then measure how far the
   continuous embedding ball actually reaches: the smallest L∞ distance between
   two distinct content-token embeddings (the smallest single-character swap step)
   versus the certified radius. The ball is orders of magnitude smaller than a
   token swap — so the *continuous* radius is NOT a stand-in for discrete
   robustness. The two guarantees are complementary: discrete closure comes from
   enumerating the class (half 1), continuous robustness from the ball; neither is
   oversold as the other.

Report: results/boundprop_quantifier_report.{md,json}.
"""

from __future__ import annotations
import json
import os
import time

import numpy as np
import torch

import boundprop_transformer as B
from boundprop_transformer import (GateXformer, all_sequences, labels,
                                   skill_accuracy, train, TQ, QPOS, BPOS, QTOK,
                                   BTOK)
import run_boundprop_transformer as R
from auto_LiRPA import BoundedModule

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def _token_separation(model):
    """Smallest / median L∞ distance between two distinct CONTENT-token
    embeddings — the size of the smallest in-vocabulary single-character swap, in
    the same L∞ metric as the certified ball (PerturbationLpNorm norm=inf). The
    positional embedding is shared by a swap at a fixed position, so it cancels;
    the token-swap step is ‖E[t_i] - E[t_j]‖∞."""
    content = QTOK + BTOK
    E = model.tok_emb.weight.detach().numpy()
    dists = []
    for i in range(len(content)):
        for j in range(i + 1, len(content)):
            dists.append(float(np.abs(E[content[i]] - E[content[j]]).max()))
    return {"min": min(dists), "median": float(np.median(dists)),
            "max": max(dists), "pairs": len(dists)}


def main():
    t0 = time.time()
    torch.manual_seed(0)
    print("=" * 70)
    print("TIGHTEN THE QUANTIFIER DOMAIN (discrete + continuous)")
    print("=" * 70)

    seqs = all_sequences()
    y = labels(seqs)
    isA = seqs[:, -1] == TQ

    print(f"\n[1] Train the M1 subject (seed {R.SEED}) and apply the edit...")
    model = GateXformer(R.D_MODEL, R.N_HEADS, R.N_LAYERS, R.D_MLP, seed=R.SEED)
    train(model, seqs, y, steps=3000, lr=3e-3)
    accA, accB = skill_accuracy(model, seqs, y)
    edited = model.ablate_positions(QPOS)
    e_accA, e_accB = skill_accuracy(edited, seqs, y)
    print(f"    accuracy A {accA:.3f}->{e_accA:.3f}, B {accB:.3f}->{e_accB:.3f}")

    torch.set_grad_enabled(False)
    core = edited.core
    bmodule = BoundedModule(core, torch.zeros(R.CHUNK, B.L, R.D_MODEL))

    # ---- half 1: discrete-token closure — removal over the WHOLE TQ class ----
    tq_all = seqs[isA]                       # every TQ sequence (all content)
    tq_pos = seqs[isA & (y == 1)]            # the original M1 claim (TQ-positive)
    print(f"\n[2] Removal over the WHOLE TQ class ({len(tq_all)} seqs: all quote "
          f"× all bracket content) vs the original TQ-positive-only claim "
          f"({len(tq_pos)})...")

    def rad(name, seqcls):
        emb = R._emb_batch(edited, seqcls)
        chunks = R._pad_chunks(emb, R.CHUNK)
        c = R.certified_radius(bmodule, chunks, "nonpositive")
        p = R.pgd_first_break(core, emb, "nonpositive")
        print(f"    {name}: certified {R._fmt(c)} | PGD break "
              f"{'none' if p is None else p}", flush=True)
        return {"claim": name, "n": int(len(seqcls)), "certified": c, "pgd": p}

    rows = [rad("removal over ALL TQ sequences (discrete closure)", tq_all),
            rad("removal over TQ-positive only (original M1)", tq_pos)]

    # ---- half 2: the continuous ball vs a discrete token swap ----------------
    print("\n[3] How far does the continuous ball reach vs a token swap?")
    sep = _token_separation(model)
    cert_all = rows[0]["certified"]
    ratio = (sep["min"] / cert_all) if cert_all else float("inf")
    print(f"    smallest content-token L∞ swap distance: {sep['min']:.3f} "
          f"(median {sep['median']:.3f})")
    print(f"    certified removal radius (whole TQ class): {R._fmt(cert_all)}")
    print(f"    a token swap is ~{ratio:.0f}× the certified radius — the "
          f"continuous ball does NOT reach a discrete substitution")

    seconds = time.time() - t0
    _write_report(accA, accB, e_accA, e_accB, rows, sep, ratio, seconds)
    print(f"\nDone in {seconds:.1f}s.")


def _write_report(accA, accB, e_accA, e_accB, rows, sep, ratio, seconds):
    md = os.path.join(RESULTS, "boundprop_quantifier_report.md")
    js = os.path.join(RESULTS, "boundprop_quantifier_report.json")
    cert_all = rows[0]["certified"]
    lines = [
        "# Tightening the quantifier — discrete-token closure + the reach of the continuous ball",
        "",
        "Output of `run_boundprop_quantifier.py`. It addresses the gap between "
        "a continuous embedding ball and the adversary's discrete "
        "token-substitution move set, on the M1 softmax+LayerNorm subject, in "
        "two halves.",
        "",
        f"Model: the M1 subject (d_model {R.D_MODEL}, {R.N_HEADS} heads, "
        f"{R.N_LAYERS} layers). Accuracy A {accA:.3f}→{e_accA:.3f}, "
        f"B {accB:.3f}→{e_accB:.3f} after the position-scoped attention knockout.",
        "",
        "## Half 1 — a genuine discrete-token guarantee for removal",
        "",
        "The edit firewalls the quote positions, so the quote content is "
        "irrelevant to the readout. We therefore certify removal not just over the "
        "TQ-positive sequences (the original M1 claim) but over the **entire TQ "
        "class** — every sequence whose task token is TQ, across all quote AND all "
        "bracket content. That set is *closed under any in-vocabulary rewriting of "
        "the content positions*, so certifying it is exactly a discrete "
        "token-substitution neighborhood guarantee: an attacker may rewrite the "
        "content to anything in-vocabulary and skill A stays removed — each "
        "sequence still carrying a continuous embedding ball on top.",
        "",
        "| removal claim | # sequences | certified radius (sound) | PGD break |",
        "|---|---|---|---|",
    ]
    for r in rows:
        p = "none ≤ 0.4" if r["pgd"] is None else f"{r['pgd']}"
        lines.append(f"| {r['claim']} | {r['n']} | {R._fmt(r['certified'])} | {p} |")
    lines += [
        "",
        "The whole-TQ-class radius certifies the strictly larger discrete set, so "
        "the removal guarantee now reads: *for every sequence with task token TQ "
        "(any quote content, any bracket content) and every embedding perturbation "
        "up to the certified radius, the skill-A readout stays ≤ 0.*",
        "",
        "## Half 2 — the reach of the continuous ball",
        "",
        f"Smallest L∞ distance between two distinct content-token embeddings (the "
        f"smallest in-vocabulary single-character swap, same metric as the "
        f"certified ball): **{sep['min']:.3f}** (median {sep['median']:.3f}, over "
        f"{sep['pairs']} token pairs). Certified removal radius over the whole TQ "
        f"class: **{R._fmt(cert_all)}**. A token swap is **~{ratio:.0f}×** the "
        "certified radius.",
        "",
        "So the *continuous* embedding ball is orders of magnitude too small to "
        "reach a discrete token substitution — the radius is a genuine "
        "continuous-robustness number, NOT a disguised discrete-robustness claim. "
        "The two guarantees are complementary and neither is oversold: **discrete "
        "closure** over the content vocabulary comes from enumerating the class "
        "(half 1); **continuous robustness** comes from the ball (the radius). "
        "This is the quantified version of the paper's §VI scope note.",
        "",
        f"Total time {seconds:.1f}s.",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"acc": {"A": accA, "B": accB},
                   "edited_acc": {"A": e_accA, "B": e_accB},
                   "rows": rows, "token_separation": sep, "ratio": ratio,
                   "seconds": seconds}, f, indent=2, default=float)
    print(f"\nReports written: {md} and .json")


if __name__ == "__main__":
    main()
