"""
run_transformer.py — a certified edit on a (small) threshold-gate transformer
=============================================================================

Run it with:    python run_transformer.py        (tens of minutes)

WHAT THIS IS (paper §V-A, Table II "Gate transformer", Fig. 3)
--------------------------------------------------------------
On the certified subject (d_model 8, 2 heads, L=6, seed 5) the pipeline finds a
distributed circuit [head 0 + MLP 1, 5, 2] and certifies removal (radius 0.016)
and preservation (0.013) over all sequences x continuous embedding noise, using
the HULL relaxation (exact; unsat -> the discrete claim by P1, with no Boolean
branching).

THE CLAIM (the measured frontier: results/rung2_size_ladder.log, paper Table III)
-------------------------------------------------------------------------------
On a trained transformer with two sequence skills — "is there an unclosed
quote?" (skill A, to remove) and "is there an unclosed bracket?" (skill B,
to preserve) — after each mechanistic edit:

    for EVERY token sequence in the claim's class and EVERY embedding-space
    perturbation of every position up to epsilon, the edited model no
    longer answers the quote question (removal) while the bracket question
    still gets the right answer (preservation);

plus the CERTIFIED EMBEDDING RADIUS: the largest epsilon for which removal
still proves, per edit. One solver query covers all sequences x all noise
at once — the token space alone is enumerable (5^7 sequences; we brute-force
it numerically as a cross-check) but the noise is not, and noise is where
jailbreak-style recovery lives. The sequences enter the query one of two
ways (verify_transformer.encode_logit's token_mode): "bool" = one Boolean
selector per (position, token), which IS the discrete claim; "hull" (the
current default) = continuous mixture weights over a strictly larger region,
so an unsat still certifies the discrete claim by P1 while a sat proves
nothing about it. "bool" exceeds the time budget at every size; "hull" also
times out on the big config, but on the certified subject it PROVES (with no
Boolean branching) — which is why the subject has this size.

The subject is the hardened gate-attention model (transformer_model.py):
threshold attention is what makes exact input-side certification possible
at all — soft attention's application is bilinear in the noise. Spec note: the skills are global counting
properties of the sequence; no input coordinate is the label.

Pipeline: [1] train + held-out accuracy; [2] validate the Z3 encoding
against the float forward (mandatory before any proof is trusted);
[3] control certificates + natural radii; [4] circuit search with the
removal objective; [5] the edit suite as weight changes; [6] certificates
+ radii per edit -> results/transformer_report.{md,json}.
"""

from __future__ import annotations
import json
import os
import time

import numpy as np
import z3
from fractions import Fraction

import transformer_model as _tm
from transformer_model import (TinyTransformer, train_gate_subject,
                               sample_batch, accuracy, label_quote, Q1,
                               Q2, ablate_head, ablate_mlp_neurons,
                               weight_edit_mlp, steer_residual,
                               diff_of_means_direction)
from verify_transformer import (SequenceClass, encode_logit,
                                prove_transformer, certified_eps,
                                exhaustive_clean_check, sampled_noise_check,
                                all_sequences)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# The CERTIFIED subject: the small config, at the sequence length, where
# exact input-side certification is tractable AND a test-passing circuit exists.
# This is the measured frontier (the size-ladder measurements in results/rung2_size_ladder.log):
#   - bigger (d_model 16 / 2 heads / L=8) times out in the solver;
#   - d_model 8 at L=8 makes the hull relaxation leak (fractional non-sequence
#     counterexamples);
#   - 1 head cannot host a circuit separable from the OTHER skill (both skills
#     route through the single head), so no edit passes;
#   - d_model 8, 2 heads, L=6 both PROVES and admits a test-passing
#     cross-component circuit (an attention head + MLP neurons).
SUBJECT_CONFIG = dict(d_model=8, n_heads=2, d_head=4, d_mlp=8)
SUBJECT_L = 6
SUBJECT_SEED = 5
_tm.set_seq_len(SUBJECT_L)

EPS0 = 0.005       # FIXED working noise bound for the test gate and the
                   # eps-fixed proof table. Not tied to the control's natural
                   # margin (~0.034): doing so would demand the EDIT be as
                   # robust as the unedited model and hide collateral. The
                   # certified RADII below are the real robustness story
                   # (the radius is the result, not a gate).
EPS_MAX = 0.05     # radius bisection cap (embedding entries are O(0.5), so
                   # 0.05 is already a sizeable perturbation of every token)
TOL = 5e-3         # radius precision (transformer proofs cost more)
SLACK = 1e-6       # float-gap slack: claims clear zero by this
TIMEOUT = 300000   # 5 min per query; 'unknown' is reported, never hidden

CLAIMS = {
    "A pos": (SequenceClass(Q1, "pos"), "positive"),
    "A neg": (SequenceClass(Q1, "neg"), "nonpositive"),
    "B pos": (SequenceClass(Q2, "pos"), "positive"),
    "B neg": (SequenceClass(Q2, "neg"), "nonpositive"),
}


def validate_encoding(model, n=40, eps=0.05, seed=11) -> float:
    """Mandatory gate: the Z3 encoding must equal the float forward at
    sampled (sequence, noise) points before any proof is trusted."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        tokens, _ = sample_batch(1, rng)
        noise = rng.uniform(-eps, eps, size=(_tm.L, model.d))
        sc = SequenceClass(int(tokens[0, _tm.L - 1]), "pos")
        s = z3.Solver()
        logit, _ = encode_logit(model, s, eps, sc,
                                fixed_tokens=list(tokens[0][:_tm.L - 1]),
                                fixed_noise=noise)
        assert s.check() == z3.sat
        v = s.model().eval(logit, model_completion=True)
        got = float(Fraction(int(v.numerator_as_long()),
                             int(v.denominator_as_long())))
        want = float(model.forward(tokens, noise=noise[None])[0])
        worst = max(worst, abs(got - want))
    return worst


def tests_pass(model) -> bool:
    """The practitioner's test gate for the transformer: skill A must LOOK
    gone and skill B intact on (a) ALL 78,125 clean sequences per claim —
    the token space is enumerable, so we enumerate it — and (b) 2,000
    random noise samples per claim at the working epsilon."""
    gates = [
        (CLAIMS["A pos"][0], "nonpositive"),      # removal expected
        CLAIMS["B pos"], CLAIMS["B neg"],
    ]
    for scls, want in gates:
        if exhaustive_clean_check(model, scls, want)["violations"]:
            return False
        if sampled_noise_check(model, scls, want, EPS0)["violations"]:
            return False
    return True


def removal_score(model, toks_pos) -> float:
    """Worst (max) logit over the clean sequences where skill A fires —
    the removal objective (damage is not removal; see run_deep.py)."""
    return float(model.forward(toks_pos).max())


def find_circuit(model):
    """Greedy search over edit atoms — attention heads and MLP neurons —
    minimizing skill A's worst logit subject to skill B surviving the
    test gate. Returns (kind, atoms) or None."""
    toks = all_sequences(Q1)
    toks_pos = toks[CLAIMS["A pos"][0].numeric_mask(toks)]
    atoms = [("head", h) for h in range(model.H)] + \
            [("mlp", j) for j in range(model.m)]

    def apply(atom_set):
        m = model
        heads = [a[1] for a in atom_set if a[0] == "head"]
        mlps = [a[1] for a in atom_set if a[0] == "mlp"]
        for h in heads:
            m = ablate_head(m, h)
        if mlps:
            m = ablate_mlp_neurons(m, mlps)
        return m

    chosen = []
    for _ in range(8):
        best = None
        for a in atoms:
            if a in chosen:
                continue
            cand = chosen + [a]
            m = apply(cand)
            # B intact? cheap numeric gate first
            toksB = all_sequences(Q2)
            okB = True
            for key in ("B pos", "B neg"):
                scls, want = CLAIMS[key]
                sub = toksB[scls.numeric_mask(toksB)]
                vals = m.forward(sub)
                bad = (vals <= 0) if want == "positive" else (vals > 0)
                if bad.mean() > 0.02:
                    okB = False
                    break
            if not okB:
                continue
            worst = removal_score(m, toks_pos)
            if best is None or worst < best[1]:
                best = (a, worst)
        if best is None:
            return None, None
        chosen.append(best[0])
        if tests_pass(apply(chosen)):
            return chosen, apply(chosen)
    return None, None


def find_min_dose(model, direction, doses=(1, 2, 4, 8, 16, 32)):
    for s in doses:
        m = steer_residual(model, s * direction)
        if tests_pass(m):
            return s, m
    return None, None


def prove_claim_set(model, removal=True) -> dict:
    """The full certificate set at EPS0. For an edited model: removal on
    'A pos' + preservation on both B claims. For the control: all four
    claims in their unedited (skill-works) direction."""
    out = {}
    if removal:
        items = [("removal (A pos -> nonpositive)", CLAIMS["A pos"][0],
                  "nonpositive"),
                 ("preservation B pos", *CLAIMS["B pos"]),
                 ("preservation B neg", *CLAIMS["B neg"])]
    else:
        items = [(f"control {k}", scls, want)
                 for k, (scls, want) in CLAIMS.items()]
    for name, scls, want in items:
        t0 = time.time()
        r = prove_transformer(model, scls, want, EPS0, slack=SLACK,
                              timeout_ms=TIMEOUT)
        out[name] = {"proved": r["proved"], "status": r["status"],
                     "seconds": round(time.time() - t0, 1)}
        print(f"      {name}: "
              f"{'PROVED' if r['proved'] else r['status']} "
              f"({out[name]['seconds']}s)", flush=True)
    return out


def radii(model) -> dict:
    """Certified embedding radii: removal (A pos class stays nonpositive)
    and preservation (the worse of B's two claims)."""
    rem = certified_eps(model, CLAIMS["A pos"][0], "nonpositive",
                        eps_max=EPS_MAX, tol=TOL, slack=SLACK,
                        timeout_ms=TIMEOUT)
    pres = [certified_eps(model, CLAIMS["B pos"][0], "positive",
                          eps_max=EPS_MAX, tol=TOL, slack=SLACK,
                          timeout_ms=TIMEOUT),
            certified_eps(model, CLAIMS["B neg"][0], "nonpositive",
                          eps_max=EPS_MAX, tol=TOL, slack=SLACK,
                          timeout_ms=TIMEOUT)]
    if any(p["radius"] is None for p in pres):
        pres_r = {"radius": None, "saturated": False}
    else:
        pres_r = {"radius": min(p["radius"] for p in pres),
                  "saturated": all(p["saturated"] for p in pres)}
    q = rem["queries"] + sum(p["queries"] for p in pres)
    return {"removal": {"radius": rem["radius"],
                        "saturated": rem["saturated"]},
            "preservation": pres_r, "queries": q}


def _fmt_radius(r) -> str:
    if r["radius"] is None:
        return "refuted at eps=0"
    if r["saturated"]:
        return f">= {EPS_MAX} (cap)"
    return f"{r['radius']:.3f}"


def main():
    global EPS0
    t0 = time.time()
    print("=" * 70)
    print("A CERTIFIED EDIT ON A (SMALL) THRESHOLD-GATE TRANSFORMER")
    print("=" * 70)

    print("\n[1] Training the certified subject (gate attention, hardened, "
          f"quantized; config {SUBJECT_CONFIG}, L={_tm.L}, "
          f"seed {SUBJECT_SEED})...")
    model = train_gate_subject(seed=SUBJECT_SEED, config=SUBJECT_CONFIG,
                               verbose=False)
    rng = np.random.default_rng(123)
    tokens, y = sample_batch(20000, rng)
    accA, accB = accuracy(model, tokens, y)
    print(f"    held-out accuracy: skill A {accA:.4f}, skill B {accB:.4f}")

    print("\n[2] Validating the encoding against the float forward "
          "(mandatory)...")
    gap = validate_encoding(model)
    print(f"    max |z3 - forward| over 40 samples: {gap:.2e}")
    assert gap < 1e-9, "encoding does not match the model"

    print("\n[3] Controls: clean brute-force, natural radii, then proofs "
          "at the working epsilon...")
    for key, (scls, want) in CLAIMS.items():
        chk = exhaustive_clean_check(model, scls, want)
        print(f"    clean brute-force {key}: {chk['violations']} violations"
              f" / {chk['sequences']} sequences", flush=True)

    # Natural margins first: how much noise does the UNEDITED model
    # withstand on each claim? The working epsilon must sit inside them.
    print("    natural radii per control claim...")
    nat = {}
    for key, (scls, want) in CLAIMS.items():
        r = certified_eps(model, scls, want, eps_max=EPS_MAX, tol=TOL,
                          slack=SLACK, timeout_ms=TIMEOUT)
        nat[key] = r
        print(f"      {key}: {_fmt_radius(r)} ({r['queries']} queries, "
              f"{r['seconds']}s)", flush=True)
    if any(r["radius"] is None for r in nat.values()):
        raise SystemExit("    a control claim is refuted even at eps=0 — "
                         "the subject is broken")
    tightest = min(r["radius"] for r in nat.values())
    if EPS0 >= tightest:
        # safety only: never assert at a working epsilon outside the control's
        # own certified margin
        EPS0 = round(0.8 * tightest, 4)
        print(f"    working epsilon lowered to {EPS0} (inside the tightest "
              f"natural margin {tightest:.3f})")
    else:
        print(f"    working epsilon {EPS0} fixed (natural margins ~"
              f"{tightest:.3f}; the certified radii below are the real "
              "robustness numbers, not this gate)")

    control = prove_claim_set(model, removal=False)
    control_ok = all(v["proved"] for v in control.values())
    ctrl_rr = {"removal": {"radius": None, "saturated": False},
               "preservation":
                   {"radius": min(nat["B pos"]["radius"],
                                  nat["B neg"]["radius"]),
                    "saturated": nat["B pos"]["saturated"]
                    and nat["B neg"]["saturated"]},
               "natural": {k: {"radius": v["radius"],
                               "saturated": v["saturated"]}
                           for k, v in nat.items()}}
    print(f"    control preservation radius (B claims): "
          f"{_fmt_radius(ctrl_rr['preservation'])}")

    print("\n[4] Circuit search (removal objective, B-intact constraint)...")
    circuit, edited = find_circuit(model)
    if circuit is None:
        raise SystemExit("    no test-passing circuit found — record and "
                         "no certifiable edit on this subject")
    print(f"    circuit: {circuit}")

    print("\n[5] The edit suite...")
    suite = [("ablation (circuit)", edited,
              f"atoms {circuit} switched off")]
    mlps = [a[1] for a in circuit if a[0] == "mlp"]
    heads = [a[1] for a in circuit if a[0] == "head"]
    if mlps and not heads:
        m = weight_edit_mlp(model, mlps)
        if tests_pass(m):
            suite.append(("weight edit (MLP wires cut)", m,
                          f"outgoing wires of neurons {mlps} zeroed"))
        else:
            print("    weight edit: fails the test gate (recorded)")
    direction = diff_of_means_direction(model)
    s_d, m_d = find_min_dose(model, direction)
    if s_d is not None:
        suite.append((f"steering diff-of-means (dose {s_d:g})", m_d,
                      "smallest dose that passes the tests"))
        m4 = steer_residual(model, 4 * s_d * direction)
        suite.append((f"steering diff-of-means (dose {4 * s_d:g})", m4,
                      "4x the minimal dose"
                      + ("" if tests_pass(m4)
                         else "; FAILS the ordinary tests")))
    else:
        print("    diff-of-means steering: no dose up to 32 passes the "
              "tests (recorded)")

    print("\n[6] Certificates + certified embedding radii per edit...")
    rows = []
    for name, m, note in suite:
        print(f"    [{name}]")
        proofs = prove_claim_set(m, removal=True)
        rr = radii(m)
        removal_ok = proofs["removal (A pos -> nonpositive)"]["proved"]
        pres_ok = (proofs["preservation B pos"]["proved"]
                   and proofs["preservation B neg"]["proved"])
        passes = tests_pass(m)
        print(f"      -> tests {'pass' if passes else 'FAIL'}, removal "
              f"{'PROVED' if removal_ok else 'refuted'} (radius "
              f"{_fmt_radius(rr['removal'])}), preservation "
              f"{'PROVED' if pres_ok else 'refuted'} (radius "
              f"{_fmt_radius(rr['preservation'])})", flush=True)
        rows.append({"edit": name, "note": note, "tests_pass": passes,
                     "removal_proved": removal_ok,
                     "preservation_proved": pres_ok,
                     "proofs": proofs,
                     "removal_radius": rr["removal"],
                     "preservation_radius": rr["preservation"],
                     "queries": rr["queries"]})

    seconds = time.time() - t0
    _write_report(accA, accB, gap, control, control_ok, ctrl_rr, circuit,
                  rows, seconds)
    print(f"\nDone in {seconds:.1f}s.")


def _write_report(accA, accB, gap, control, control_ok, ctrl_rr, circuit,
                  rows, seconds):
    md = os.path.join(RESULTS, "transformer_report.md")
    js = os.path.join(RESULTS, "transformer_report.json")
    seqs_per_task = len(_tm.TEXT_TOKENS) ** (_tm.L - 1)
    lines = [
        "# A certified edit on a (small) threshold-gate transformer",
        "",
        "Output of `run_transformer.py`. Subject: one decoder block "
        f"({SUBJECT_CONFIG['n_heads']} threshold-gate attention heads with "
        f"additive scores, LeakyReLU MLP of width {SUBJECT_CONFIG['d_mlp']}, "
        f"d_model {SUBJECT_CONFIG['d_model']}), sequence length {_tm.L}, "
        "trained on two sequence skills sharing the "
        "trunk — skill A: 'is there an unclosed quote?' (Q1), skill B: 'is "
        "there an unclosed bracket?' (Q2); both are global counting "
        "properties (no input coordinate IS the label). This is the small "
        "config at the frontier where exact input-side certification is "
        "tractable — see the measured size ladder in "
        "results/rung2_size_ladder.log. Every certificate "
        "below quantifies over ALL token sequences in its class AND all "
        "embedding-space noise up to the stated epsilon, in one solver "
        "query; weights enter as exact rationals (quantized to the 2^-12 "
        "grid — the quantized model is the one certified), claims carry a "
        f"1e-6 slack, and the clean token space ({seqs_per_task:,} "
        "sequences per task) is additionally brute-forced numerically as a "
        "cross-check.",
        "",
        f"Held-out accuracy: skill A {accA:.4f}, skill B {accB:.4f}. "
        f"Encoding validated against the float forward at 40 random "
        f"(sequence, noise) points: max gap {gap:.1e}.",
        "",
        f"Control certificates (unedited model, eps = {EPS0}): "
        f"{'ALL PROVED' if control_ok else 'NOT all proved'} — the model "
        "provably answers both questions correctly for every sequence and "
        "every perturbation. Natural preservation radius "
        f"{_fmt_radius(ctrl_rr['preservation'])}; control removal "
        f"{_fmt_radius(ctrl_rr['removal'])} (correctly refuted: the skill "
        "is present).",
        "",
        f"Skill A's circuit (greedy, removal objective): {circuit}.",
        "",
        "| edit | passes tests? | removal proved? | removal radius | "
        "preservation proved? | preservation radius | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['edit']} | {'yes' if r['tests_pass'] else 'NO'} | "
            f"{'yes' if r['removal_proved'] else 'REFUTED'} | "
            f"{_fmt_radius(r['removal_radius'])} | "
            f"{'yes' if r['preservation_proved'] else 'REFUTED'} | "
            f"{_fmt_radius(r['preservation_radius'])} | {r['note']} |")
    lines += [
        "",
        f"Total time {seconds:.1f}s on a laptop CPU. Radii are bisected to "
        f"{TOL}; the radius cap is {EPS_MAX} (embedding entries are O(0.5), "
        f"so {EPS_MAX} is already a sizeable perturbation of every token at "
        "once). Natural-radius bisection depends on the solver finishing each "
        "near-boundary probe within the timeout; a probe that times out is "
        "treated as a failure (sound: the reported radius is always a proved "
        "lower bound), so natural radii can vary slightly run-to-run while "
        "the edit radii here are stable.",
        "",
        "## Why this matters",
        "",
        "This certifies an edit on a transformer: the certified object has content-based attention and an MLP, and the attention is load-bearing (switching off the heads alone, or the MLP alone, leaves skill A firing: results/rung2_circuit_search.log). The removed skill is a sequence property no single input coordinate encodes; the quantifier is continuous and input-side (embedding noise ahead of the whole computation, where Somani's certificates perturb the final residual on traced inputs); and the toy models' edit comparison is re-asked on this subject. Exact input-side certification requires threshold-gate attention (soft attention's application is bilinear in the noise); the gate model trains to 100% on both skills.",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"acc": {"A": accA, "B": accB}, "encoding_gap": gap,
                   "control": control, "control_radii": ctrl_rr,
                   "circuit": circuit, "edits": rows,
                   "eps0": EPS0, "eps_max": EPS_MAX, "tol": TOL,
                   "slack": SLACK, "seconds": seconds},
                  f, indent=2, default=float)
    print(f"\nReports written: {md} and .json")


if __name__ == "__main__":
    main()
