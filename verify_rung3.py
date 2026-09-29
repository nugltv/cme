"""
verify_rung3.py — the exact certificate for the modular-adder edit
==================================================================

The modular-adder prover. It certifies the STRONG
removal claim on the two-adder subject (`mod_arith_model.py`, mode 'twoadd'):

    after ablating skill A's circuit, under task ADD the p-way output is
    INDEPENDENT of the summand positions (a1, b1) — for EVERY sequence and EVERY
    embedding-space perturbation up to epsilon, changing a1/b1 does not move any
    output logit. So the edited model provably no longer READS the numbers it is
    supposed to add: it cannot compute (a1+b1) mod p.

Encoding: a two-copy / siamese construction (the same idea as
verify_transformer.encode_influence_pair, generalised to the p-way readout and to
FREE POSITIONS rather than free tokens). Two copies of [a1, b1, a2, b2, ADD] share
the task token, the embedding noise, and the distractor positions (a2, b2), and
are free — an independent digit hull per copy — only at the summand positions
(a1, b1). We ask the solver for a pair whose outputs differ by more than `bound`
on some class; unsat = independence proved (bound = 0 = exact). Everything runs on
the SHARED, validated forward trunk `verify_transformer._forward_x2` +
`_forward_logits`; weights enter as exact rationals; the hull relaxation makes a
proof cover every discrete sequence (P1).

The subject uses mod_arith_model's sequence length; callers must set BOTH
`mod_arith_model.set_len(L)` and `transformer_model.set_seq_len(L)` (the shared
trunk reads the latter) to the same L before encoding.
"""

from __future__ import annotations
from fractions import Fraction
import numpy as np
import z3

import transformer_model as _tm
from verify_transformer import _q, _forward_logits


def encode_twoadd_pair(model, s, eps, cfg, free_pos, task_tok, hints=True,
                       pin=None):
    """Two coupled copies of a twoadd sequence. Shared: the task token, the
    noise, and every non-free text position. Free (independent per copy): the
    positions in `free_pos` (each an independent digit hull). Returns
    (logitsA, logitsB), each a length-p list of the p-way output logits.

    pin = (tokens1, tokens2, noise) fixes both copies to concrete sequences
    (agreeing outside free_pos) for the encoding-validation gate."""
    L = _tm.L
    p = cfg["p"]
    d = model.d
    t = L - 1
    digits = list(range(p))                      # digit tokens 0..p-1
    free = set(free_pos)

    def emb_vals(pos, toks):
        return np.array([model.E[tk] + model.P[pos] for tk in toks])

    xsA, xsB, xsA_iv, xsB_iv = [], [], [], []
    for pos in range(L):
        if pin is not None:
            delta = [_q(pin[2][pos, i]) for i in range(d)]
        elif eps == 0:
            delta = [_q(0.0) for _ in range(d)]
        else:
            delta = [z3.Real(f"n_{pos}_{i}") for i in range(d)]
            for dd in delta:
                s.add(dd >= _q(-eps), dd <= _q(eps))

        if pos == t:                              # task token (concrete, shared)
            base = [_q(model.E[task_tok, i] + model.P[pos, i])
                    for i in range(d)]
            baseA = baseB = base
            vals = np.array([model.E[task_tok] + model.P[pos]])
        elif pin is not None:                     # concrete digits per copy
            baseA = [_q(model.E[pin[0][pos], i] + model.P[pos, i])
                     for i in range(d)]
            baseB = [_q(model.E[pin[1][pos], i] + model.P[pos, i])
                     for i in range(d)]
            vals = emb_vals(pos, digits)
        else:                                     # symbolic digit hull
            vals = emb_vals(pos, digits)
            lamA = [z3.Real(f"la_{pos}_{k}") for k in range(p)]
            for lv in lamA:
                s.add(lv >= _q(0.0), lv <= _q(1.0))
            s.add(sum(lamA[1:], lamA[0]) == _q(1.0))
            if pos in free:                       # copy B gets its OWN hull
                lamB = [z3.Real(f"lb_{pos}_{k}") for k in range(p)]
                for lv in lamB:
                    s.add(lv >= _q(0.0), lv <= _q(1.0))
                s.add(sum(lamB[1:], lamB[0]) == _q(1.0))
            else:                                 # shared: same hull weights
                lamB = lamA

            def mk(lam):
                out = []
                for i in range(d):
                    e = _q(model.P[pos, i])
                    for k in range(p):
                        e = e + lam[k] * _q(model.E[digits[k], i])
                    out.append(e)
                return out
            baseA, baseB = mk(lamA), mk(lamB)

        if pin is not None:
            eA = model.E[pin[0][pos]] + model.P[pos] + pin[2][pos]
            eB = model.E[pin[1][pos]] + model.P[pos] + pin[2][pos]
            loA = hiA = eA
            loB = hiB = eB
        else:
            loA = loB = vals.min(axis=0) - eps
            hiA = hiB = vals.max(axis=0) + eps
        xsA.append([z3.Real(f"a_x_{pos}_{i}") for i in range(d)])
        xsB.append([z3.Real(f"b_x_{pos}_{i}") for i in range(d)])
        for i in range(d):
            s.add(xsA[pos][i] == baseA[i] + delta[i])
            s.add(xsB[pos][i] == baseB[i] + delta[i])
        xsA_iv.append((np.asarray(loA, float), np.asarray(hiA, float)))
        xsB_iv.append((np.asarray(loB, float), np.asarray(hiB, float)))

    logitsA = _forward_logits(model, s, xsA, xsA_iv, hints=hints, prefix="a_")
    logitsB = _forward_logits(model, s, xsB, xsB_iv, hints=hints, prefix="b_")
    return logitsA, logitsB


def prove_summand_independence(model, cfg, eps, bound=0.0, free_pos=(0, 1),
                               task_tok=None, timeout_ms=300000,
                               hints=True) -> dict:
    """Prove: no pair of sequences differing only at the summand positions
    moves ANY output logit by more than `bound`, over all embedding noise up to
    eps. unsat = proved (bound=0 = the output is exactly independent of the
    summands -> the model cannot compute the sum)."""
    if task_tok is None:
        task_tok = cfg["ADD"]
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    lA, lB = encode_twoadd_pair(model, s, eps, cfg, free_pos, task_tok,
                                hints=hints)
    s.add(z3.Or([z3.Or(lA[k] - lB[k] > _q(bound), lA[k] - lB[k] < _q(-bound))
                 for k in range(len(lA))]))
    res = s.check()
    if res == z3.unsat:
        return {"proved": True, "status": "proved"}
    if res == z3.sat:
        return {"proved": False, "status": "refuted (a summand-only change moves "
                "an output logit by more than the bound)"}
    return {"proved": False, "status": f"unknown ({res})"}


def certified_summand_radius(model, cfg, free_pos=(0, 1), task_tok=None,
                             eps_max=0.05, tol=5e-3, timeout_ms=300000) -> dict:
    """Largest embedding-noise eps for which EXACT (bound=0) summand-
    independence still proves, by bisection (verify_transformer.certified_eps's
    recipe/soundness)."""
    import time as _time
    t0 = _time.time()
    q = 0

    def probe(e):
        nonlocal q
        q += 1
        return prove_summand_independence(model, cfg, e, 0.0, free_pos,
                                          task_tok, timeout_ms=timeout_ms)
    base = probe(0.0)
    if not base["proved"]:
        return {"radius": None, "saturated": False, "queries": q,
                "seconds": round(_time.time() - t0, 1)}
    top = probe(eps_max)
    if top["proved"]:
        return {"radius": eps_max, "saturated": True, "queries": q,
                "seconds": round(_time.time() - t0, 1)}
    lo, hi = 0.0, eps_max
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if probe(mid)["proved"]:
            lo = mid
        else:
            hi = mid
    return {"radius": lo, "saturated": False, "queries": q,
            "seconds": round(_time.time() - t0, 1)}


def _build_seq_xs(model, s, eps, cfg, task_tok, hints=True, prefix=""):
    """Build one hull sequence [digit, digit, digit, digit, task] + noise as z3
    embeddings (xs) with interval bounds — shared by both models in the
    preservation proof (edited and unedited have the same E, P, so the same xs)."""
    L = _tm.L
    p = cfg["p"]
    d = model.d
    t = L - 1
    digits = list(range(p))
    xs, xs_iv = [], []
    for pos in range(L):
        if eps == 0:
            delta = [_q(0.0) for _ in range(d)]
        else:
            delta = [z3.Real(f"{prefix}n_{pos}_{i}") for i in range(d)]
            for dd in delta:
                s.add(dd >= _q(-eps), dd <= _q(eps))
        if pos == t:
            base = [_q(model.E[task_tok, i] + model.P[pos, i]) for i in range(d)]
            vals = np.array([model.E[task_tok] + model.P[pos]])
        else:
            vals = np.array([model.E[tk] + model.P[pos] for tk in digits])
            lam = [z3.Real(f"{prefix}lam_{pos}_{k}") for k in range(p)]
            for lv in lam:
                s.add(lv >= _q(0.0), lv <= _q(1.0))
            s.add(sum(lam[1:], lam[0]) == _q(1.0))
            base = []
            for i in range(d):
                e = _q(model.P[pos, i])
                for k in range(p):
                    e = e + lam[k] * _q(model.E[digits[k], i])
                base.append(e)
        row = [z3.Real(f"{prefix}x_{pos}_{i}") for i in range(d)]
        for i in range(d):
            s.add(row[i] == base[i] + delta[i])
        xs.append(row)
        xs_iv.append((vals.min(axis=0) - eps, vals.max(axis=0) + eps))
    return xs, xs_iv


def prove_preservation_unchanged(edited, unedited, cfg, eps, task_tok=None,
                                 bound=0.0, timeout_ms=300000,
                                 hints=True) -> dict:
    """Prove: under the PRESERVED task, the edit changes NO output logit, for
    every sequence and all embedding noise up to eps. Both models run on the
    SAME hull sequence (they share E, P — only Wo differs); unsat = the edit
    provably leaves skill B untouched."""
    if task_tok is None:
        task_tok = cfg["SUB"]
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    xs, xs_iv = _build_seq_xs(unedited, s, eps, cfg, task_tok, hints=hints)
    lo_e = _forward_logits(edited, s, xs, xs_iv, hints=hints, prefix="e_")
    lo_u = _forward_logits(unedited, s, xs, xs_iv, hints=hints, prefix="u_")
    s.add(z3.Or([z3.Or(lo_e[k] - lo_u[k] > _q(bound),
                       lo_e[k] - lo_u[k] < _q(-bound))
                 for k in range(len(lo_e))]))
    res = s.check()
    if res == z3.unsat:
        return {"proved": True, "status": "proved"}
    if res == z3.sat:
        return {"proved": False, "status": "refuted (the edit moves a skill-B "
                "output)"}
    return {"proved": False, "status": f"unknown ({res})"}


def _build_concrete_xs(model, s, seq, eps, prefix=""):
    """Build z3 embeddings for ONE concrete sequence `seq` (fixed token ids at
    every position) plus a continuous noise ball of radius eps. Returns (xs,
    xs_iv) for the shared forward trunk. Only the noise is symbolic, so this is a
    much lighter query than the hull encodings."""
    L = _tm.L
    d = model.d
    xs, xs_iv = [], []
    for pos in range(L):
        tok = int(seq[pos])
        val = model.E[tok] + model.P[pos]                 # concrete embedding
        base = [_q(model.E[tok, i] + model.P[pos, i]) for i in range(d)]
        if eps == 0:
            delta = [_q(0.0) for _ in range(d)]
        else:
            delta = [z3.Real(f"{prefix}n_{pos}_{i}") for i in range(d)]
            for dd in delta:
                s.add(dd >= _q(-eps), dd <= _q(eps))
        row = [z3.Real(f"{prefix}x_{pos}_{i}") for i in range(d)]
        for i in range(d):
            s.add(row[i] == base[i] + delta[i])
        xs.append(row)
        xs_iv.append((val - eps, val + eps))
    return xs, xs_iv


def prove_skillB_correct(edited, cfg, seq, correct, eps, timeout_ms=300000,
                         slack=0.0, hints=True) -> dict:
    """Prove: for the concrete PRESERVED-task sequence `seq`, the edited model's
    output class stays `correct` (= (a2+b2) mod p) for EVERY embedding
    perturbation up to eps. We ask for a counterexample — some other class k
    whose logit reaches within `slack` of the correct class's — and get none.
    unsat = skill B is provably still computed correctly over the whole ball.
    This is the noise-robust preservation claim in its natural form (argmax
    correctness), lighter than exact logit-equality because only noise is free."""
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    xs, xs_iv = _build_concrete_xs(edited, s, seq, eps, prefix="e_")
    lo = _forward_logits(edited, s, xs, xs_iv, hints=hints, prefix="e_")
    # violation: some wrong class k reaches (within slack of) the correct logit
    s.add(z3.Or([lo[k] - lo[correct] >= _q(-slack)
                 for k in range(len(lo)) if k != correct]))
    res = s.check()
    if res == z3.unsat:
        return {"proved": True, "status": "proved"}
    if res == z3.sat:
        return {"proved": False, "status": "refuted (a wrong class ties/beats "
                "the correct one under noise)"}
    return {"proved": False, "status": f"unknown ({res})"}


def certified_preservation_radius(edited, cfg, task_tok=None, eps_max=0.05,
                                  tol=5e-3, timeout_ms=300000, slack=0.0,
                                  hints=True) -> dict:
    """Largest embedding-noise eps for which the edited model STILL classifies
    EVERY preserved-task (SUB) sequence correctly — i.e. skill B is provably
    intact over all sequences × a continuous ball of this radius. Bisection on a
    single global eps; at each eps every SUB sequence is checked, early-exiting on
    the first failure (that sequence + its failing eps is the binding constraint).
    Returns the radius and the binding sequence."""
    import time as _time
    from mod_arith_model import all_sequences, true_label
    if task_tok is None:
        task_tok = cfg["SUB"]
    toks = all_sequences(cfg)
    sub = toks[toks[:, -1] == task_tok]
    labels = true_label(cfg, toks)[toks[:, -1] == task_tok]
    t0 = _time.time()
    q = {"n": 0}

    def all_correct(eps):
        """True iff every SUB sequence certifies at this eps. Returns
        (ok, binding_seq_or_None)."""
        for seq, c in zip(sub, labels):
            q["n"] += 1
            r = prove_skillB_correct(edited, cfg, seq, int(c), eps,
                                     timeout_ms=timeout_ms, slack=slack,
                                     hints=hints)
            if not r["proved"]:
                return False, (seq.tolist(), r["status"])
        return True, None

    base_ok, binding = all_correct(0.0)
    if not base_ok:
        return {"radius": None, "saturated": False, "queries": q["n"],
                "binding": binding, "seconds": round(_time.time() - t0, 1)}
    top_ok, _ = all_correct(eps_max)
    if top_ok:
        return {"radius": eps_max, "saturated": True, "queries": q["n"],
                "binding": None, "seconds": round(_time.time() - t0, 1)}
    lo, hi, binding = 0.0, eps_max, None
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        ok, b = all_correct(mid)
        if ok:
            lo = mid
        else:
            hi, binding = mid, b
    return {"radius": lo, "saturated": False, "queries": q["n"],
            "binding": binding, "seconds": round(_time.time() - t0, 1)}


def validate_concrete_encoding(model, cfg, task_tok=None, n=10, eps=0.03,
                               seed=23) -> float:
    """Gate for the preservation prover: the concrete single-sequence encoding
    (`_build_concrete_xs` + `_forward_logits`) must equal the float p-way forward
    at pinned (sequence, noise) points, before any preservation proof is trusted."""
    from mod_arith_model import all_sequences
    L = _tm.L
    if task_tok is None:
        task_tok = cfg["SUB"]
    rng = np.random.default_rng(seed)
    subs = all_sequences(cfg)
    subs = subs[subs[:, -1] == task_tok]
    worst = 0.0
    for _ in range(n):
        seq = subs[rng.integers(0, len(subs))]
        noise = rng.uniform(-eps, eps, size=(L, model.d))
        s = z3.Solver()
        xs, xs_iv = [], []
        for pos in range(L):
            tok = int(seq[pos])
            val = model.E[tok] + model.P[pos]
            row = [z3.Real(f"x_{pos}_{i}") for i in range(model.d)]
            for i in range(model.d):
                s.add(row[i] == _q(model.E[tok, i] + model.P[pos, i]
                                   + noise[pos, i]))
            xs.append(row)
            xs_iv.append((val - eps, val + eps))
        lo = _forward_logits(model, s, xs, xs_iv)
        assert s.check() == z3.sat
        mdl = s.model()

        def val_(x):
            v = mdl.eval(x, model_completion=True)
            return float(Fraction(int(v.numerator_as_long()),
                                  int(v.denominator_as_long())))
        got = np.array([val_(x) for x in lo])
        want = model.forward(seq[None], noise=noise[None])[0]
        worst = max(worst, float(np.abs(got - want).max()))
    return worst


def validate_twoadd_encoding(model, cfg, free_pos=(0, 1), task_tok=None,
                             n=12, eps=0.03, seed=17) -> float:
    """Gate: the two-copy encoding must equal the float p-way forward at pinned
    pairs (sequences agreeing outside free_pos) before any proof is trusted."""
    import mod_arith_model as MA
    L = _tm.L
    p = cfg["p"]
    if task_tok is None:
        task_tok = cfg["ADD"]
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        base = rng.integers(0, p, L - 1)
        t1 = base.copy()
        t2 = base.copy()
        for pos in free_pos:
            t1[pos] = rng.integers(0, p)
            t2[pos] = rng.integers(0, p)
        toks1 = np.append(t1, task_tok)
        toks2 = np.append(t2, task_tok)
        noise = rng.uniform(-eps, eps, size=(L, model.d))
        s = z3.Solver()
        lA, lB = encode_twoadd_pair(model, s, eps, cfg, free_pos, task_tok,
                                    pin=(toks1, toks2, noise))
        assert s.check() == z3.sat
        m = s.model()

        def val(x):
            v = m.eval(x, model_completion=True)
            return float(Fraction(int(v.numerator_as_long()),
                                  int(v.denominator_as_long())))
        gotA = np.array([val(x) for x in lA])
        gotB = np.array([val(x) for x in lB])
        wantA = model.forward(toks1[None], noise=noise[None])[0]
        wantB = model.forward(toks2[None], noise=noise[None])[0]
        worst = max(worst, float(np.abs(gotA - wantA).max()),
                    float(np.abs(gotB - wantB).max()))
    return worst
