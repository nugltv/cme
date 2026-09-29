"""
verify_transformer.py — exact Z3 encoding of the tiny transformer, with the
                        all-sequences-times-noise quantifier
============================================================================

The threshold-gate transformer prover (the measured frontier:
results/rung2_size_ladder.log, paper Table III). One solver query
covers, at once:

  * EVERY token sequence in a claim's class (token choices are Boolean
    one-hot selectors; the class — e.g. "more quote-opens than closes" —
    is a cardinality constraint over those Booleans), and
  * EVERY embedding-space perturbation of every position up to epsilon
    (a continuous |delta|_inf <= eps variable per position and dimension,
    added to the selected embedding BEFORE anything else runs).

unsat = the claim holds for the entire (exponentially many balls) region.
No enumeration could do this: the token space alone is enumerable, but the
noise is continuous — which is exactly the part testing cannot cover
(proposition P3; the grid/solver division of labor of verify.py).

The encoded model is the GATE-attention variant of transformer_model.py
(hardened 0/1 gates, quantized weights): threshold attention keeps the
weight-times-value application piecewise-linear, so the whole forward pass
lands in decidable linear real arithmetic. Soft attention's application is
bilinear in the noise — not encodable exactly. Same house rules as verify.py: every weight enters
as an exact rational, claims are proved by absence of counterexample, and
`slack` demands the logit clear zero by a margin (float-gap discipline; see results/float_gap_report.md).

Because the model reads its answer at the last position of a single block,
only the last attention row exists in the encoding — one row per head.
"""

from __future__ import annotations
from fractions import Fraction
import numpy as np
import z3

import transformer_model as _tm
from transformer_model import (TinyTransformer, V, TEXT_TOKENS,
                               Q1, Q2, QO, QC, BO, BC)


def _q(value) -> z3.ArithRef:
    fr = Fraction(float(value))
    return z3.RealVal(fr.numerator) / z3.RealVal(fr.denominator)


def _fresh(s, name, expr):
    """Name an intermediate: keeps Z3's terms small (a measured multi-minute
    difference on this model)."""
    v = z3.Real(name)
    s.add(v == expr)
    return v


def _pad(lo, hi):
    # The interval bounds are computed in round-to-nearest float64, so a bound
    # can land a hair INSIDE the true exact-rational value (the float
    # gap, biting our own tooling — caught by the encoding validation).
    # Outward-pad every bound: loosening is always sound.
    return 1e-9 + 1e-12 * max(abs(float(lo)), abs(float(hi)))


def _forward_x2(model, s, xs, xs_iv, hints=True, prefix=""):
    """The transformer trunk, from a list of per-position embedding vectors
    `xs` (each a list of z3 Reals, already including token choice and noise)
    with matching interval bounds `xs_iv`, down to x2 — the post-MLP residual
    just BEFORE the readout — as exact Z3 real arithmetic with valid interval
    hints. Returns (x2, x2lo, x2hi).

    This is the soundness-critical core SHARED by every certificate: the binary
    readout (`_forward_logit`, threshold-gate subject) and the p-way readout
    (`_forward_logits`, modular adders) both apply their readout on top of this, and the
    removal/preservation and siamese provers all build their `xs` and call it.
    `prefix` namespaces the intermediate variables so two copies of the pass can
    live in one solver without name clashes (the siamese encoding). With
    prefix="" the variable names are identical to the original inline version,
    so the existing encoding-validation gate still covers this code path."""
    d, H, dh = model.d, model.H, model.dh
    t = _tm.L - 1

    def fresh_iv(name, expr, lo, hi):
        v = z3.Real(prefix + name)
        s.add(v == expr)
        if hints:
            p = _pad(lo, hi)
            s.add(v >= _q(float(lo) - p), v <= _q(float(hi) + p))
        return v

    def iv_aff(W, b, lo, hi):
        Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
        blo = Wp @ lo + Wn @ hi
        bhi = Wp @ hi + Wn @ lo
        if b is not None:
            blo, bhi = blo + b, bhi + b
        return blo, bhi

    def aff(W, b, vec, iv, nm):
        lo, hi = iv_aff(W, b, *iv)
        out = [fresh_iv(f"{nm}_{r}",
                        sum((_q(W[r, i]) * vec[i] for i in range(len(vec))),
                            _q(0.0))
                        + (_q(b[r]) if b is not None else _q(0.0)),
                        lo[r], hi[r]) for r in range(W.shape[0])]
        return out, (lo, hi)

    def lky(vec, iv):
        lo, hi = iv
        out = [z3.If(x >= 0, x, _q(model.alpha) * x) for x in vec]
        llo = np.where(lo >= 0, lo, model.alpha * lo)
        lhi = np.where(hi >= 0, hi, model.alpha * hi)
        return out, (llo, lhi)

    o_all, o_all_iv = [], ([], [])
    for h in range(H):
        qv, qiv = aff(model.Wq[h], model.bq[h], xs[t], xs_iv[t], f"q{h}")
        o = [_q(0.0)] * dh
        olo, ohi = np.zeros(dh), np.zeros(dh)
        for pos in range(_tm.L):
            kv, kiv = aff(model.Wk[h], model.bk[h], xs[pos], xs_iv[pos],
                          f"k{h}_{pos}")
            raw = [qv[r] + kv[r] for r in range(dh)]
            riv = (qiv[0] + kiv[0], qiv[1] + kiv[1])
            if model.score_act == "linear":
                pre, piv = raw, riv
            else:
                pre, piv = lky(raw, riv)
            up, un = np.maximum(model.u[h], 0), np.minimum(model.u[h], 0)
            slo = float(up @ piv[0] + un @ piv[1])
            shi = float(up @ piv[1] + un @ piv[0])
            sc = fresh_iv(f"sc{h}_{pos}",
                          sum((_q(model.u[h][r]) * pre[r]
                               for r in range(dh)), _q(0.0)), slo, shi)
            vv, viv = aff(model.Wv[h], None, xs[pos], xs_iv[pos],
                          f"v{h}_{pos}")
            sp = _pad(slo, shi)
            if hints and slo - sp > 0:
                glo, ghi = viv
                o = [o[r] + vv[r] for r in range(dh)]
            elif hints and shi + sp <= 0:
                glo, ghi = np.zeros(dh), np.zeros(dh)
            else:
                glo = np.minimum(viv[0], 0)
                ghi = np.maximum(viv[1], 0)
                o = [o[r] + z3.If(sc > 0, vv[r], _q(0.0)) for r in range(dh)]
            olo, ohi = olo + glo, ohi + ghi
        for r in range(dh):
            o_all.append(fresh_iv(f"o{h}_{r}", o[r], olo[r], ohi[r]))
        o_all_iv[0].extend(olo)
        o_all_iv[1].extend(ohi)

    oiv = (np.array(o_all_iv[0]), np.array(o_all_iv[1]))
    wlo, whi = iv_aff(model.Wo, model.bo, *oiv)
    x1lo, x1hi = xs_iv[t][0] + wlo, xs_iv[t][1] + whi
    x1 = [fresh_iv(f"x1_{i}",
                   xs[t][i] + sum((_q(model.Wo[i, j]) * o_all[j]
                                   for j in range(H * dh)), _q(0.0))
                   + _q(model.bo[i]), x1lo[i], x1hi[i]) for i in range(d)]
    z1, z1iv = aff(model.W1, model.b1, x1, (x1lo, x1hi), "z1")
    a1, a1iv = lky(z1, z1iv)
    mlo, mhi = iv_aff(model.W2, model.b2, *a1iv)
    x2lo, x2hi = x1lo + mlo, x1hi + mhi
    x2 = [fresh_iv(f"x2_{i}",
                   x1[i] + sum((_q(model.W2[i, j]) * a1[j]
                                for j in range(model.m)), _q(0.0))
                   + _q(model.b2[i]), x2lo[i], x2hi[i]) for i in range(d)]
    return x2, x2lo, x2hi


def _forward_logit(model, s, xs, xs_iv, hints=True, prefix=""):
    """The binary (threshold-gate) readout on top of the shared trunk: logit =
    r . x2 + c. Byte-for-byte the original single-logit path when prefix=""."""
    d = model.d
    x2, x2lo, x2hi = _forward_x2(model, s, xs, xs_iv, hints=hints,
                                 prefix=prefix)

    def fresh_iv(name, expr, lo, hi):
        v = z3.Real(prefix + name)
        s.add(v == expr)
        if hints:
            p = _pad(lo, hi)
            s.add(v >= _q(float(lo) - p), v <= _q(float(hi) + p))
        return v

    rp, rn = np.maximum(model.r, 0), np.minimum(model.r, 0)
    llo = float(rp @ x2lo + rn @ x2hi + model.c)
    lhi = float(rp @ x2hi + rn @ x2lo + model.c)
    logit = sum((_q(model.r[i]) * x2[i] for i in range(d)), _q(0.0)) \
        + _q(model.c)
    return fresh_iv("logit", logit, llo, lhi)


def _forward_logits(model, s, xs, xs_iv, hints=True, prefix=""):
    """The p-way (modular-adder) readout on top of the shared trunk: a list of the
    output logits, logits[k] = R[k] . x2 + c[k], one per class. Same exact
    arithmetic and interval hints as the binary readout."""
    d, K = model.d, model.R.shape[0]
    x2, x2lo, x2hi = _forward_x2(model, s, xs, xs_iv, hints=hints,
                                 prefix=prefix)

    def fresh_iv(name, expr, lo, hi):
        v = z3.Real(prefix + name)
        s.add(v == expr)
        if hints:
            p = _pad(lo, hi)
            s.add(v >= _q(float(lo) - p), v <= _q(float(hi) + p))
        return v

    out = []
    for k in range(K):
        Rk = model.R[k]
        rp, rn = np.maximum(Rk, 0), np.minimum(Rk, 0)
        llo = float(rp @ x2lo + rn @ x2hi + model.c[k])
        lhi = float(rp @ x2hi + rn @ x2lo + model.c[k])
        expr = sum((_q(Rk[i]) * x2[i] for i in range(d)), _q(0.0)) \
            + _q(model.c[k])
        out.append(fresh_iv(f"logit_{k}", expr, llo, lhi))
    return out


class SequenceClass:
    """Which sequences a claim quantifies over: the task token, plus a
    label-class constraint on the text (counting comparison over the
    Boolean selectors)."""

    def __init__(self, task: int, kind: str):
        assert task in (Q1, Q2) and kind in ("pos", "neg")
        self.task, self.kind = task, kind
        self.open_t, self.close_t = (QO, QC) if task == Q1 else (BO, BC)

    def describe(self) -> str:
        skill = "quote" if self.task == Q1 else "bracket"
        side = "MORE opens than closes" if self.kind == "pos" \
            else "opens <= closes"
        return f"task {'Q1' if self.task == Q1 else 'Q2'} ({skill}), {side}"

    def constrain(self, s, sel):
        """sel[pos][i] is the Boolean 'text position pos holds TEXT_TOKENS
        [i]'. Add this class's counting constraint."""
        L = _tm.L
        def count(tok):
            idx = TEXT_TOKENS.index(tok)
            return sum((z3.If(sel[pos][idx], 1, 0)
                        for pos in range(L - 1)), 0)
        if self.kind == "pos":
            s.add(count(self.open_t) > count(self.close_t))
        else:
            s.add(count(self.open_t) <= count(self.close_t))

    def constrain_hull(self, s, lam):
        """The RELAXED class constraint over continuous mixture weights
        lam[pos][i] (see token_mode='hull' in encode_logit): open-mass
        exceeds close-mass by >= 1 for the 'pos' class (every discrete
        member has an integer count gap >= 1), by <= 0 for 'neg'. Each
        relaxed class CONTAINS its discrete class, so proving over it
        proves the discrete claim (P1)."""
        io = TEXT_TOKENS.index(self.open_t)
        ic = TEXT_TOKENS.index(self.close_t)
        mass_o = sum((lam[pos][io] for pos in range(len(lam))), _q(0.0))
        mass_c = sum((lam[pos][ic] for pos in range(len(lam))), _q(0.0))
        if self.kind == "pos":
            s.add(mass_o >= mass_c + _q(1.0))
        else:
            s.add(mass_o <= mass_c)

    def numeric_mask(self, tokens: np.ndarray) -> np.ndarray:
        L = _tm.L
        text = tokens[:, :L - 1]
        diff = (text == self.open_t).sum(1) - (text == self.close_t).sum(1)
        return diff > 0 if self.kind == "pos" else diff <= 0


def encode_logit(model: TinyTransformer, s, eps, seq_class: SequenceClass,
                 fixed_tokens=None, fixed_noise=None, hints=True,
                 token_mode="hull"):
    """Build the model's output logit as a Z3 expression over one symbolic
    (sequence, noise) pair drawn from the class. Pass fixed_tokens /
    fixed_noise (concrete values) to pin either — used by the encoding
    validation. Returns (logit, sel) where sel are the token selectors.

    hints=True also asserts INTERVAL BOUNDS on every named intermediate,
    computed by conservative interval arithmetic in the SAME traversal
    that builds the symbolic expression (one loop — no mirrored code to
    drift). Valid bounds exclude no feasible point, so the encoding stays
    exact and complete; they just hand the solver the pruning that pure
    DPLL(T) has to discover the slow way (decisively open/closed gates
    collapse instantly).

    token_mode:
      "bool" — one Boolean selector per (position, token): the region is
      exactly the union of per-sequence noise balls. Measured cost: the
      joint Boolean-times-noise space times out at 30 min even at L = 6.
      "hull" (default) — continuous mixture weights lam[pos] on the token
      simplex: the region is the CONVEX HULL of the token embeddings plus
      noise, with the label class as a mass-gap constraint
      (constrain_hull). This is a STRICTLY LARGER region containing every
      discrete sequence of the class, so 'unsat' still certifies the
      discrete claim (P1) — and there is no Boolean branching at all. A
      'sat' here returns a possibly-FRACTIONAL witness: refutes the
      relaxation, leaves the discrete claim undecided (reported as such)."""
    L = _tm.L
    assert model.attn == "gate" and model.gate_hard, \
        "the certified subject is the hardened gate model"
    d, H, dh = model.d, model.H, model.dh
    t = L - 1

    def fresh_iv(name, expr, lo, hi):
        v = z3.Real(name)
        s.add(v == expr)
        if hints:
            p = _pad(lo, hi)
            s.add(v >= _q(float(lo) - p), v <= _q(float(hi) + p))
        return v

    sel = []
    xs, xs_iv = [], []
    for pos in range(L):
        if fixed_noise is not None:
            delta = [_q(fixed_noise[pos, i]) for i in range(d)]
        else:
            delta = [z3.Real(f"n_{pos}_{i}") for i in range(d)]
            for dd in delta:
                s.add(dd >= _q(-eps), dd <= _q(eps))
        if pos == t:
            vals = np.array([model.E[seq_class.task] + model.P[pos]])
            base = [_q(vals[0, i]) for i in range(d)]
            sel.append(None)
        else:
            if fixed_tokens is not None:
                choices = [fixed_tokens[pos]]
            else:
                choices = TEXT_TOKENS
            vals = np.array([model.E[v] + model.P[pos] for v in choices])
            if token_mode == "bool":
                bs = [z3.Bool(f"s_{pos}_{i}")
                      for i in range(len(TEXT_TOKENS))]
                s.add(z3.PbEq([(b, 1) for b in bs], 1))   # exactly one token
                if fixed_tokens is not None:
                    s.add(bs[TEXT_TOKENS.index(fixed_tokens[pos])])
                base = []
                for i in range(d):
                    expr = _q(model.P[pos, i])
                    for b, v in zip(bs, TEXT_TOKENS):
                        expr = expr + z3.If(b, _q(model.E[v, i]), _q(0))
                    base.append(expr)
                sel.append(bs)
            else:                                          # "hull"
                lm = [z3.Real(f"lam_{pos}_{i}")
                      for i in range(len(TEXT_TOKENS))]
                for lv in lm:
                    s.add(lv >= _q(0.0), lv <= _q(1.0))
                s.add(sum(lm[1:], lm[0]) == _q(1.0))
                if fixed_tokens is not None:
                    s.add(lm[TEXT_TOKENS.index(fixed_tokens[pos])]
                          == _q(1.0))
                base = []
                for i in range(d):
                    expr = _q(model.P[pos, i])
                    for lv, v in zip(lm, TEXT_TOKENS):
                        expr = expr + lv * _q(model.E[v, i])
                    base.append(expr)
                sel.append(lm)
        if fixed_noise is not None:
            lo = vals.min(axis=0) + fixed_noise[pos]
            hi = vals.max(axis=0) + fixed_noise[pos]
        else:
            lo = vals.min(axis=0) - eps
            hi = vals.max(axis=0) + eps
        xs.append([fresh_iv(f"x_{pos}_{i}", base[i] + delta[i],
                            lo[i], hi[i]) for i in range(d)])
        xs_iv.append((np.asarray(lo, float), np.asarray(hi, float)))
    if fixed_tokens is None:
        chosen = [b for b in sel if b is not None]
        if token_mode == "bool":
            seq_class.constrain(s, chosen)
        else:
            seq_class.constrain_hull(s, chosen)

    return _forward_logit(model, s, xs, xs_iv, hints=hints, prefix=""), sel


def prove_transformer(model: TinyTransformer, seq_class: SequenceClass,
                      want: str, eps: float, slack: float = 0.0,
                      timeout_ms: int = 600000, token_mode="hull") -> dict:
    """THE certificate primitive: for ALL sequences in the class (hull
    mode: all fractional token mixtures containing them) and ALL embedding
    noise up to eps, the logit is > slack ('positive') or <= -slack
    ('nonpositive'). 'proved' certifies the discrete claim in either mode
    (hull is a superset region — P1). A hull-mode counterexample may be
    FRACTIONAL: it refutes the relaxation; the discrete claim is then
    undecided by this query ('discrete' False in the counterexample)."""
    L = _tm.L
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    logit, sel = encode_logit(model, s, eps, seq_class,
                              token_mode=token_mode)
    if want == "positive":
        s.add(logit <= _q(slack))
    elif want == "nonpositive":
        s.add(logit > _q(-slack))
    else:
        raise ValueError(want)
    res = s.check()
    if res == z3.unsat:
        return {"proved": True, "status": "proved (no counterexample exists)",
                "counterexample": None}
    if res == z3.sat:
        m = s.model()

        def rat(name):
            v = m.eval(z3.Real(name), model_completion=True)
            return Fraction(int(v.numerator_as_long()),
                            int(v.denominator_as_long()))

        toks, discrete = [], True
        weights = []
        for pos in range(L - 1):
            if token_mode == "bool":
                chosen = [i for i, b in enumerate(sel[pos])
                          if z3.is_true(m.eval(b, model_completion=True))]
                toks.append(TEXT_TOKENS[chosen[0]])
                weights.append(None)
            else:
                lam = [rat(f"lam_{pos}_{i}")
                       for i in range(len(TEXT_TOKENS))]
                weights.append([float(x) for x in lam])
                ints = [i for i, x in enumerate(lam) if x == 1]
                if ints:
                    toks.append(TEXT_TOKENS[ints[0]])
                else:
                    toks.append(None)                     # fractional mix
                    discrete = False
        toks.append(seq_class.task)
        noise = np.zeros((L, model.d))
        for pos in range(L):
            for i in range(model.d):
                noise[pos, i] = float(rat(f"n_{pos}_{i}"))
        status = ("FAILED (solver found a counterexample)" if discrete else
                  "relaxation refuted (FRACTIONAL witness — discrete claim "
                  "undecided by this query)")
        return {"proved": False, "status": status,
                "counterexample": {"tokens": toks, "noise": noise,
                                   "weights": weights, "discrete": discrete}}
    return {"proved": False, "status": f"unknown ({res})",
            "counterexample": None}


def certified_eps(model, seq_class, want, eps_max=0.15, tol=2e-3,
                  slack=0.0, timeout_ms=600000) -> dict:
    """The certified EMBEDDING radius: the largest noise bound for which
    the claim still proves, by bisection (verify.py::certified_radius's
    recipe; same soundness notes)."""
    import time as _time
    t0 = _time.time()
    queries = 0

    def probe(e):
        nonlocal queries
        queries += 1
        return prove_transformer(model, seq_class, want, e, slack=slack,
                                 timeout_ms=timeout_ms)

    base = probe(0.0)
    if not base["proved"]:
        return {"radius": None, "saturated": False, "first_failure": 0.0,
                "counterexample": base["counterexample"],
                "queries": queries, "seconds": round(_time.time() - t0, 1)}
    top = probe(eps_max)
    if top["proved"]:
        return {"radius": eps_max, "saturated": True, "first_failure": None,
                "counterexample": None, "queries": queries,
                "seconds": round(_time.time() - t0, 1)}
    lo, hi, cx = 0.0, eps_max, top["counterexample"]
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        r = probe(mid)
        if r["proved"]:
            lo = mid
        else:
            hi = mid
            cx = r["counterexample"] or cx
    return {"radius": lo, "saturated": False, "first_failure": hi,
            "counterexample": cx, "queries": queries,
            "seconds": round(_time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# The STRENGTHENED removal claim on the transformer: "the skill-A readout no
# longer LISTENS to the quote content" (a two-copy / siamese certificate — the
# transformer analogue of verify.py::prove_independence; paper §V-C).
#
# The ordinary removal certificate says the A-readout logit stays nonpositive
# everywhere. The stronger claim proved here: for ALL pairs of token-hulls that
# AGREE on the task token, the embedding noise, and every NON-quote token weight,
# and are FREE only in how quote mass splits between « and », the A-readout logit
# moves by at most `bound`. At bound = 0 the readout provably ignores the quote
# balance entirely — the exact open/close comparison skill A computes — so no
# arrangement of quotes, sub-threshold or not, can move it.
#
# Built over the hull relaxation (a superset region, so a proof certifies the
# discrete claim by P1) and sharing the validated `_forward_logit`.
# ---------------------------------------------------------------------------
def _skill_tokens(task: int):
    """(open, close) text tokens the skill counts, and the indices into
    TEXT_TOKENS that are FREE (the skill's own tokens) vs SHARED."""
    open_t, close_t = (QO, QC) if task == Q1 else (BO, BC)
    free_idx = [TEXT_TOKENS.index(open_t), TEXT_TOKENS.index(close_t)]
    shared_idx = [i for i in range(len(TEXT_TOKENS)) if i not in free_idx]
    return open_t, close_t, free_idx, shared_idx


def encode_influence_pair(model, s, eps, task, hints=True, pin=None):
    """Build TWO copies of the forward pass over token-hulls that share the
    task token, the embedding noise, and every non-quote token weight, and
    differ only in how the skill's open/close mass splits at each position.
    Returns (logitA, logitB) — the two copies' skill readouts.

    pin = (tokens1, tokens2, noise) fixes both copies to concrete sequences
    (they must agree on the shared tokens and the task token) and a concrete
    noise array — used by the encoding-validation gate."""
    L = _tm.L
    assert model.attn == "gate" and model.gate_hard, \
        "the certified subject is the hardened gate model"
    d = model.d
    t = L - 1
    _open, _close, free_idx, shared_idx = _skill_tokens(task)

    def emb_iv(pos):
        vals = np.array([model.E[v] + model.P[pos] for v in TEXT_TOKENS])
        return vals

    xsA, xsB, xsA_iv, xsB_iv = [], [], [], []
    for pos in range(L):
        # shared noise (same delta enters both copies)
        if pin is not None:
            delta = [_q(pin[2][pos, i]) for i in range(d)]
        elif eps == 0:
            delta = [_q(0.0) for _ in range(d)]
        else:
            delta = [z3.Real(f"n_{pos}_{i}") for i in range(d)]
            for dd in delta:
                s.add(dd >= _q(-eps), dd <= _q(eps))

        if pos == t:                       # readout: fixed task token, shared
            base = [_q(model.E[task, i] + model.P[pos, i]) for i in range(d)]
            baseA = baseB = base
            vals = np.array([model.E[task] + model.P[pos]])
        elif pin is not None:              # concrete tokens for both copies
            baseA = [_q(model.E[pin[0][pos], i] + model.P[pos, i])
                     for i in range(d)]
            baseB = [_q(model.E[pin[1][pos], i] + model.P[pos, i])
                     for i in range(d)]
            vals = emb_iv(pos)
        else:                              # symbolic coupled hull weights
            ws = {i: z3.Real(f"ws_{pos}_{i}") for i in shared_idx}
            fA = {i: z3.Real(f"fa_{pos}_{i}") for i in free_idx}
            fB = {i: z3.Real(f"fb_{pos}_{i}") for i in free_idx}
            for v in list(ws.values()) + list(fA.values()) + list(fB.values()):
                s.add(v >= _q(0.0), v <= _q(1.0))
            shared_sum = sum(ws.values(), _q(0.0))
            s.add(shared_sum + fA[free_idx[0]] + fA[free_idx[1]] == _q(1.0))
            s.add(shared_sum + fB[free_idx[0]] + fB[free_idx[1]] == _q(1.0))
            lamA = {**ws, **fA}
            lamB = {**ws, **fB}
            baseA, baseB = [], []
            for i in range(d):
                ea = _q(model.P[pos, i])
                eb = _q(model.P[pos, i])
                for k in range(len(TEXT_TOKENS)):
                    ea = ea + lamA[k] * _q(model.E[TEXT_TOKENS[k], i])
                    eb = eb + lamB[k] * _q(model.E[TEXT_TOKENS[k], i])
                baseA.append(ea)
                baseB.append(eb)
            vals = emb_iv(pos)

        # interval bounds: the full token hull (or the pinned point) ± noise
        if pin is not None:
            eA = model.E[pin[0][pos]] + model.P[pos] + pin[2][pos]
            eB = model.E[pin[1][pos]] + model.P[pos] + pin[2][pos]
            loA = hiA = eA
            loB = hiB = eB
        else:
            loA = loB = vals.min(axis=0) - eps
            hiA = hiB = vals.max(axis=0) + eps
        xsA.append([_fresh(s, f"a_x_{pos}_{i}", baseA[i] + delta[i])
                    for i in range(d)])
        xsB.append([_fresh(s, f"b_x_{pos}_{i}", baseB[i] + delta[i])
                    for i in range(d)])
        xsA_iv.append((np.asarray(loA, float), np.asarray(hiA, float)))
        xsB_iv.append((np.asarray(loB, float), np.asarray(hiB, float)))

    logitA = _forward_logit(model, s, xsA, xsA_iv, hints=hints, prefix="a_")
    logitB = _forward_logit(model, s, xsB, xsB_iv, hints=hints, prefix="b_")
    return logitA, logitB


def prove_independence_transformer(model, task, eps, bound,
                                   timeout_ms=300000, hints=True) -> dict:
    """Prove: no pair of sequences differing only in quote content moves the
    skill-A readout by more than `bound`, over all embedding noise up to eps.
    unsat = proved (independence at bound=0)."""
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    lA, lB = encode_influence_pair(model, s, eps, task, hints=hints)
    s.add(z3.Or(lA - lB > _q(bound), lA - lB < _q(-bound)))
    res = s.check()
    if res == z3.unsat:
        return {"proved": True, "status": "proved", "gap": None}
    if res == z3.sat:
        return {"proved": False, "status": "refuted (a quote-only change moves "
                "the readout by more than the bound)", "gap": None}
    return {"proved": False, "status": f"unknown ({res})", "gap": None}


def certified_quote_influence(model, task, eps=0.0, tol=1e-2,
                              timeout_ms=300000, hi0=None) -> dict:
    """The certified influence of quote content on the skill-A readout: the
    smallest PROVED ceiling on how far a quote-only change can move the logit,
    anywhere over the region, by bisection of prove_independence_transformer
    (same recipe/guarantees as verify.py::certified_influence — the returned
    ceiling is always genuinely proved; the bracket is tol-tight if no probe
    times out). eps=0 is the clean claim (no noise); eps>0 adds the noise ball.

    Returns {influence, exact_zero, largest_failing_bound, queries, seconds}."""
    import time as _time
    t0 = _time.time()
    queries = 0

    def probe(b):
        nonlocal queries
        queries += 1
        return prove_independence_transformer(model, task, eps, b,
                                               timeout_ms=timeout_ms)

    base = probe(0.0)
    if base["proved"]:
        return {"influence": 0.0, "exact_zero": True,
                "largest_failing_bound": None, "queries": queries,
                "seconds": round(_time.time() - t0, 1)}

    # Grow an upper bound until it PROVES (a proved bound is a real ceiling).
    hi = hi0 if hi0 is not None else 1.0
    top = probe(hi)
    grows = 0
    while not top["proved"] and top["status"].startswith("refuted") \
            and grows < 12:
        hi *= 2.0
        top = probe(hi)
        grows += 1
    if not top["proved"]:
        return {"influence": None, "exact_zero": False,
                "largest_failing_bound": hi, "queries": queries,
                "seconds": round(_time.time() - t0, 1)}

    lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        r = probe(mid)
        if r["proved"]:
            hi = mid
        else:
            lo = mid
    return {"influence": hi, "exact_zero": False,
            "largest_failing_bound": lo, "queries": queries,
            "seconds": round(_time.time() - t0, 1)}


def validate_influence_encoding(model, task, n=12, eps=0.03, seed=17) -> float:
    """Gate: the two-copy encoding must equal the float forward at concrete
    pinned pairs (sequences agreeing on shared tokens + task, differing in
    quote content) before any independence proof is trusted."""
    L = _tm.L
    _open, _close, free_idx, shared_idx = _skill_tokens(task)
    shared_toks = [TEXT_TOKENS[i] for i in shared_idx]
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        base = rng.choice(TEXT_TOKENS, size=L - 1)
        t1 = base.copy()
        t2 = base.copy()
        # at each position, independently pick a quote token for each copy
        for pos in range(L - 1):
            t1[pos] = rng.choice([_open, _close])
            t2[pos] = rng.choice([_open, _close])
        toks1 = np.append(t1, task)
        toks2 = np.append(t2, task)
        noise = rng.uniform(-eps, eps, size=(L, model.d))
        s = z3.Solver()
        lA, lB = encode_influence_pair(model, s, eps, task,
                                       pin=(toks1, toks2, noise))
        assert s.check() == z3.sat
        m = s.model()

        def val(x):
            v = m.eval(x, model_completion=True)
            return float(Fraction(int(v.numerator_as_long()),
                                  int(v.denominator_as_long())))
        gotA, gotB = val(lA), val(lB)
        wantA = float(model.forward(toks1[None], noise=noise[None])[0])
        wantB = float(model.forward(toks2[None], noise=noise[None])[0])
        worst = max(worst, abs(gotA - wantA), abs(gotB - wantB))
    return worst


# ---------------------------------------------------------------------------
# The two numeric cross-checks (the grid_check analogues).
# ---------------------------------------------------------------------------
def all_sequences(task: int) -> np.ndarray:
    """EVERY length-(L-1) text over the alphabet, task token appended —
    5^7 = 78,125 sequences: token space is small enough to enumerate, so
    the clean (no-noise) part of every claim gets brute-forced numerically.
    The noise dimension is what only the solver can cover."""
    L = _tm.L
    grids = np.meshgrid(*[TEXT_TOKENS] * (L - 1), indexing="ij")
    toks = np.stack([g.ravel() for g in grids], axis=1)
    return np.concatenate([toks, np.full((len(toks), 1), task)], axis=1)


def exhaustive_clean_check(model, seq_class, want) -> dict:
    toks = all_sequences(seq_class.task)
    toks = toks[seq_class.numeric_mask(toks)]
    vals = model.forward(toks)
    bad = (vals <= 0) if want == "positive" else (vals > 0)
    return {"sequences": int(len(toks)), "violations": int(bad.sum())}


def sampled_noise_check(model, seq_class, want, eps, n=2000, seed=0) -> dict:
    """Random (sequence, noise) samples inside the claim's region — the
    float-side sanity companion (float-gap tolerance conventions apply)."""
    L = _tm.L
    rng = np.random.default_rng(seed)
    toks = all_sequences(seq_class.task)
    toks = toks[seq_class.numeric_mask(toks)]
    idx = rng.integers(0, len(toks), n)
    noise = rng.uniform(-eps, eps, size=(n, L, model.d))
    vals = np.array([model.forward(toks[i:i + 1], noise=noise[j:j + 1])[0]
                     for j, i in enumerate(idx)])
    bad = (vals <= 0) if want == "positive" else (vals > 0)
    return {"samples": n, "violations": int(bad.sum())}
