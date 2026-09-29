"""
verify.py
=========

This is where the "proof" happens. It turns the little network into exact maths
that a solver (Z3) can reason about, and then asks the solver yes/no questions
that amount to PROOFS about EVERY input at once (not just tested samples).

The key mental model
--------------------
Testing a model means: try some inputs, look at the answers. If you didn't try a
bad input, you won't know it exists.

Proving with a solver means: ask "does there EXIST any input in this whole region
where the property is violated?" If the solver answers "no, that's impossible"
(technically: "unsatisfiable"), then the property holds for the ENTIRE region —
including the infinitely many inputs you never tested. If instead the solver says
"yes", it hands you a concrete counterexample input.

So each of our "certificates" is really one solver question of the form:
    "Is there an input in this box where the edited model does the WRONG thing?"
    - answer NO  -> certificate PROVED
    - answer YES -> certificate FAILED, and here is the offending input.

Exactness / soundness note — and exactly WHAT the proofs are about
------------------------------------------------------------------
The network's weights are ordinary floating-point numbers. We feed them to Z3
as EXACT fractions (every float IS an exact fraction), so Z3 reasons with no
rounding of its own. The certificates are therefore statements about the
IDEAL network: the true real-number function those weights define.

That is not quite the same thing as the program that runs: `model.forward`
computes in float64 and picks up rounding noise of order 1e-16 per step. For
a claim with real margin (our claims keep margins around 1e-1; the trained
models' logits clear zero by at least ~1e-4 everywhere in their regions) the
noise cannot flip any verdict, and the proofs carry over to the executed
program — provable via the `slack` option of `prove_forall` below: prove
"logit stays at least gamma clear of zero" for a gamma far above the float
noise, and the float program's answer is pinned too. But a claim with ZERO
real margin does not carry over: a model can sit exactly ON zero across a
region (proved <= 0 by Z3, correctly) while float rounding makes the executed
forward() wobble to either side. `run_float_gap.py` demonstrates exactly
that, and the paper's float-gap appendix tells the story in plain
words. The practical rule: the solver and the float grid may legitimately
disagree WITHIN float noise of zero; disagreement beyond that means a bug.
"""

from __future__ import annotations
from fractions import Fraction
import numpy as np
import z3

from tiny_model import TinyMLP


# ---------------------------------------------------------------------------
# Helper: turn a Python/NumPy float into an EXACT Z3 rational number.
# ---------------------------------------------------------------------------
def _q(value) -> z3.ArithRef:
    fr = Fraction(float(value))          # exact fraction equal to the float64 value
    return z3.RealVal(fr.numerator) / z3.RealVal(fr.denominator)


# ---------------------------------------------------------------------------
# Build the network's two output logits as SYMBOLIC Z3 expressions.
# ---------------------------------------------------------------------------
def build_logits(model, xs: list[z3.ArithRef],
                 ablate: list[int] | None = None):
    """
    Given the symbolic inputs xs (one Z3 variable per input coordinate — two
    for the original model, more for the higher-dimensional ones), return
    symbolic expressions (logit_A, logit_B).

    'ablate' is the edit: hidden neurons in that list are forced to output 0,
    exactly like TinyMLP.forward(..., ablate=...) does numerically.

    Models with MORE THAN ONE hidden layer (DeepMLP in deep_model.py, marked
    by a `hidden_weights` attribute) are dispatched to the multi-layer
    encoder below — depth changes nothing about exactness, since every layer
    is still ReLU-of-affine.
    """
    if hasattr(model, "hidden_weights"):
        return _build_logits_deep(model, xs, ablate)
    ablate = set(ablate or [])
    H = model.H
    d = model.W1.shape[1]
    assert len(xs) == d, f"model expects {d} inputs, got {len(xs)} variables"

    a = []  # symbolic hidden-neuron outputs
    for j in range(H):
        if j in ablate:
            a.append(z3.RealVal(0))                      # switched-off neuron
            continue
        z = _q(model.b1[j])
        for i in range(d):
            z = z + _q(model.W1[j, i]) * xs[i]
        # ReLU as an exact piecewise expression: max(z, 0)
        a.append(z3.If(z > 0, z, z3.RealVal(0)))

    logit_A = _q(model.b2[0])
    logit_B = _q(model.b2[1])
    for j in range(H):
        logit_A = logit_A + _q(model.W2[0, j]) * a[j]
        logit_B = logit_B + _q(model.W2[1, j]) * a[j]
    return logit_A, logit_B


def _build_logits_deep(model, xs: list[z3.ArithRef], ablate=None):
    """
    The same exact encoding for a model with several hidden layers: walk the
    layers, building each neuron's output as ReLU(affine combination of the
    previous layer's symbolic outputs). Edits to deep models are expressed
    as weight changes (see deep_model.py), so no ablate parameter is
    supported here — pass the edited model instead.
    """
    assert not ablate, "deep-model edits are weight changes; pass the " \
        "edited model rather than an ablate list"
    assert len(xs) == model.d, \
        f"model expects {model.d} inputs, got {len(xs)} variables"

    acts = list(xs)
    for W, b in zip(model.hidden_weights, model.hidden_biases):
        nxt = []
        for j in range(W.shape[0]):
            z = _q(b[j])
            for i in range(W.shape[1]):
                z = z + _q(W[j, i]) * acts[i]
            nxt.append(z3.If(z > 0, z, z3.RealVal(0)))   # exact ReLU
        acts = nxt

    logit_A = _q(model.b_out[0])
    logit_B = _q(model.b_out[1])
    for j in range(len(acts)):
        logit_A = logit_A + _q(model.W_out[0, j]) * acts[j]
        logit_B = logit_B + _q(model.W_out[1, j]) * acts[j]
    return logit_A, logit_B


def _normalize_box(box):
    """
    Accept the box in either of two formats and return a list of (low, high)
    pairs, one per input coordinate:
      * the original 2-input flat form:  (lo0, hi0, lo1, hi1)
      * the general form:                [(lo0, hi0), (lo1, hi1), ...]
    """
    if len(box) == 4 and not hasattr(box[0], "__len__"):
        return [(box[0], box[1]), (box[2], box[3])]
    return [tuple(pair) for pair in box]


# ---------------------------------------------------------------------------
# The one primitive every certificate is built from:
#   "prove that, for all inputs in this box, <condition on the logit> holds."
# We prove it by asking the solver for a COUNTEREXAMPLE and hoping there is none.
# ---------------------------------------------------------------------------
def prove_forall(model, box, logit_selector, want, ablate=None, timeout_ms=20000,
                 slack=0.0, logits_fn=None):
    """
    box            : the region of inputs to reason over. Either the original
                     flat 2-input form (lo0, hi0, lo1, hi1) or a list of
                     (low, high) pairs, one per input coordinate.
    logit_selector : "A" or "B" — which head we are making a claim about.
    want           : "positive" -> claim: logit > 0 for ALL inputs in the box
                     "nonpositive" -> claim: logit <= 0 for ALL inputs in the box
    ablate         : the edit (list of neurons to switch off), or None.
    slack          : demand the logit clears zero by at least this much
                     (claim becomes "logit > slack" / "logit <= -slack").
                     Default 0 = the ordinary claims. A positive slack larger
                     than the float evaluation error makes the certificate
                     carry over from the ideal network to the float program
                     that actually runs (see run_float_gap.py).

    Returns a dict: {proved: bool, status: str,
                     counterexample: tuple of input coords, or None}
    """
    bounds = _normalize_box(box)
    xs = [z3.Real(f"x{i}") for i in range(len(bounds))]

    # logits_fn lets a caller supply a DIFFERENT exact encoding of the edited
    # network (e.g. an SAE feature-clamp, see sae_model.build_logits_sae_clamp)
    # while reusing all of the counterexample-search machinery below. When it is
    # None we use the built-in ablation encoding.
    if logits_fn is not None:
        logit_A, logit_B = logits_fn(model, xs)
    else:
        logit_A, logit_B = build_logits(model, xs, ablate=ablate)
    logit = logit_A if logit_selector == "A" else logit_B

    s = z3.Solver()
    s.set("timeout", timeout_ms)
    # input must lie inside the box
    for xi, (lo, hi) in zip(xs, bounds):
        s.add(xi >= _q(lo), xi <= _q(hi))
    # add the NEGATION of what we want to prove (i.e. look for a violation)
    if want == "positive":
        s.add(logit <= _q(slack))    # a point where "logit > slack" fails
    elif want == "nonpositive":
        s.add(logit > _q(-slack))    # a point where "logit <= -slack" fails
    else:
        raise ValueError(want)

    result = s.check()
    if result == z3.unsat:
        # No violating input exists anywhere in the box -> the claim is PROVED.
        return {"proved": True, "status": "proved (no counterexample exists)",
                "counterexample": None}
    if result == z3.sat:
        m = s.model()
        cx = tuple(_model_to_float(m, xi) for xi in xs)
        return {"proved": False, "status": "FAILED (solver found a counterexample)",
                "counterexample": cx}
    return {"proved": False, "status": f"unknown ({result}) — try longer timeout",
            "counterexample": None}


def _model_to_float(m, var) -> float:
    v = m.eval(var, model_completion=True)
    # v is an exact rational; convert to float for human-readable printing
    return float(Fraction(int(v.numerator_as_long()), int(v.denominator_as_long())))


# ---------------------------------------------------------------------------
# "Wiggle room" (robustness of an edit's effect) and the certified radius
# (paper Definition 4, Proposition 2).
#
# The idea in plain words: the base certificate proves "the skill is gone for
# every input in the approved region". The stronger question an attacker forces:
# "and if someone NUDGES an input — moves every coordinate by up to epsilon,
# including toward the ambiguous zone the region deliberately excluded — is the
# skill STILL gone?" Geometrically that means growing the region outward by
# epsilon on every side (never beyond the model's legal input range) and
# proving the same claim over the bigger region. The CERTIFIED RADIUS is the
# largest epsilon for which the proof still succeeds: a single number that
# says how much nudging the edit provably withstands. Bigger radius = edit
# with less attack surface. Comparing this number across kinds of edit
# (ablation vs steering vs weight edit) is one of the paper's results (§V-E).
# ---------------------------------------------------------------------------
def inflate_box(box, eps, domain=(0.0, 1.0)):
    """
    Grow the region outward by eps on every side, clipped to the legal input
    range. box is a list of (low, high) pairs (or the legacy flat 2-input
    form); domain is the legal range every input coordinate must stay inside.
    """
    dlo, dhi = domain
    return [(max(dlo, lo - eps), min(dhi, hi + eps))
            for lo, hi in _normalize_box(box)]


def certified_radius(model, box, logit_selector, want, ablate=None,
                     domain=(0.0, 1.0), eps_max=1.0, tol=1e-3,
                     timeout_ms=30000, logits_fn=None):
    """
    Find the largest wiggle room epsilon for which the claim still PROVES over
    the inflated region. Works by bisection: each probe is one solver call
    ("does the claim hold over the region grown by this epsilon?"), and we
    home in on the break-point to within tol.

    Returns a dict:
      radius     : largest epsilon that proved (None if even eps=0 fails —
                   i.e. the base claim itself is false)
      saturated  : True if the claim proved at eps_max, meaning inflation hit
                   the edge of the legal input range everywhere and there is
                   no room left to grow — the strongest possible outcome
      first_failure : the smallest epsilon observed to fail (None if saturated)
      counterexample_at_failure : the solver's violating input there, if any
      queries, seconds : cost bookkeeping (each query is one proof attempt)
    """
    import time as _time
    t0 = _time.time()
    queries = 0

    def probe(eps):
        nonlocal queries
        queries += 1
        return prove_forall(model, inflate_box(box, eps, domain),
                            logit_selector, want, ablate=ablate,
                            timeout_ms=timeout_ms, logits_fn=logits_fn)

    # Step 1: the base claim (no wiggle room). If this fails, there is no
    # radius to speak of — the edit's effect doesn't even hold unwiggled.
    base = probe(0.0)
    if not base["proved"]:
        return {"radius": None, "saturated": False, "first_failure": 0.0,
                "counterexample_at_failure": base["counterexample"],
                "queries": queries, "seconds": round(_time.time() - t0, 2)}

    # Step 2: try the maximum. If the claim holds with the region grown to
    # the full legal range, no attack within the range can ever succeed.
    top = probe(eps_max)
    if top["proved"]:
        return {"radius": eps_max, "saturated": True, "first_failure": None,
                "counterexample_at_failure": None,
                "queries": queries, "seconds": round(_time.time() - t0, 2)}

    # Step 3: bisect between "proves" (lo) and "fails" (hi) until the gap is
    # smaller than tol. ~10 solver calls for tol=1e-3.
    lo, hi = 0.0, eps_max
    cx_at_failure = top["counterexample"]
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        r = probe(mid)
        if r["proved"]:
            lo = mid
        else:
            hi = mid
            if r["counterexample"] is not None:
                cx_at_failure = r["counterexample"]
    return {"radius": lo, "saturated": False, "first_failure": hi,
            "counterexample_at_failure": cx_at_failure,
            "queries": queries, "seconds": round(_time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# The STRENGTHENED removal claim: "this head no longer LISTENS to these
# inputs" (a two-copy / "siamese" certificate).
#
# The ordinary removal certificate says: the head's output stays on the low
# side everywhere in a region. That can be satisfied by a head that merely sits below
# zero while still READING the skill's input — a leftover pathway at
# sub-threshold strength, exactly the raw material of the intervention
# illusion. The stronger claim proved here: for ALL pairs of inputs that
# differ ONLY in the chosen coordinates, the head's output moves by at most
# `bound`. At bound = 0 that is literal independence — the head provably
# ignores those inputs everywhere in the box, so no value of them can ever
# matter, sub-threshold or not.
#
# Encoding: build the network TWICE over two input vectors constrained to
# agree on every coordinate except the free ones, and ask the solver for a
# pair where the outputs differ by more than the bound. unsat = proved.
# ---------------------------------------------------------------------------
def prove_independence(model, box, logit_selector, free_coords, ablate=None,
                       bound=0.0, timeout_ms=60000):
    """
    box            : the region both copies live in (same formats as
                     prove_forall).
    logit_selector : "A" or "B" — the head the claim is about.
    free_coords    : input coordinates (indices) allowed to differ between
                     the two copies — the inputs the head must ignore.
    bound          : the claim is |logit(x) - logit(y)| <= bound for all
                     such pairs; 0 means the head ignores them exactly.

    Returns {proved, status, counterexample_pair: (x, y) or None,
             logit_gap: float or None (numeric check at the pair)}.
    """
    bounds = _normalize_box(box)
    free = set(free_coords)
    xs = [z3.Real(f"x{i}") for i in range(len(bounds))]
    ys = [z3.Real(f"y{i}") for i in range(len(bounds))]

    lA1, lB1 = build_logits(model, xs, ablate=ablate)
    lA2, lB2 = build_logits(model, ys, ablate=ablate)
    l1 = lA1 if logit_selector == "A" else lB1
    l2 = lA2 if logit_selector == "A" else lB2

    s = z3.Solver()
    s.set("timeout", timeout_ms)
    for i, (lo, hi) in enumerate(bounds):
        s.add(xs[i] >= _q(lo), xs[i] <= _q(hi))
        s.add(ys[i] >= _q(lo), ys[i] <= _q(hi))
        if i not in free:
            s.add(xs[i] == ys[i])          # copies agree except on free coords
    diff = l1 - l2
    # negation of the claim: some pair moves the head by MORE than the bound
    s.add(z3.Or(diff > _q(bound), diff < _q(-bound)))

    result = s.check()
    if result == z3.unsat:
        return {"proved": True,
                "status": "proved (no pair of inputs differing only in the "
                          "free coordinates moves the head by more than the "
                          "bound)",
                "counterexample_pair": None, "logit_gap": None}
    if result == z3.sat:
        m = s.model()
        x = tuple(_model_to_float(m, v) for v in xs)
        y = tuple(_model_to_float(m, v) for v in ys)
        col = 0 if logit_selector == "A" else 1
        pts = np.array([x, y])
        vals = model.forward(pts, ablate=ablate)[:, col]
        return {"proved": False,
                "status": "FAILED (solver found a pair the head tells apart)",
                "counterexample_pair": (x, y),
                "logit_gap": float(vals[0] - vals[1])}
    return {"proved": False, "status": f"unknown ({result}) — try longer "
            "timeout", "counterexample_pair": None, "logit_gap": None}


def logit_magnitude_cap(model, logit_selector) -> float:
    """A coarse but rigorous ceiling on |logit| anywhere in the unit box
    (weights' worst case), used to start the influence bisection. One
    hidden layer only — matches what prove_independence is used on."""
    col = 0 if logit_selector == "A" else 1
    mag_a = np.maximum(np.abs(model.b1) + np.abs(model.W1).sum(axis=1), 0.0)
    return float(abs(model.b2[col]) + np.abs(model.W2[col]) @ mag_a)


def certified_influence(model, box, logit_selector, free_coords, ablate=None,
                        tol=1e-3, timeout_ms=60000):
    """
    The certified INFLUENCE of the free coordinates on the head: the
    smallest provable ceiling on how much changing only those inputs can
    move the head's logit, anywhere in the box. Found by bisection of
    prove_independence — the same recipe as certified_radius, so the same
    guarantees (the returned ceiling is always genuinely PROVED; the
    bracket is tol-tight if no probe times out).

    Returns {influence: proved ceiling (0.0 if exactly independent),
             exact_zero: bool, largest_failing_bound: the biggest bound the
             solver REFUTED (so the true influence lies between that and the
             ceiling), counterexample_pair, queries, seconds}.
    """
    import time as _time
    t0 = _time.time()
    queries = 0

    def probe(b):
        nonlocal queries
        queries += 1
        return prove_independence(model, box, logit_selector, free_coords,
                                  ablate=ablate, bound=b,
                                  timeout_ms=timeout_ms)

    base = probe(0.0)
    if base["proved"]:
        return {"influence": 0.0, "exact_zero": True,
                "largest_failing_bound": None,
                "counterexample_pair": None, "queries": queries,
                "seconds": round(_time.time() - t0, 2)}

    # A ceiling that cannot fail mathematically: the logit itself is capped,
    # so the gap between two logits is capped at twice that.
    hi = 2.0 * logit_magnitude_cap(model, logit_selector)
    top = probe(hi)
    if not top["proved"]:                  # only a timeout can cause this
        return {"influence": None, "exact_zero": False,
                "largest_failing_bound": 0.0,
                "counterexample_pair": base["counterexample_pair"],
                "queries": queries, "seconds": round(_time.time() - t0, 2)}

    lo, cx = 0.0, base["counterexample_pair"]
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        r = probe(mid)
        if r["proved"]:
            hi = mid
        else:
            lo = mid
            if r["counterexample_pair"] is not None:
                cx = r["counterexample_pair"]
    return {"influence": hi, "exact_zero": False,
            "largest_failing_bound": lo,
            "counterexample_pair": cx, "queries": queries,
            "seconds": round(_time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# Independent double-check: brute-force a fine grid with plain NumPy.
# If the solver says "proved for all inputs", the grid had better agree on the
# points it happens to sample. (The grid can MISS violations between its points;
# the solver cannot. That gap is exactly why we use a solver.)
#
# One honest wrinkle (see the module docstring and run_float_gap.py): the grid
# runs the FLOAT program while the solver reasons about the IDEAL network, so
# on a claim whose true margin is zero the two can legitimately disagree by
# float rounding noise (~1e-16). The `tol` parameter ignores violations that
# small. Default 0 keeps the strict behavior — fine for all our
# trained models, whose margins are ~12 orders of magnitude above the noise.
# The convention is therefore: grid/solver disagreement BEYOND tol means a
# bug; disagreement within float noise means the claim has zero real margin.
# ---------------------------------------------------------------------------
def grid_check(model, box, logit_selector, want, ablate=None, n=201, tol=0.0):
    lo0, hi0, lo1, hi1 = box
    xs0 = np.linspace(lo0, hi0, n)
    xs1 = np.linspace(lo1, hi1, n)
    G0, G1 = np.meshgrid(xs0, xs1)
    pts = np.stack([G0.ravel(), G1.ravel()], axis=1)
    logits = model.forward(pts, ablate=ablate)
    col = 0 if logit_selector == "A" else 1
    vals = logits[:, col]
    if want == "positive":
        violations = int((vals <= -tol).sum())
    else:
        violations = int((vals > tol).sum())
    return {"grid_points": pts.shape[0], "violations_found": violations}
