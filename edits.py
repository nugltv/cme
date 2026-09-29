"""
edits.py — the different KINDS of mechanistic edit, all as weight changes
=========================================================================

The base slice (run_slice.py) uses one kind of edit: switching neurons off
(ablation). The paper compares SEVERAL kinds, because a key question is: which
kinds of edit can be certified, and how strongly?  This file implements three:

  1. ABLATION      — switch circuit neurons off entirely (as run_slice.py does).
  2. WEIGHT EDIT   — a more surgical version: cut only the WIRES from the
                     circuit neurons to head A. The neurons stay alive and can
                     still serve head B; head A just can't hear them anymore.
  3. STEERING      — the "nudge" style of edit used in practice on large
                     models: add a fixed push to the model's internal numbers
                     so the unwanted behavior (here: head A saying HIGH)
                     stops happening. Unlike ablation, steering is a tug of
                     war: the push must beat the inputs' own signal, and a
                     strong enough input might still win — which is exactly
                     the kind of failure a proof can catch and a test can miss.

THE ONE TRICK THAT KEEPS EVERYTHING SIMPLE
------------------------------------------
Each edit is expressed as a change to the model's WEIGHTS, producing a new
TinyMLP. That means the proving machinery in verify.py needs no changes at
all: an edited model is just another model, and we prove things about it the
same way. (For the record: switching neuron j off is the same as setting its
incoming weights and offset to zero — then it always outputs ReLU(0) = 0.
Steering by v is the same as adding v to the neurons' offsets b1.)
"""

from __future__ import annotations
import numpy as np

from tiny_model import TinyMLP


def _clone(model: TinyMLP) -> TinyMLP:
    """An independent copy of the model (so edits never touch the original)."""
    m = TinyMLP(H=model.H, seed=0, d=model.W1.shape[1])
    m.W1 = model.W1.copy()
    m.b1 = model.b1.copy()
    m.W2 = model.W2.copy()
    m.b2 = model.b2.copy()
    return m


# ---------------------------------------------------------------------------
# 1. Ablation: switch the circuit neurons off completely.
# ---------------------------------------------------------------------------
def apply_ablation(model: TinyMLP, neurons: list[int]) -> TinyMLP:
    """
    Force each chosen neuron to output 0 for every input, by zeroing its
    incoming weights and offset. Identical in effect to the `ablate=` argument
    used in run_slice.py — but expressed as a weight change, so the edited model can
    be handed to the prover as-is. Affects BOTH heads (the neuron is dead for
    everyone), which is ablation's characteristic bluntness.
    """
    m = _clone(model)
    for j in neurons:
        m.W1[j, :] = 0.0
        m.b1[j] = 0.0
    return m


# ---------------------------------------------------------------------------
# 2. Weight edit: cut only the wires from the circuit neurons to head A.
# ---------------------------------------------------------------------------
def apply_weight_edit(model: TinyMLP, neurons: list[int], head: int = 0) -> TinyMLP:
    """
    Zero the OUTGOING weights from the chosen neurons to one head (default:
    head A). The neurons keep firing and keep serving the other head — this is
    the "precision surgery" version of removal, in the spirit of ROME/MEMIT
    weight edits on large models. On a perfectly tidy model (each neuron
    serves one skill) this is indistinguishable from ablation; on a messy
    model it should cause LESS collateral damage — a difference our
    preservation certificates can measure.
    """
    m = _clone(model)
    for j in neurons:
        m.W2[head, j] = 0.0
    return m


# ---------------------------------------------------------------------------
# 3. Steering: add a fixed push to the neurons' pre-activation values.
# ---------------------------------------------------------------------------
def apply_steering(model: TinyMLP, vector: np.ndarray) -> TinyMLP:
    """
    Add a fixed vector (one entry per hidden neuron) to every neuron's
    pre-activation value, for every input. A negative entry pushes that neuron
    toward silence; a positive one toward firing. Implemented by shifting the
    offsets b1 — mathematically identical, and it keeps the edited model an
    ordinary TinyMLP.
    """
    m = _clone(model)
    m.b1 = m.b1 + np.asarray(vector, dtype=float)
    return m


def targeted_suppression_vector(model: TinyMLP, neurons: list[int],
                                strength: float) -> np.ndarray:
    """
    The simplest steering recipe: push ONLY the circuit neurons down by
    `strength`, leave everyone else alone. With a big enough strength the
    circuit neurons stay silent for every input in range and the skill is
    gone; with a too-small strength, strong inputs can still overcome the push
    — the classic way a steering edit quietly under-delivers.
    """
    v = np.zeros(model.H)
    for j in neurons:
        v[j] = -strength
    return v


def diff_of_means_vector(model: TinyMLP, strength: float, seed: int = 11,
                         n: int = 4000) -> np.ndarray:
    """
    The realistic steering recipe (how it's actually done on large models,
    "representation engineering" style): collect the neurons' pre-activation
    values on inputs where the skill SHOULD fire (x0 high) and where it
    shouldn't (x0 low), take the difference of the two averages, and push the
    model along "low minus high" — i.e. toward its own 'the skill is off'
    internal state. Nobody chooses which neurons to touch; the statistics do.
    That makes it easy to apply and inherently imprecise — the vector can
    brush against neurons that serve the OTHER skill, which is exactly the
    collateral damage our preservation certificates put a number on.
    """
    d = model.W1.shape[1]
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, d))
    z = X @ model.W1.T + model.b1                  # pre-activations, (n, H)
    high = z[X[:, 0] > 0.5].mean(axis=0)           # avg when skill A active
    low = z[X[:, 0] <= 0.5].mean(axis=0)           # avg when skill A inactive
    direction = low - high
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return strength * direction
