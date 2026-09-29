"""
deep_model.py — a DEEPER model with a skill no single neuron can hold
=====================================================================

Why this exists: every experiment so far used a one-hidden-layer network
whose skill A ("is x0 > 0.5?") can — and, tidily trained, does — live in a
single neuron. Maybe certified edits only work on toys like that. This file
provides the harder subject: a network with TWO hidden
layers and a skill that provably CANNOT sit in one neuron, because it is a
composition:

    skill A:  "is exactly one of x0, x1 above 0.5?"   (an XOR of thresholds)
    skill B:  "is x2 above 0.5?"                       (as before, but on a
                                                        third input)

XOR is the classic example of a function a single ReLU neuron cannot
represent: the layer-1 neurons must detect the pieces (x0 high, x1 high, and
their overlap) and a later stage must COMBINE them. So skill A's circuit is
necessarily spread across neurons — and, in trained models, usually across
both layers. Editing it is closer to the real-world situation where a
capability lives in a distributed pathway.

The second half of the answer is run_deep.py, which runs the full pipeline
(find circuit -> edit -> prove removal + preservation -> certified radii per
edit type) on this model.

Everything here mirrors tiny_model.py deliberately: NumPy only, fixed seeds,
generous comments. The verifier needs one addition (a multi-layer encoder in
verify.py); every EDIT is expressed as a weight change, exactly like
edits.py does for the shallow model, so the prover needs nothing else.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# The two skills (ground truth labels).
# ---------------------------------------------------------------------------
def deep_label_A(x: np.ndarray) -> np.ndarray:
    """Skill A: 1 if EXACTLY ONE of x0, x1 is above 0.5 (XOR), else 0."""
    return ((x[:, 0] > 0.5) ^ (x[:, 1] > 0.5)).astype(float)


def deep_label_B(x: np.ndarray) -> np.ndarray:
    """Skill B: 1 if x2 is above 0.5, else 0."""
    return (x[:, 2] > 0.5).astype(float)


class DeepMLP:
    """
    A 3 -> H1 -> H2 -> 2 network:
        inputs (3) -> hidden layer 1 (H1 ReLU neurons)
                   -> hidden layer 2 (H2 ReLU neurons) -> 2 output logits.

    Decision rule as everywhere else: logit > 0 means "yes/high".

    Weights (plain NumPy arrays):
        hidden_weights = [W1 (H1 x 3), W2 (H2 x H1)]
        hidden_biases  = [b1 (H1),     b2 (H2)]
        W_out (2 x H2), b_out (2)      : the two heads (row 0 = A, row 1 = B)

    verify.py recognises a model as "deep" by the `hidden_weights`
    attribute and encodes every layer exactly (ReLU is still piecewise
    linear, so nothing about the proving story changes with depth).
    """

    def __init__(self, H1: int = 12, H2: int = 8, seed: int = 0, d: int = 3):
        rng = np.random.default_rng(seed)
        # He-style scaling keeps early training healthy for XOR.
        self.hidden_weights = [
            rng.normal(0.0, np.sqrt(2.0 / d), size=(H1, d)),
            rng.normal(0.0, np.sqrt(2.0 / H1), size=(H2, H1)),
        ]
        self.hidden_biases = [np.zeros(H1), np.zeros(H2)]
        self.W_out = rng.normal(0.0, np.sqrt(2.0 / H2), size=(2, H2))
        self.b_out = np.zeros(2)
        self.d = d

    @property
    def layer_sizes(self) -> list[int]:
        return [W.shape[0] for W in self.hidden_weights]

    def forward(self, x: np.ndarray,
                ablate: set[tuple[int, int]] | None = None) -> np.ndarray:
        """
        Inputs -> logits. `ablate` is a set of (layer, neuron) pairs to
        switch off numerically — used only by the circuit SEARCH; the edits
        we actually certify are weight changes (see ablate_deep below).
        Layers are numbered 1 and 2 to match how people talk about them.
        """
        ablate = ablate or set()
        a = x
        for li, (W, b) in enumerate(zip(self.hidden_weights,
                                        self.hidden_biases), start=1):
            a = np.maximum(a @ W.T + b, 0.0)
            for (layer, j) in ablate:
                if layer == li:
                    a[:, j] = 0.0
        return a @ self.W_out.T + self.b_out

    def clone(self) -> "DeepMLP":
        m = DeepMLP(seed=0, d=self.d,
                    H1=self.layer_sizes[0], H2=self.layer_sizes[1])
        m.hidden_weights = [W.copy() for W in self.hidden_weights]
        m.hidden_biases = [b.copy() for b in self.hidden_biases]
        m.W_out = self.W_out.copy()
        m.b_out = self.b_out.copy()
        return m


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_deep(model: DeepMLP, steps: int = 20000, lr: float = 0.2,
               n_train: int = 8000, seed: int = 1,
               verbose: bool = True) -> DeepMLP:
    """
    Teach both skills at once — plain backpropagation through the two
    hidden layers, written out so every step is visible. No tidiness (L1)
    penalty here: we WANT the realistic, entangled regime; XOR forces
    distribution anyway.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_train, model.d))
    Y = np.stack([deep_label_A(X), deep_label_B(X)], axis=1)
    W1, W2 = model.hidden_weights
    b1, b2 = model.hidden_biases

    for t in range(steps):
        # ----- forward, keeping every intermediate for the backward pass ----
        z1 = X @ W1.T + b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ W2.T + b2
        a2 = np.maximum(z2, 0.0)
        logits = a2 @ model.W_out.T + model.b_out
        p = _sigmoid(logits)

        # ----- backward (binary cross-entropy on both heads) ----------------
        dlogits = (p - Y) / X.shape[0]
        dW_out = dlogits.T @ a2
        db_out = dlogits.sum(axis=0)
        da2 = dlogits @ model.W_out
        dz2 = da2 * (z2 > 0.0)
        dW2 = dz2.T @ a1
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2
        dz1 = da1 * (z1 > 0.0)
        dW1 = dz1.T @ X
        db1 = dz1.sum(axis=0)

        # ----- update --------------------------------------------------------
        model.W_out -= lr * dW_out
        model.b_out -= lr * db_out
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        if verbose and (t % 4000 == 0 or t == steps - 1):
            acc = deep_accuracy(model, X, Y)
            print(f"  step {t:5d}  train-acc A={acc[0]:.3f} B={acc[1]:.3f}")
    return model


def deep_accuracy(model: DeepMLP, X: np.ndarray, Y: np.ndarray,
                  ablate=None) -> tuple[float, float]:
    logits = model.forward(X, ablate=ablate)
    pred = (logits > 0).astype(float)
    return float((pred[:, 0] == Y[:, 0]).mean()), \
        float((pred[:, 1] == Y[:, 1]).mean())


# ---------------------------------------------------------------------------
# Edits on the deep model, all as weight changes (the edits.py trick).
# ---------------------------------------------------------------------------
def ablate_deep(model: DeepMLP, neurons: list[tuple[int, int]]) -> DeepMLP:
    """Switch off the given (layer, neuron) pairs: zero their incoming
    weights and bias, so they output ReLU(0) = 0 for every input."""
    m = model.clone()
    for (layer, j) in neurons:
        m.hidden_weights[layer - 1][j, :] = 0.0
        m.hidden_biases[layer - 1][j] = 0.0
    return m


def weight_edit_deep(model: DeepMLP, neurons: list[tuple[int, int]],
                     head: int = 0) -> DeepMLP:
    """
    Cut only the wires from LAYER-2 circuit neurons to one head. (For a
    layer-1 neuron there is no single wire to a head — its influence flows
    through all of layer 2 — so this surgical edit is only defined for the
    layer-2 members of a circuit; callers should check the circuit first.)
    """
    m = model.clone()
    for (layer, j) in neurons:
        if layer != 2:
            raise ValueError("weight edit reaches only layer-2 neurons; "
                             f"got layer {layer}")
        m.W_out[head, j] = 0.0
    return m


def steer_deep(model: DeepMLP, layer: int, vector: np.ndarray) -> DeepMLP:
    """Add a fixed vector to one hidden layer's pre-activations (a bias
    shift) — steering, exactly as in edits.py but with a layer choice."""
    m = model.clone()
    m.hidden_biases[layer - 1] = m.hidden_biases[layer - 1] + \
        np.asarray(vector, dtype=float)
    return m


def targeted_vector_deep(model: DeepMLP, layer: int,
                         neurons: list[int], strength: float) -> np.ndarray:
    """Push only the chosen neurons of one layer down by `strength`."""
    v = np.zeros(model.layer_sizes[layer - 1])
    for j in neurons:
        v[j] = -strength
    return v


def diff_of_means_deep(model: DeepMLP, layer: int, strength: float,
                       seed: int = 11, n: int = 6000) -> np.ndarray:
    """
    The realistic steering recipe at a chosen layer: average that layer's
    pre-activations over inputs where skill A should be OFF and where it
    should be ON, push along "off minus on". Statistics pick the direction;
    nobody chooses neurons.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, model.d))
    a = X
    pre = None
    for li, (W, b) in enumerate(zip(model.hidden_weights,
                                    model.hidden_biases), start=1):
        pre = a @ W.T + b
        a = np.maximum(pre, 0.0)
        if li == layer:
            break
    on = deep_label_A(X) == 1.0
    direction = pre[~on].mean(axis=0) - pre[on].mean(axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return strength * direction
