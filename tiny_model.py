"""
tiny_model.py
=============

A very small neural network with TWO separate skills, plus the code to train it.

Why this file exists
--------------------
The paper ("Certified Mechanistic Edits", CME) is about PROVING what
happens when you edit a model: if we switch off the part that does skill A, can we
*prove* that skill A is gone everywhere, while skill B still works everywhere?

To prove things we need the smallest possible model where we already know the
"right answer". So this model:

  * takes two numbers as input:            x = (x0, x1),  each between 0 and 1
  * has ONE hidden layer of ReLU neurons   (ReLU = "keep positive numbers, turn
                                             negatives into 0" — the simplest
                                             non-linear building block)
  * produces TWO outputs (we call them "heads"):
        - head A  should answer: "is x0 greater than 0.5?"   (skill A)
        - head B  should answer: "is x1 greater than 0.5?"   (skill B)

The two heads read from the SAME hidden neurons. That shared layer is what makes
the experiment interesting: when we later switch off "skill A's neurons", it is
not obvious that skill B survives — because they were sharing the same parts.
Proving B survives is therefore a real result, not a triviality.

Everything here is plain NumPy (no PyTorch), so it runs anywhere and is easy to read.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# 1. The "ground truth": what the two skills are SUPPOSED to answer.
#    label 1 means "yes / high", label 0 means "no / low".
# ---------------------------------------------------------------------------
def true_label_A(x: np.ndarray) -> np.ndarray:
    """Skill A's correct answer: 1 if the first coordinate x0 > 0.5, else 0."""
    return (x[..., 0] > 0.5).astype(np.float64)


def true_label_B(x: np.ndarray) -> np.ndarray:
    """Skill B's correct answer: 1 if the second coordinate x1 > 0.5, else 0."""
    return (x[..., 1] > 0.5).astype(np.float64)


# ---------------------------------------------------------------------------
# 2. The model itself.
# ---------------------------------------------------------------------------
class TinyMLP:
    """
    A 2 -> H -> 2 network:
        inputs (2)  ->  hidden layer of H ReLU neurons  ->  2 output logits.

    A "logit" is just the raw output number before we turn it into a yes/no.
    Decision rule: logit > 0  means "yes/high" (label 1); logit <= 0 means "no/low".

    The weights are plain NumPy arrays:
        W1 (H x 2), b1 (H)   : hidden layer
        W2 (2 x H), b2 (2)   : the two output heads. Row 0 = head A, row 1 = head B.
    """

    def __init__(self, H: int = 16, seed: int = 0, d: int = 2):
        """
        H : number of hidden neurons.
        d : number of INPUT numbers. The default 2 is the base toy model
            (x0, x1). Higher d (4, 5, ...) adds extra input dimensions that the
            two skills IGNORE — the labels still depend only on x0 and x1 — but
            the network doesn't know that, so its weights can (and do) touch the
            extra inputs. Those stray connections are exactly where hidden
            leftover pathways can lurk after an edit, which is what the
            higher-dimensional illusion search goes looking for.
        """
        rng = np.random.default_rng(seed)
        # Small random starting weights. Training will shape them.
        self.W1 = rng.normal(0.0, 1.0, size=(H, d))
        self.b1 = np.zeros(H)
        self.W2 = rng.normal(0.0, 1.0, size=(2, H))
        self.b2 = np.zeros(2)
        self.H = H
        self.d = d

    # ---- the forward pass: input numbers -> output logits ----
    def forward(self, x: np.ndarray, ablate: list[int] | None = None) -> np.ndarray:
        """
        Run inputs through the network.

        x       : array of shape (N, 2) — N input points.
        ablate  : optional list of hidden-neuron indices to SWITCH OFF
                  (force their output to 0). This is our "edit"/"intervention".
                  With ablate=None we get the original, unedited model.

        Returns : array of shape (N, 2) — [logit_A, logit_B] for each input.
        """
        z = x @ self.W1.T + self.b1          # hidden pre-activations, shape (N, H)
        a = np.maximum(z, 0.0)               # ReLU: negatives become 0
        if ablate:
            a = a.copy()
            a[:, ablate] = 0.0               # switch off the chosen neurons
        logits = a @ self.W2.T + self.b2     # shape (N, 2)
        return logits

    def hidden_activations(self, x: np.ndarray) -> np.ndarray:
        """Return just the hidden-neuron outputs (used for inspecting the 'circuit')."""
        return np.maximum(x @ self.W1.T + self.b1, 0.0)


# ---------------------------------------------------------------------------
# 3. Training (plain gradient descent, written out by hand so it's transparent).
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train(model: TinyMLP, steps: int = 8000, lr: float = 0.15,
          n_train: int = 6000, seed: int = 1, l1: float = 8e-3,
          verbose: bool = True) -> TinyMLP:
    """
    Teach the model both skills at once.

    We generate random points in the unit square, compute the correct A and B
    labels, and nudge the weights to reduce the error (standard logistic-
    regression-style training, but done manually so every step is visible).

    l1 : a small "tidiness" penalty that pushes unused weights to exactly zero.
         This encourages each hidden neuron to specialise to ONE skill, which
         makes the later "switch off skill A's neurons" experiment clean. With
         l1=8e-3 the trained model puts skill A in a single neuron and skill B
         in another, with no neurons shared between them. Set l1=0 to see the
         messier, more realistic "skills share neurons" regime instead.
    """
    rng = np.random.default_rng(seed)
    # Training points have as many coordinates as the model has inputs. The
    # labels only ever look at the first two coordinates; any extra ones are
    # "distractor" inputs the network must learn to (mostly) ignore.
    X = rng.uniform(0.0, 1.0, size=(n_train, model.W1.shape[1]))
    yA = true_label_A(X)
    yB = true_label_B(X)
    Y = np.stack([yA, yB], axis=1)           # shape (N, 2)

    for t in range(steps):
        # ----- forward -----
        z = X @ model.W1.T + model.b1        # (N, H)
        a = np.maximum(z, 0.0)               # (N, H)
        logits = a @ model.W2.T + model.b2   # (N, 2)
        p = _sigmoid(logits)                 # predicted probabilities (N, 2)

        # ----- backward (gradients of the binary cross-entropy loss) -----
        dlogits = (p - Y) / X.shape[0]       # (N, 2)
        dW2 = dlogits.T @ a + l1 * np.sign(model.W2)   # +L1 tidiness penalty
        db2 = dlogits.sum(axis=0)            # (2,)
        da = dlogits @ model.W2              # (N, H)
        dz = da * (z > 0.0)                  # ReLU passes gradient only where active
        dW1 = dz.T @ X + l1 * np.sign(model.W1)        # +L1 tidiness penalty
        db1 = dz.sum(axis=0)                 # (H,)

        # ----- update -----
        model.W2 -= lr * dW2
        model.b2 -= lr * db2
        model.W1 -= lr * dW1
        model.b1 -= lr * db1

        if verbose and (t % 2000 == 0 or t == steps - 1):
            acc = accuracy(model, X, Y)
            print(f"  step {t:5d}  loss~{bce(p, Y):.4f}  train-acc A={acc[0]:.3f} B={acc[1]:.3f}")

    return model


def find_skill_circuit(model: TinyMLP, skill: str, seed: int = 7,
                       n: int = 8000, hurt: float = 0.02, spare: float = 0.01):
    """
    Find the hidden neurons that make up one skill's "circuit" — the standard
    interpretability move of knocking out each neuron and seeing which ones matter.

    A neuron is counted as part of skill A's circuit if switching it off by itself
    HURTS skill A's accuracy by more than `hurt`, while changing skill B's accuracy
    by less than `spare` (so we target neurons that serve A, not B). Same idea for B.

    Returns the list of neuron indices in that skill's circuit.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, model.W1.shape[1]))
    Y = np.stack([true_label_A(X), true_label_B(X)], axis=1)
    base = accuracy(model, X, Y)
    mine, other = (0, 1) if skill == "A" else (1, 0)
    circuit = []
    for j in range(model.H):
        acc_j = accuracy(model, X, Y, ablate=[j])
        drop_mine = base[mine] - acc_j[mine]
        drop_other = base[other] - acc_j[other]
        if drop_mine > hurt and abs(drop_other) < spare:
            circuit.append(j)
    return circuit


def bce(p, Y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return float(-(Y * np.log(p) + (1 - Y) * np.log(1 - p)).mean())


def accuracy(model: TinyMLP, X: np.ndarray, Y: np.ndarray, ablate=None):
    """Fraction of points where each head's yes/no decision is correct."""
    logits = model.forward(X, ablate=ablate)
    pred = (logits > 0.0).astype(np.float64)
    return (pred == Y).mean(axis=0)          # [accuracy_A, accuracy_B]


if __name__ == "__main__":
    # Quick self-test: train and print accuracy on a fresh grid.
    m = train(TinyMLP(H=16, seed=0), verbose=True)
    rng = np.random.default_rng(123)
    Xt = rng.uniform(0, 1, size=(5000, 2))
    Yt = np.stack([true_label_A(Xt), true_label_B(Xt)], axis=1)
    print("test accuracy [A, B]:", accuracy(m, Xt, Yt))
