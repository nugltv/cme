"""
sae_model.py — a sparse autoencoder over the toy model's hidden layer, and an
               EXACTLY-CERTIFIABLE feature-clamp edit (paper §V-E)
=============================================================================

WHY THIS EXISTS
---------------
Sparse autoencoders (SAEs) are the edit surface practitioners most associate
with steering and unlearning: train an overcomplete sparse dictionary on a
layer's activations, find the feature that stands for a concept, and "clamp" it
(force it to zero, or to a fixed value) to remove or inject that concept. The
other experiments certify ablation / weight-edit / steering; this adds the SAE
feature-clamp as a fourth edit type — and, crucially, one that is still EXACTLY
verifiable, because an SAE is piecewise-linear:

    encode:  f = ReLU(We @ a + be)          (a = the model's hidden activations)
    decode:  a_hat = Wd @ f + bd

A feature-clamp that zeroes feature k applies the edit as the standard additive
correction (so it is a no-op wherever the feature is already off, and cancels
the SAE's reconstruction error):

    a_edited = a - f_k * Wd[:, k],   f_k = ReLU(We[k] . a + be[k])

Both ReLUs and all the affine maps encode exactly in Z3, so removal and
preservation of the edited network are provable over a whole input region, the
same way ablation is. So the feature clamp is not outside the
method — it is one more piecewise-linear edit, and we certify it.

Everything here is NumPy + (for the proof) the exact-fraction Z3 helpers from
verify.py. The SAE is trained on the frozen toy model's hidden activations.
"""

from __future__ import annotations

import numpy as np
import z3

from verify import _q


# ---------------------------------------------------------------------------
# The sparse autoencoder
# ---------------------------------------------------------------------------
class SAE:
    """A tiny ReLU sparse autoencoder over an H-dim activation space with F
    overcomplete features. Weights are plain NumPy arrays so verify.py can
    convert them to exact fractions."""

    def __init__(self, H: int, F: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.H, self.F = H, F
        # He-ish init for the encoder; decoder tied-ish but free.
        self.We = rng.normal(0.0, 1.0 / np.sqrt(H), size=(F, H))
        self.be = np.zeros(F)
        self.Wd = rng.normal(0.0, 1.0 / np.sqrt(F), size=(H, F))
        self.bd = np.zeros(H)

    def encode(self, a: np.ndarray) -> np.ndarray:
        return np.maximum(a @ self.We.T + self.be, 0.0)

    def decode(self, f: np.ndarray) -> np.ndarray:
        return f @ self.Wd.T + self.bd

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(a))


def train_sae(acts: np.ndarray, F: int, l1: float = 4e-3, steps: int = 4000,
              lr: float = 0.02, seed: int = 0, verbose: bool = False) -> SAE:
    """Fit an SAE to reconstruct the activation matrix `acts` (N x H) with an L1
    sparsity penalty on the features. Plain full-batch gradient descent — the
    activation cloud is tiny (a few thousand 16-dim vectors), so this is fast
    and deterministic. Returns the trained SAE."""
    N, H = acts.shape
    sae = SAE(H, F, seed=seed)
    A = acts
    for t in range(steps):
        # forward
        pre = A @ sae.We.T + sae.be            # (N, F)
        f = np.maximum(pre, 0.0)
        recon = f @ sae.Wd.T + sae.bd          # (N, H)
        err = recon - A                        # (N, H)
        # losses: 0.5*MSE + l1*|f|
        # gradients
        drecon = err / N                       # d(0.5*mse)/d recon
        dWd = drecon.T @ f                      # (H, F)
        dbd = drecon.sum(axis=0)               # (H,)
        df = drecon @ sae.Wd                    # (N, F)
        df = df + (l1 / N) * (f > 0)           # L1 on active features
        dpre = df * (pre > 0)                   # through ReLU
        dWe = dpre.T @ A                        # (F, H)
        dbe = dpre.sum(axis=0)                  # (F,)
        # step
        sae.Wd -= lr * dWd
        sae.bd -= lr * dbd
        sae.We -= lr * dWe
        sae.be -= lr * dbe
        if verbose and t % 1000 == 0:
            mse = float((err ** 2).mean())
            spars = float((f > 0).mean())
            print(f"    sae step {t}: mse {mse:.5f}, active-frac {spars:.3f}")
    return sae


def skill_feature_scores(sae: SAE, acts: np.ndarray,
                         skill_on: np.ndarray) -> np.ndarray:
    """Per-feature separation score = mean(feature | skill on) - mean(feature |
    skill off). The features a practitioner would clamp to remove the skill are
    the ones with the largest scores."""
    f = sae.encode(acts)                       # (N, F)
    return f[skill_on].mean(axis=0) - f[~skill_on].mean(axis=0)


def feature_for_skill(sae: SAE, acts: np.ndarray, skill_on: np.ndarray) -> int:
    """The single best-separating feature (kept for callers that want one)."""
    return int(np.argmax(skill_feature_scores(sae, acts, skill_on)))


def features_for_skill(sae: SAE, acts: np.ndarray, skill_on: np.ndarray,
                       n: int = 1) -> list[int]:
    """The top-`n` features for a skill by separation score — the feature SET a
    practitioner clamps for SAE-based unlearning (concepts usually 'split'
    across several features, so a single clamp under-removes)."""
    scores = skill_feature_scores(sae, acts, skill_on)
    return sorted(np.argsort(scores)[::-1][:n].tolist())


# ---------------------------------------------------------------------------
# The feature-clamp edit — numeric (for grids) and exact (for Z3)
# ---------------------------------------------------------------------------
def _as_list(features) -> list[int]:
    return [int(features)] if np.isscalar(features) else [int(k) for k in features]


def clamp_forward(model, sae: SAE, features, x: np.ndarray) -> np.ndarray:
    """The toy model's forward pass with the given SAE feature(s) clamped to
    zero, applied as the additive correction a_edited = a - sum_k f_k * Wd[:,k].
    `features` is an int or a list of feature indices. Returns the (N, 2) logits.
    This is the NumPy twin of build_logits_sae_clamp below — the grid cross-check
    runs this while Z3 reasons about the exact-fraction version."""
    K = _as_list(features)
    z = x @ model.W1.T + model.b1
    a = np.maximum(z, 0.0)                      # (N, H)
    a_edited = a.copy()
    for k in K:
        fk = np.maximum(a @ sae.We[k] + sae.be[k], 0.0)      # (N,)
        a_edited = a_edited - fk[:, None] * sae.Wd[:, k][None, :]
    return a_edited @ model.W2.T + model.b2


def build_logits_sae_clamp(model, sae: SAE, features):
    """Return a `logits_fn(model, xs) -> (logit_A, logit_B)` closure that encodes
    the toy model WITH the given SAE feature(s) clamped to zero, exactly, for
    verify.prove_forall / verify.certified_radius (pass it as logits_fn=...).
    `features` is an int or a list of indices (a concept usually splits across
    several features, so the set form is the realistic one).

    Every step is ReLU-of-affine or affine, so the whole edited network is
    encoded exactly in Z3 rationals — the clamp is not an approximation."""
    K = _as_list(features)
    H = model.H
    d = model.W1.shape[1]

    def logits_fn(_model, xs):
        assert len(xs) == d, f"model expects {d} inputs, got {len(xs)}"
        # hidden activations a[j] = ReLU(W1 x + b1)
        a = []
        for j in range(H):
            z = _q(model.b1[j])
            for i in range(d):
                z = z + _q(model.W1[j, i]) * xs[i]
            a.append(z3.If(z > 0, z, z3.RealVal(0)))
        # clamp each feature: a_edited[j] = a[j] - sum_k f_k * Wd[j, k]
        a_ed = list(a)
        for k in K:
            pre = _q(sae.be[k])
            for j in range(H):
                pre = pre + _q(sae.We[k, j]) * a[j]
            fk = z3.If(pre > 0, pre, z3.RealVal(0))
            a_ed = [a_ed[j] - fk * _q(sae.Wd[j, k]) for j in range(H)]
        logit_A = _q(model.b2[0])
        logit_B = _q(model.b2[1])
        for j in range(H):
            logit_A = logit_A + _q(model.W2[0, j]) * a_ed[j]
            logit_B = logit_B + _q(model.W2[1, j]) * a_ed[j]
        return logit_A, logit_B

    return logits_fn
