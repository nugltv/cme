"""
mod_arith_model.py — the modular-adder subject: a two-skill modular-arithmetic
                     transformer (paper Table II, "Adder transformer")
===============================================================================

Same exactly-encodable pieces as the threshold-gate subject
(`transformer_model.py`) — gate attention with additive LINEAR scores, LeakyReLU
MLP, quantized weights — but with a p-WAY readout, because the task is a formula
with p possible answers, not a yes/no. The correct behavior is a KNOWN FORMULA,
so "skill removed" = "no longer computes the formula" is crisp and checkable.

TWO SKILLS, ONE TRUNK, SELECTED BY A TASK TOKEN
-----------------------------------------------
Sequence = [a, b, TASK], length L = 3. Positions 0, 1 hold digit tokens
a, b in {0..p-1}; the LAST position holds a task token that says which formula
to compute, and is where the p-way answer is read (the CLS pattern, exactly like
the threshold-gate subject):

  * TASK = ADD -> skill A: output (a + b) mod p     (the skill to REMOVE)
  * TASK = SUB -> skill B: output (a - b) mod p      (the skill to PRESERVE)

Both use the same digit embeddings and the same trunk, so removing A while
provably preserving B is a genuine surgery question on shared machinery — and
because the whole input space is p*p*2 sequences, every claim is checkable
exhaustively as a cross-check.

Everything is plain NumPy with hand-written backprop (numerically gradient-checked
in `gradient_check`), like every other model in this repo. Edits (head / MLP
ablation) are weight-independent forward-time knockouts, the same primitives the
threshold-gate circuit search uses.
"""

from __future__ import annotations
import numpy as np

L = 3                      # sequence length (set per task; see set_len)
ADD_OFF = 0                # task-token offsets past the p digit tokens
SUB_OFF = 1


def set_len(n):
    """Set the sequence length (the two tasks below use different lengths)."""
    global L
    L = n


def set_prime(p, skillB="sub", mode="twoop"):
    """Vocab layout for prime p: tokens 0..p-1 are digits, p+ADD_OFF and
    p+SUB_OFF are the two task tokens. `mode` selects the task family:

      'twoop'  — [a, b, TASK] (L=3): both skills read the SAME two digits, and
                 differ only by OPERATION. ADD -> (a+b) mod p; skill B per
                 `skillB` ('sub'/'mul'/'copa'/'dbl'). The skills are ENTANGLED
                 (shared digit-reading trunk) — kept for the record; not editable.
      'twoadd' — [a1, b1, a2, b2, TASK] (L=5): two INDEPENDENT modular adders on
                 DISJOINT positions. ADD -> (a1+b1) mod p (reads positions 0,1);
                 SUB-slot -> (a2+b2) mod p (reads positions 2,3). Disjoint inputs
                 give the threshold-gate task's separability with arithmetic —
                 ablating one adder should leave the other intact. This is the
                 editable subject.

    Callers must `set_len(3)` for twoop, `set_len(5)` for twoadd."""
    return {"p": p, "V": p + 2, "ADD": p + ADD_OFF, "SUB": p + SUB_OFF,
            "skillB": skillB, "mode": mode}


def leaky(x, a=0.1):
    return np.where(x >= 0, x, a * x)


def dleaky(x, a=0.1):
    return np.where(x >= 0, 1.0, a)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Ground truth and data.
# ---------------------------------------------------------------------------
def all_sequences(cfg):
    """Every sequence for the task family: p*p pairs x 2 tasks (twoop), or
    p^4 quadruples x 2 tasks (twoadd)."""
    p = cfg["p"]
    if cfg["mode"] == "twoop":
        a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
        digs = np.stack([a.ravel(), b.ravel()], axis=1)
    else:                                   # twoadd: 4 digit positions
        grids = np.meshgrid(*[np.arange(p)] * 4, indexing="ij")
        digs = np.stack([g.ravel() for g in grids], axis=1)
    add = np.concatenate([digs, np.full((len(digs), 1), cfg["ADD"])], axis=1)
    sub = np.concatenate([digs, np.full((len(digs), 1), cfg["SUB"])], axis=1)
    return np.concatenate([add, sub], axis=0)


def true_label(cfg, tokens):
    """The answer owed. twoop: (a+b) mod p under ADD, skill B's formula else.
    twoadd: (a1+b1) mod p under ADD (positions 0,1), (a2+b2) mod p under the
    other task (positions 2,3)."""
    p = cfg["p"]
    task = tokens[:, -1]
    if cfg["mode"] == "twoop":
        a, b = tokens[:, 0], tokens[:, 1]
        add = (a + b) % p
        skB = {"sub": (a - b) % p, "mul": (a * b) % p, "copa": a % p,
               "dbl": (2 * a) % p}[cfg["skillB"]]
        return np.where(task == cfg["ADD"], add, skB)
    sum1 = (tokens[:, 0] + tokens[:, 1]) % p
    sum2 = (tokens[:, 2] + tokens[:, 3]) % p
    return np.where(task == cfg["ADD"], sum1, sum2)


# ---------------------------------------------------------------------------
# The model.
# ---------------------------------------------------------------------------
class ModTransformer:
    """One decoder block, p-way readout. Gate attention (additive LINEAR
    scores), LeakyReLU MLP, exactly the exact-encodable pieces of the
    threshold-gate subject. gate_hard switches training surrogate -> exact 0/1 gates."""

    def __init__(self, cfg, d_model=8, n_heads=2, d_head=4, d_mlp=16, seed=0,
                 alpha=0.1, gate_T=0.5):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.cfg = cfg
        self.p, self.V = cfg["p"], cfg["V"]
        self.d, self.H, self.dh, self.m = d_model, n_heads, d_head, d_mlp
        self.alpha, self.gate_T = alpha, gate_T
        self.gate_hard = False
        # tags read by verify_transformer's shared encoder (this model always
        # uses hardened gate attention with linear additive scores)
        self.attn = "gate"
        self.score_act = "linear"
        self.E = rng.normal(0, 1.0, (self.V, d_model)) * s * 1.5
        self.P = rng.normal(0, 1.0, (L, d_model)) * s * 1.5
        self.Wq = rng.normal(0, s * 0.3, (n_heads, d_head, d_model))
        self.bq = np.zeros((n_heads, d_head))
        self.Wk = rng.normal(0, s * 0.3, (n_heads, d_head, d_model))
        self.bk = np.zeros((n_heads, d_head))
        self.u = rng.normal(0, 0.3, (n_heads, d_head))
        self.Wv = rng.normal(0, s, (n_heads, d_head, d_model))
        self.Wo = rng.normal(0, s, (d_model, n_heads * d_head))
        self.bo = np.zeros(d_model)
        self.W1 = rng.normal(0, s, (d_mlp, d_model))
        self.b1 = np.zeros(d_mlp)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(d_mlp), (d_model, d_mlp))
        self.b2 = np.zeros(d_model)
        self.R = rng.normal(0, 1.0 / np.sqrt(d_model), (self.p, d_model))
        self.c = np.zeros(self.p)

    def names(self):
        return ["E", "P", "Wq", "bq", "Wk", "bk", "u", "Wv", "Wo", "bo",
                "W1", "b1", "W2", "b2", "R", "c"]

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def forward(self, tokens, noise=None, ablate_heads=(), ablate_mlp=(),
                cache=None):
        """p-way logits (N, p) at the readout (last) position."""
        x = self.E[tokens] + self.P[None, :, :]
        if noise is not None:
            x = x + noise
        t = L - 1
        heads = []
        for h in range(self.H):
            q_t = x[:, t] @ self.Wq[h].T + self.bq[h]
            k = x @ self.Wk[h].T + self.bk[h]
            pre = q_t[:, None, :] + k                    # linear additive score
            scores = pre @ self.u[h]
            if self.gate_hard:
                a = (scores > 0).astype(float)
            else:
                a = _sigmoid(scores / self.gate_T)
            v = x @ self.Wv[h].T
            o = (a[:, :, None] * v).sum(axis=1)
            if h in ablate_heads:
                o = np.zeros_like(o)
            heads.append((q_t, k, pre, scores, a, v, o))
        o_cat = np.concatenate([hd[6] for hd in heads], axis=1)
        x1 = x[:, t] + o_cat @ self.Wo.T + self.bo
        z1 = x1 @ self.W1.T + self.b1
        a1 = leaky(z1, self.alpha)
        if ablate_mlp:
            a1 = a1.copy()
            a1[:, list(ablate_mlp)] = 0.0
        x2 = x1 + a1 @ self.W2.T + self.b2
        logits = x2 @ self.R.T + self.c
        if cache is not None:
            cache.update(x=x, heads=heads, o_cat=o_cat, x1=x1, z1=z1,
                         a1=a1, x2=x2)
        return logits

    def predict(self, tokens, **kw):
        return self.forward(tokens, **kw).argmax(axis=1)

    def harden(self):
        self.gate_hard = True
        return self

    def quantize(self, bits=12):
        scale = float(2 ** bits)
        for name in self.names():
            v = getattr(self, name)
            setattr(self, name, np.round(np.asarray(v) * scale) / scale)
        return self


def ablate_head_weights(model, heads):
    """Head ablation as a WEIGHT change (so a verifier encoding the weights sees
    the edit): zero the output-projection columns that read each head's output.
    Equivalent to forward(ablate_heads=heads) — the head contributes nothing to
    the residual — but baked in, the way the threshold-gate edits are (edits.py)."""
    m = model.clone()
    for h in heads:
        m.Wo[:, h * m.dh:(h + 1) * m.dh] = 0.0
    return m


def weight_edit_head(model, heads):
    """Weight-edit of an attention HEAD. For a head there is no incoming/outgoing
    distinction the way there is for an MLP neuron: the only weight change that
    stops the head from reaching the residual is zeroing its output-projection
    columns — which is exactly `ablate_head_weights`. So for a head-only circuit
    ablation and weight-edit COINCIDE; this alias exists so the edit-type table
    can name the row and record that fact honestly."""
    return ablate_head_weights(model, heads)


def steer_residual(model, vector):
    """Steering as a weight change: add a fixed vector to the post-attention
    residual (bo is exactly that hook), matching edits.py / transformer_model.py.
    Lives in the residual stream, so it can shift the MLP's operating point but
    has no handle on which summands the attention head reads."""
    m = model.clone()
    m.bo = m.bo + np.asarray(vector, dtype=float)
    return m


def summand_steer_direction(model, cfg):
    """A realistic activation-steering direction for the summand skill: the axis
    of the post-attention readout residual along which the representation varies
    MOST with the summands under task A — the dominant principal component of x1
    over the skill-A sequences. This is the direction a practitioner's steering
    vector points along; steering pushes along its negative to try to erase the
    summand signal. (A diff-of-means between two summand classes can cancel to
    zero by symmetry; the top principal component is the robust, non-degenerate
    version of the same idea.)

    No choice of residual offset can actually make the p-way output constant over
    the summands: a constant added to x1 leaves x1's summand variation intact on
    the linear skip path, and the MLP nonlinearity cannot cancel it at every sum.
    The table measures exactly how far this realistic recipe gets."""
    toks = all_sequences(cfg)
    add = toks[toks[:, -1] == cfg["ADD"]]
    cache = {}
    model.forward(add, cache=cache)
    x1 = cache["x1"]
    xc = x1 - x1.mean(axis=0, keepdims=True)     # center over the summand grid
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    d = vt[0]                                     # top principal axis of variation
    return d / (np.linalg.norm(d) + 1e-12)


# ---------------------------------------------------------------------------
# Training (softmax cross-entropy, full manual backprop; gradient-checked).
# ---------------------------------------------------------------------------
def grads(model: ModTransformer, tokens, y):
    cache = {}
    logits = model.forward(tokens, cache=cache)
    N = tokens.shape[0]
    P = softmax(logits)
    loss = -np.log(P[np.arange(N), y] + 1e-12).mean()
    dlogits = P.copy()
    dlogits[np.arange(N), y] -= 1.0
    dlogits /= N

    g = {k: np.zeros_like(np.asarray(getattr(model, k), dtype=float))
         for k in model.names()}
    x, o_cat, x1, z1, a1, x2 = (cache[j] for j in
                                ("x", "o_cat", "x1", "z1", "a1", "x2"))
    t = L - 1

    g["R"] = dlogits.T @ x2
    g["c"] = dlogits.sum(axis=0)
    dx2 = dlogits @ model.R

    g["W2"] = dx2.T @ a1
    g["b2"] = dx2.sum(axis=0)
    da1 = dx2 @ model.W2
    dz1 = da1 * dleaky(z1, model.alpha)
    g["W1"] = dz1.T @ x1
    g["b1"] = dz1.sum(axis=0)
    dx1 = dx2 + dz1 @ model.W1

    g["Wo"] = dx1.T @ o_cat
    g["bo"] = dx1.sum(axis=0)
    do_cat = dx1 @ model.Wo
    dx = np.zeros_like(x)
    dx[:, t] += dx1

    for h in range(model.H):
        q_t, k, pre, scores, a, v, o = cache["heads"][h]
        do = do_cat[:, h * model.dh:(h + 1) * model.dh]
        da = (do[:, None, :] * v).sum(axis=2)
        dv = a[:, :, None] * do[:, None, :]
        g["Wv"][h] = np.einsum("nld,nle->de", dv, x)
        dx += dv @ model.Wv[h]
        asig = _sigmoid(scores / model.gate_T)
        dscores = da * asig * (1 - asig) / model.gate_T
        g["u"][h] = np.einsum("nl,nld->d", dscores, pre)
        dpre = dscores[:, :, None] * model.u[h][None, None, :]
        dq_t = dpre.sum(axis=1)
        dk = dpre
        g["Wq"][h] = dq_t.T @ x[:, t]
        g["bq"][h] = dq_t.sum(axis=0)
        dx[:, t] += dq_t @ model.Wq[h]
        g["Wk"][h] = np.einsum("nld,nle->de", dk, x)
        g["bk"][h] = dk.sum(axis=(0, 1))
        dx += dk @ model.Wk[h]

    np.add.at(g["E"], tokens.reshape(-1), dx.reshape(-1, model.d))
    g["P"] = dx.sum(axis=0)
    return loss, g


def gradient_check(p=5, seed=3):
    cfg = set_prime(p)
    m = ModTransformer(cfg, seed=seed)
    toks = all_sequences(cfg)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(toks), 10)
    toks, y = toks[idx], true_label(cfg, toks[idx])
    _, g = grads(m, toks, y)
    worst, eps = 0.0, 1e-6
    for name in ["Wq", "bq", "u", "Wv", "Wo", "W1", "W2", "R", "E", "P", "c"]:
        flat = np.asarray(getattr(m, name)).ravel()
        for i in {0, len(flat) // 2, len(flat) - 1}:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = grads(m, toks, y)
            flat[i] = orig - eps
            lm, _ = grads(m, toks, y)
            flat[i] = orig
            worst = max(worst, abs((lp - lm) / (2 * eps)
                                   - np.asarray(g[name]).ravel()[i]))
    return worst


def train(model: ModTransformer, steps=12000, lr=0.05, wd=1e-4, seed=1,
          verbose=False):
    cfg = model.cfg
    toks = all_sequences(cfg)
    y = true_label(cfg, toks)
    vel = {k: np.zeros_like(np.asarray(getattr(model, k), dtype=float))
           for k in model.names()}
    for step in range(steps):
        loss, g = grads(model, toks, y)
        for k in model.names():
            gk = g[k] + wd * np.asarray(getattr(model, k))
            vel[k] = 0.9 * vel[k] - lr * gk
            setattr(model, k, np.asarray(getattr(model, k)) + vel[k])
        if verbose and step % 2000 == 0:
            print(f"    step {step}: loss {loss:.4f}, "
                  f"acc {(model.predict(toks) == y).mean():.3f}")
    return model


def skill_accuracy(model: ModTransformer, **kw):
    """Per-skill accuracy over the whole input space (ADD rows, SUB rows)."""
    cfg = model.cfg
    toks = all_sequences(cfg)
    y = true_label(cfg, toks)
    pred = model.predict(toks, **kw)
    is_add = toks[:, -1] == cfg["ADD"]         # task token is the last position
    accA = float((pred[is_add] == y[is_add]).mean())
    accB = float((pred[~is_add] == y[~is_add]).mean())
    return accA, accB


def train_subject(cfg, d_model=8, n_heads=2, seed=0, verbose=False):
    """Full build: train the surrogate, harden the gates, fine-tune, quantize —
    the certified artifact (mirrors train_gate_subject for the threshold-gate
    subject)."""
    m = ModTransformer(cfg, d_model=d_model, n_heads=n_heads, seed=seed)
    train(m, steps=12000, lr=0.05, seed=seed, verbose=verbose)
    m.harden()
    train(m, steps=3000, lr=0.01, seed=seed + 50, verbose=verbose)
    m.quantize(bits=12)
    return m


if __name__ == "__main__":
    print("gradient check:", f"{gradient_check():.2e}")
    cfg = set_prime(5)
    m = train_subject(cfg, d_model=8, seed=0, verbose=True)
    print("per-skill accuracy (hardened+quantized):", skill_accuracy(m))
