"""
transformer_model.py — a tiny, solver-friendly transformer with two skills
===========================================================================

The threshold-gate subject (the measured frontier: results/rung2_size_ladder.log,
paper Table III). A GPT-style
decoder block, sized for exact verification, built ONLY from pieces that
encode exactly into linear real arithmetic:

  * attention that is piecewise-linear: sparsemax (the trainable reference)
    or THRESHOLD-GATE attention (exact 0/1 gates; the certified form), with
    ADDITIVE (Bahdanau-style) scores — score = u . LeakyReLU(q + k) —
    instead of dot products. Dot-product scores are bilinear in the
    embeddings, which would push symbolic input-noise queries into
    nonlinear arithmetic (measured ~15x slower at micro scale). Additive scoring is a classical attention form
    and stays piecewise-linear end to end.
  * LeakyReLU everywhere a nonlinearity is needed (exact If-encoding).
  * no LayerNorm stand-in at all: at one block and d_model 16 training
    converges without it, and every skipped op is solver work saved.

THE TASK (two skills, one trunk)
--------------------------------
Inputs are token sequences of length L (8 by default; the certified subject
uses L = 6, set by run_transformer.py). Positions 0..L-2 are text over
the alphabet {a, «, », (, )}; the LAST position carries a TASK TOKEN:

  * task token Q1 — skill A: "is there an unclosed quote?"
    (more « than » in the text)
  * task token Q2 — skill B: "is there an unclosed bracket?"
    (more ( than ) in the text)

One shared trunk, one output logit read at the last position — which is
the task token's position, so the attention QUERY that gathers evidence is
conditioned on which question is being asked (the CLS-token pattern). Both skills are
GLOBAL properties of the sequence (counting opens vs closes across all
positions) — no input coordinate IS the label. Removing skill A while provably preserving
skill B is therefore a genuine surgery question on shared machinery.

Because there is a single block and the logit is read at the last
position, only the LAST attention row is load-bearing — the verifier
exploits that (encode one row per head, not L), and the experiments must
show the attention is genuinely doing the work (knock a head out and the
skill should die; see run_transformer.py).

Everything is plain NumPy with hand-written backprop, like every other
model in this repo. Fixed seeds; no dependencies beyond numpy.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Vocabulary and ground truth.
# ---------------------------------------------------------------------------
VOCAB = ["Q1", "Q2", "a", "qo", "qc", "bo", "bc"]   # qo/qc = « », bo/bc = ( )
Q1, Q2, TOK_A, QO, QC, BO, BC = range(7)
V = len(VOCAB)
TEXT_TOKENS = [TOK_A, QO, QC, BO, BC]               # legal at positions 1..L-1
L = 8                                               # 1 task token + 7 text


def set_seq_len(n: int):
    """Change the sequence length (fallback-ladder lever: the symbolic
    sequence space is 5^(L-1)). Call BEFORE building/training a model;
    models built under different L are not interchangeable. The sibling
    modules (verify_transformer, run_transformer) read L dynamically."""
    global L
    L = n


def label_quote(tokens: np.ndarray) -> np.ndarray:
    """Skill A's truth: more « than » in the text positions."""
    text = tokens[:, :L - 1]
    return ((text == QO).sum(axis=1) > (text == QC).sum(axis=1)).astype(float)


def label_bracket(tokens: np.ndarray) -> np.ndarray:
    """Skill B's truth: more ( than ) in the text positions."""
    text = tokens[:, :L - 1]
    return ((text == BO).sum(axis=1) > (text == BC).sum(axis=1)).astype(float)


def true_label(tokens: np.ndarray) -> np.ndarray:
    """The answer the model owes: skill A's label under Q1, skill B's under
    Q2."""
    return np.where(tokens[:, L - 1] == Q1, label_quote(tokens),
                    label_bracket(tokens))


def sample_batch(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Random text + random task token (at the last position)."""
    tokens = np.empty((n, L), dtype=int)
    tokens[:, :L - 1] = rng.choice(TEXT_TOKENS, size=(n, L - 1))
    tokens[:, L - 1] = rng.integers(0, 2, n)        # Q1 or Q2
    return tokens, true_label(tokens)


# ---------------------------------------------------------------------------
# Sparsemax (forward + the piece of its Jacobian backprop needs).
# ---------------------------------------------------------------------------
def sparsemax(z: np.ndarray) -> np.ndarray:
    """Rowwise projection of z (..., n) onto the probability simplex."""
    zs = np.sort(z, axis=-1)[..., ::-1]
    css = np.cumsum(zs, axis=-1)
    k = np.arange(1, z.shape[-1] + 1)
    cond = 1.0 + k * zs > css
    kz = cond.sum(axis=-1)                          # support size, >= 1
    tau = (np.take_along_axis(css, kz[..., None] - 1, axis=-1)[..., 0]
           - 1.0) / kz
    return np.maximum(z - tau[..., None], 0.0)


def sparsemax_backward(p: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Jacobian-vector product: on the support S, dz = g - mean_S(g)."""
    support = (p > 0).astype(float)
    ssum = (grad * support).sum(axis=-1, keepdims=True)
    scount = support.sum(axis=-1, keepdims=True)
    return support * (grad - ssum / scount)


def leaky(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)


def dleaky(x, alpha=0.01):
    return np.where(x >= 0, 1.0, alpha)


# ---------------------------------------------------------------------------
# The model.
# ---------------------------------------------------------------------------
class TinyTransformer:
    """
    Embeddings (V x d) + learned positions (L x d), one decoder block:
      per head h:  q_t = Wq x_t + bq,  k_s = Wk x_s + bk,  v_s = Wv x_s
                   score_{t,s} = u . LeakyReLU(q_t + k_s)      (additive)
                   a_{t,.} = sparsemax over s <= t (causal)
                   o_t = sum_s a_{t,s} v_s
      x'_t  = x_t + Wo [o^1_t; ...; o^H_t] + bo                 (residual)
      x''_t = x'_t + W2 LeakyReLU(W1 x'_t + b1) + b2            (MLP)
      logit = r . x''_{L-1} + c                                 (readout)

    Decision rule as everywhere in this repo: logit > 0 = "yes".
    """

    def __init__(self, d_model=16, n_heads=2, d_head=8, d_mlp=32, seed=0,
                 alpha=0.1, attn="sparsemax", gate_T=0.5):
        """
        attn: "sparsemax" — soft simplex weights (the trainable reference).
              "gate"      — THRESHOLD attention: position s contributes its
              value iff its score clears zero; contributions are SUMMED,
              not averaged. This is the variant the solver certifies: soft
              attention's weight-times-value application is bilinear in
              input noise (weights and values both depend on it), which no
              exact linear-arithmetic encoding can express — gates make the
              weights 0/1 If-selections and keep everything piecewise
              linear. Gate models train with a sigmoid surrogate of
              temperature gate_T and are HARDENED (exact step) afterwards;
              `self.gate_hard` switches the forward between the two.

        score_act: "leaky" — a LeakyReLU inside the additive score (the
              Bahdanau form). "linear" — scores are affine in query and
              key. Certified gate subjects use "linear": the score
              nonlinearity added 128 solver case-splits (2 heads x 8
              positions x 8 dims), measured as the difference between a
              30-minute timeout and tractable proofs, while content-based
              selection only needs the learned embeddings to be linearly
              separable — which they are free to become in training.
        """
        # Init scales matter here (found by sweep): the score pathway (u, Wq, Wk) starts SMALL so
        # early attention is nearly flat — sparsemax then keeps a wide
        # support and gradients reach every position. Starting sharp makes
        # sparsemax's exact zeros cut off learning (its gradient is zero
        # outside the support) and training plateaus far below 99%.
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.d, self.H, self.dh, self.m = d_model, n_heads, d_head, d_mlp
        self.alpha = alpha
        self.attn = attn
        self.gate_T = gate_T
        self.gate_hard = False
        self.score_act = "leaky"      # gate subjects switch to "linear"
        self.E = rng.normal(0, 1.0, (V, d_model)) * s * 1.5
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
        self.r = rng.normal(0, 1.0 / np.sqrt(d_model), d_model)
        self.c = 0.0

    # ---- forward -----------------------------------------------------------
    def embed(self, tokens: np.ndarray, noise: np.ndarray | None = None):
        """tokens (N, L) -> embedded inputs (N, L, d); optional additive
        embedding-space noise of the same shape (this is where the certified
        perturbation lives — at the very input)."""
        x = self.E[tokens] + self.P[None, :, :]
        if noise is not None:
            x = x + noise
        return x

    def forward(self, tokens: np.ndarray, noise: np.ndarray | None = None,
                ablate_heads: tuple = (), cache: dict | None = None):
        """Return the logit (N,). Only the last position is read, and with a
        single block only the LAST attention row matters — computed as such.
        ablate_heads: head indices whose output is forced to zero (the
        numeric twin of the head-ablation edit)."""
        x = self.embed(tokens, noise)                    # (N, L, d)
        N = x.shape[0]
        t = L - 1
        heads = []
        for h in range(self.H):
            q_t = x[:, t] @ self.Wq[h].T + self.bq[h]    # (N, dh)
            k = x @ self.Wk[h].T + self.bk[h]            # (N, L, dh)
            pre = q_t[:, None, :] + k                    # (N, L, dh)
            act = pre if self.score_act == "linear" \
                else leaky(pre, self.alpha)
            scores = act @ self.u[h]                     # (N, L)
            if self.attn == "gate":
                if self.gate_hard:
                    a = (scores > 0).astype(float)       # exact 0/1 gates
                else:
                    a = _sigmoid(scores / self.gate_T)   # training surrogate
            else:
                a = sparsemax(scores)                    # causal row t = L-1
            v = x @ self.Wv[h].T                         # (N, L, dh)
            o = (a[:, :, None] * v).sum(axis=1)          # (N, dh)
            if h in ablate_heads:
                o = np.zeros_like(o)
            heads.append((q_t, k, pre, act, scores, a, v, o))
        o_cat = np.concatenate([hd[7] for hd in heads], axis=1)
        x1 = x[:, t] + o_cat @ self.Wo.T + self.bo       # (N, d)
        z1 = x1 @ self.W1.T + self.b1
        a1 = leaky(z1, self.alpha)
        x2 = x1 + a1 @ self.W2.T + self.b2
        logit = x2 @ self.r + self.c
        if cache is not None:
            cache.update(x=x, heads=heads, o_cat=o_cat, x1=x1, z1=z1,
                         a1=a1, x2=x2)
        return logit

    def params(self):
        return ["E", "P", "Wq", "bq", "Wk", "bk", "u", "Wv", "Wo", "bo",
                "W1", "b1", "W2", "b2", "r", "c"]

    def clone(self):
        import copy
        return copy.deepcopy(self)

    def harden(self):
        """Switch gate attention from the training surrogate to exact 0/1
        gates. The hardened model is the certified subject."""
        assert self.attn == "gate"
        self.gate_hard = True
        return self

    def quantize(self, bits=12):
        """Round every weight to the 2^-bits grid. Small rationals keep the
        solver's exact arithmetic fast, and the quantized model IS the
        certified artifact — certificates then describe exactly what runs
        (float-gap discipline). Re-measure accuracy after calling this."""
        scale = float(2 ** bits)
        for name in self.params():
            v = getattr(self, name)
            if isinstance(v, float):
                setattr(self, name, round(v * scale) / scale)
            else:
                setattr(self, name, np.round(v * scale) / scale)
        return self


# ---------------------------------------------------------------------------
# Training: sigmoid cross-entropy on the single logit, full manual backprop.
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _grads(model: TinyTransformer, tokens, y):
    """Forward + hand-written backward. Returns (loss, grads dict)."""
    cache = {}
    logit = model.forward(tokens, cache=cache)
    N = tokens.shape[0]
    p = _sigmoid(logit)
    loss = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean()
    dlogit = (p - y) / N                                  # (N,)

    g = {k: np.zeros_like(getattr(model, k)) if k != "c" else 0.0
         for k in model.params()}
    x, x1, z1, a1, x2 = (cache[k] for k in ("x", "x1", "z1", "a1", "x2"))
    t = L - 1

    # readout
    g["r"] = x2.T @ dlogit
    g["c"] = dlogit.sum()
    dx2 = dlogit[:, None] * model.r[None, :]              # (N, d)

    # MLP (residual)
    g["W2"] = dx2.T @ a1
    g["b2"] = dx2.sum(axis=0)
    da1 = dx2 @ model.W2
    dz1 = da1 * dleaky(z1, model.alpha)
    g["W1"] = dz1.T @ x1
    g["b1"] = dz1.sum(axis=0)
    dx1 = dx2 + dz1 @ model.W1                            # (N, d)

    # output projection (residual)
    o_cat = cache["o_cat"]
    g["Wo"] = dx1.T @ o_cat
    g["bo"] = dx1.sum(axis=0)
    do_cat = dx1 @ model.Wo                               # (N, H*dh)
    dx = np.zeros_like(x)                                 # (N, L, d)
    dx[:, t] += dx1                                       # residual path

    for h in range(model.H):
        q_t, k, pre, act, scores, a, v, o = cache["heads"][h]
        do = do_cat[:, h * model.dh:(h + 1) * model.dh]   # (N, dh)
        # o = sum_s a_s v_s
        da = (do[:, None, :] * v).sum(axis=2)             # (N, L)
        dv = a[:, :, None] * do[:, None, :]               # (N, L, dh)
        # v = x @ Wv^T
        g["Wv"][h] = np.einsum("nld,nle->de", dv, x)
        dx += dv @ model.Wv[h]
        # attention-weight backward, per variant
        if model.attn == "gate":
            # sigmoid surrogate; when hardened, straight-through (forward
            # used the step, gradient flows through the surrogate)
            asig = _sigmoid(scores / model.gate_T)
            dscores = da * asig * (1 - asig) / model.gate_T
        else:
            dscores = sparsemax_backward(a, da)           # (N, L)
        # scores = act @ u
        g["u"][h] = np.einsum("nl,nld->d", dscores, act)
        dact = dscores[:, :, None] * model.u[h][None, None, :]
        dpre = dact if model.score_act == "linear" \
            else dact * dleaky(pre, model.alpha)          # (N, L, dh)
        # pre = q_t[:, None] + k
        dq_t = dpre.sum(axis=1)                           # (N, dh)
        dk = dpre                                         # (N, L, dh)
        g["Wq"][h] = dq_t.T @ x[:, t]
        g["bq"][h] = dq_t.sum(axis=0)
        dx[:, t] += dq_t @ model.Wq[h]
        g["Wk"][h] = np.einsum("nld,nle->de", dk, x)
        g["bk"][h] = dk.sum(axis=(0, 1))
        dx += dk @ model.Wk[h]

    # embeddings and positions
    np.add.at(g["E"], tokens.reshape(-1),
              dx.reshape(-1, model.d))
    g["P"] = dx.sum(axis=0)
    return loss, g


def train(model: TinyTransformer, steps=6000, lr=0.02, batch=512, seed=1,
          weight_decay=1e-4, verbose=True):
    """Plain gradient descent with momentum; prints progress occasionally.
    weight_decay keeps the trained logits O(1)-ish instead of letting BCE
    push them to +-70 — big logits mean big Lipschitz constants, which both
    weakens tiny-epsilon certificates and slows the solver."""
    rng = np.random.default_rng(seed)
    vel = {k: np.zeros_like(getattr(model, k)) if k != "c" else 0.0
           for k in model.params()}
    for step in range(steps):
        tokens, y = sample_batch(batch, rng)
        loss, g = _grads(model, tokens, y)
        for kname in model.params():
            gk = g[kname] + weight_decay * getattr(model, kname)
            vel[kname] = 0.9 * vel[kname] - lr * gk
            setattr(model, kname, getattr(model, kname) + vel[kname])
        if verbose and (step % 1000 == 0 or step == steps - 1):
            acc = accuracy(model, *sample_batch(4000, rng))
            print(f"    step {step:5d}: loss {loss:.4f}  "
                  f"acc A {acc[0]:.4f}  acc B {acc[1]:.4f}")
    return model


# ---------------------------------------------------------------------------
# Edits, all expressed as weight changes (house rule: the prover only ever
# sees an ordinary model, never an edit flag).
# ---------------------------------------------------------------------------
def ablate_head(model: TinyTransformer, h: int) -> TinyTransformer:
    """Switch attention head h off: zero the output-projection columns that
    read it. Numerically identical to forward(ablate_heads=(h,))."""
    m = model.clone()
    m.Wo[:, h * m.dh:(h + 1) * m.dh] = 0.0
    return m


def ablate_mlp_neurons(model: TinyTransformer, neurons) -> TinyTransformer:
    """Switch MLP neurons off entirely (zero their incoming row and bias)."""
    m = model.clone()
    for j in neurons:
        m.W1[j, :] = 0.0
        m.b1[j] = 0.0
    return m


def weight_edit_mlp(model: TinyTransformer, neurons) -> TinyTransformer:
    """Cut only the outgoing wires of the chosen MLP neurons (they still
    fire; nothing downstream listens)."""
    m = model.clone()
    for j in neurons:
        m.W2[:, j] = 0.0
    return m


def steer_residual(model: TinyTransformer, vector: np.ndarray
                   ) -> TinyTransformer:
    """Steering as a weight change: add a fixed vector to the residual
    stream right after attention (bo is exactly that hook)."""
    m = model.clone()
    m.bo = m.bo + vector
    return m


def targeted_suppression_direction(model: TinyTransformer, neurons) -> np.ndarray:
    """The SURGICAL steering recipe, transformer edition (the analogue of the
    toy model's `targeted_suppression_vector`): a residual-stream push that
    drives ONLY the chosen circuit MLP neurons' pre-activations down, leaving
    every other neuron's pre-activation exactly unchanged.

    The MLP pre-activation of neuron j is z1[j] = (residual) @ W1[j] + b1[j].
    Steering adds a vector v to the residual (via bo), so z1[j] -> z1[j] +
    W1[j] @ v. We solve for the least-norm v that sets W1[j] @ v = -1 for every
    target neuron j and W1[k] @ v = 0 for... — actually we only *constrain* the
    targets (least-norm solution to the underdetermined target rows), which
    keeps v small and, because it lives in the span of the target rows, touches
    other neurons only through whatever overlap those rows already have. The
    caller scales this unit-ish direction by a dose. Returns the normalized
    direction; `steer_residual(model, -dose * dir)` applies the suppression.

    The point of the row: unlike ablation, steering lives in the RESIDUAL, so it
    can push MLP neurons but has no handle on an attention head. When skill A's
    circuit includes a load-bearing head, even a perfectly targeted residual
    push cannot remove the skill — which the certified table then shows as a
    number."""
    Wt = model.W1[list(neurons), :]                 # (k, d_model)
    # least-norm v with Wt @ v = -1 (push each target pre-activation down by 1)
    target = -np.ones(len(neurons))
    v, *_ = np.linalg.lstsq(Wt, target, rcond=None)
    return v / (np.linalg.norm(v) + 1e-12)


def diff_of_means_direction(model: TinyTransformer, n=6000, seed=17
                            ) -> np.ndarray:
    """The realistic steering recipe, transformer edition: the mean
    post-attention residual (at the readout position) on Q1 inputs where
    the quote skill should fire, minus the mean where it shouldn't —
    normalized. Steering pushes along 'off minus on'."""
    rng = np.random.default_rng(seed)
    tokens, _ = sample_batch(n, rng)
    tokens[:, L - 1] = Q1
    yA = label_quote(tokens)
    cache = {}
    model.forward(tokens, cache=cache)
    x1 = cache["x1"]
    direction = x1[yA == 0].mean(axis=0) - x1[yA == 1].mean(axis=0)
    return direction / np.linalg.norm(direction)


def train_gate_subject(seed=0, config=None, verbose=True) -> TinyTransformer:
    """The CERTIFIED threshold-gate SUBJECT, start to finish: train gate attention
    on the sigmoid surrogate, harden to exact 0/1 gates, fine-tune
    straight-through to recover any hardening loss, quantize the weights
    to the 2^-12 grid (the solver's exact rationals stay small, and the
    quantized model is the deployed artifact). Callers should re-measure
    accuracy on held-out data after this returns.

    `config` (a dict of TinyTransformer kwargs — d_model, n_heads, d_head,
    d_mlp) selects the architecture. The default (None) is the full model;
    the CERTIFIED subject is the smaller config in run_transformer.py,
    because exact input-side certification only becomes tractable there —
    see the size ladder in results/rung2_size_ladder.log."""
    m = TinyTransformer(seed=seed, attn="gate", **(config or {}))
    m.score_act = "linear"        # see __init__: the certified subject's
                                  # scores are affine (solver-cost decision)
    train(m, steps=6000, verbose=verbose)
    m.harden()
    train(m, steps=1500, lr=0.01, seed=seed + 50, verbose=verbose)
    m.quantize(bits=12)
    return m


def accuracy(model: TinyTransformer, tokens, y,
             ablate_heads: tuple = ()) -> tuple[float, float]:
    """Per-skill accuracy (skill A on Q1 rows, skill B on Q2 rows)."""
    pred = (model.forward(tokens, ablate_heads=ablate_heads) > 0).astype(float)
    isA = tokens[:, L - 1] == Q1
    accA = float((pred[isA] == y[isA]).mean()) if isA.any() else float("nan")
    accB = float((pred[~isA] == y[~isA]).mean()) if (~isA).any() \
        else float("nan")
    return accA, accB


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    tokens, y = sample_batch(20000, rng)

    print("Training the sparsemax reference (smoke run)...")
    m = train(TinyTransformer(seed=0), verbose=False)
    a, b = accuracy(m, tokens, y)
    lg = m.forward(tokens)
    print(f"  held-out: A {a:.4f}, B {b:.4f}; logit range "
          f"[{lg.min():.1f}, {lg.max():.1f}]")

    print("Training the certified subject (gate attention, hardened + "
          "quantized)...")
    g = train_gate_subject(seed=0, verbose=False)
    a, b = accuracy(g, tokens, y)
    lg = g.forward(tokens)
    print(f"  held-out: A {a:.4f}, B {b:.4f}; logit range "
          f"[{lg.min():.1f}, {lg.max():.1f}]")
