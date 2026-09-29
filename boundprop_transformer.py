"""
boundprop_transformer.py — a standard softmax+LayerNorm transformer (paper M1, Fig. 2)
=============================================================================

The threshold-gate subject uses *gate* attention on purpose: softmax and LayerNorm are
not exactly Z3-encodable, so the exact pipeline can't touch a standard
transformer at all. Bound propagation (auto_LiRPA) *can* — it bounds softmax and
LayerNorm. This module is the standard-architecture subject M1 certifies with
auto_LiRPA, past the exact-Z3 frontier.

Design choices that matter:
- **Manual attention from primitives** (explicit Q/K/V linears, scaled
  dot-product, softmax, matmul) — NOT `nn.MultiheadAttention`, so auto_LiRPA sees
  the ops it supports (BoundMatMul / BoundSoftmax / BoundLayerNormalization).
- **The forward that gets bounded takes EMBEDDINGS**, not token ids, so the
  perturbation region is an embedding-space ball *ahead of the whole forward
  pass* (never a late internal value).
- **Short sequence, real-size model.** L is small enough to *enumerate every
  sequence* (so the certified claim keeps the "all sequences" quantifier),
  while d_model / heads / layers put it well past the exact frontier (~48 vars).

Task: the two skills live on DISJOINT
POSITIONS so their circuits can be edited apart. A from-scratch softmax
transformer entangles two skills that share content positions (ablating skill
A's component also kills skill B), as on the modular adders. Disjoint
positions give the needed separability inside a standard architecture:
  positions [q0,q1,q2, b0,b1,b2, TASK], L=7.
  q0..q2 hold QUOTE content in {«,»};  b0..b2 hold BRACKET content in {(,)}.
  TQ  -> label 1 iff  #« > #»  among q-positions   (skill A — to be removed)
  TB  -> label 1 iff  #( > #)  among b-positions   (skill B — to be preserved)
Three positions per skill => odd count => no ties (majority is always decided).
Single scalar readout at the TASK position; >0 means "more open than close" for
the active skill. Skill A depends ONLY on q-positions, skill B ONLY on
b-positions, so a head/neuron that carries the quote count can be ablated
without touching the bracket count. Edits (ablation) are weight changes.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

# ---- vocabulary / task ------------------------------------------------------
LQ, RQ, LB, RB, TQ, TB = 0, 1, 2, 3, 4, 5   # « » ( ) task-quote task-bracket
V = 6
QPOS = [0, 1, 2]                             # quote-content positions (skill A)
BPOS = [3, 4, 5]                             # bracket-content positions (skill B)
L = 7                                        # 3 quote + 3 bracket + 1 task
QTOK = [LQ, RQ]
BTOK = [LB, RB]


def all_sequences():
    """Every sequence: 2^3 quote combos x 2^3 bracket combos x 2 task tokens."""
    seqs = []
    nq, nb = len(QPOS), len(BPOS)
    for qi in range(len(QTOK) ** nq):
        q = []
        k = qi
        for _ in range(nq):
            q.append(QTOK[k % 2]); k //= 2
        for bi in range(len(BTOK) ** nb):
            b = []
            k = bi
            for _ in range(nb):
                b.append(BTOK[k % 2]); k //= 2
            for task in (TQ, TB):
                seqs.append(q + b + [task])
    return np.array(seqs, dtype=np.int64)


def labels(seqs: np.ndarray) -> np.ndarray:
    """Binary label per sequence. Skill A reads q-positions, B reads b-positions;
    3 positions each => majority is always decided (no ties)."""
    seqs = np.asarray(seqs)
    y = np.zeros(len(seqs), dtype=np.float64)
    for i, s in enumerate(seqs):
        task = s[-1]
        if task == TQ:
            q = s[QPOS]
            y[i] = 1.0 if (q == LQ).sum() > (q == RQ).sum() else 0.0
        else:
            b = s[BPOS]
            y[i] = 1.0 if (b == LB).sum() > (b == RB).sum() else 0.0
    return y


# ---- model ------------------------------------------------------------------
class Block(nn.Module):
    """One standard pre-LN transformer block with manual softmax attention."""

    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)
        self.scale = 1.0 / (self.dh ** 0.5)
        self.ablate_heads: tuple = ()
        self.ablate_mlp: tuple = ()
        self.ablate_kv_positions: tuple = ()   # record of firewalled positions
        # constant per-position value mask (all ones = no edit). Built as a fixed
        # buffer OUTSIDE the traced forward, so auto_LiRPA sees only a multiply by
        # a constant (an in-forward index assignment would be an unsupported op).
        self.register_buffer("kv_mask", torch.ones(1, 1, L, 1))

    def _attn(self, x):
        B, L_, d = x.shape
        q = self.Wq(x).view(B, L_, self.h, self.dh).transpose(1, 2)   # B,h,L,dh
        k = self.Wk(x).view(B, L_, self.h, self.dh).transpose(1, 2)
        v = self.Wv(x).view(B, L_, self.h, self.dh).transpose(1, 2)
        # the edit: firewall the masked key/value positions — their VALUE
        # contribution is zeroed, so attending to them delivers nothing (an
        # attention knockout / path-patch of the read pathway to those
        # positions). kv_mask is a constant buffer (ones = no edit), so this is a
        # multiply by a constant and bound propagation stays linear here.
        v = v * self.kv_mask
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale     # B,h,L,L
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                                    # B,h,L,dh
        if self.ablate_heads:                                          # the edit
            mask = torch.ones(self.h, dtype=out.dtype, device=out.device)
            for hh in self.ablate_heads:
                mask[hh] = 0.0
            out = out * mask.view(1, self.h, 1, 1)
        out = out.transpose(1, 2).reshape(B, L_, d)
        return self.Wo(out)

    def _mlp(self, x):
        h = torch.relu(self.fc1(x))
        if self.ablate_mlp:                                            # the edit
            mask = torch.ones(h.shape[-1], dtype=h.dtype, device=h.device)
            for j in self.ablate_mlp:
                mask[j] = 0.0
            h = h * mask
        return self.fc2(h)

    def forward(self, x):
        x = x + self._attn(self.ln1(x))
        x = x + self._mlp(self.ln2(x))
        return x


class BoundCore(nn.Module):
    """The part that gets bounded: takes the (already looked-up) embedding tensor
    (B, L, d_model), adds positional embedding, runs the blocks, and reads out a
    scalar at the TASK (last) position. Perturbing this module's INPUT is an
    embedding-space perturbation ahead of the whole forward pass."""

    def __init__(self, pos, blocks: nn.ModuleList, ln_f, readout):
        super().__init__()
        self.pos = nn.Parameter(pos, requires_grad=False)
        self.blocks = blocks
        self.ln_f = ln_f
        self.readout = readout

    def forward(self, emb):                                            # emb: B,L,d
        x = emb + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        last = x[:, -1, :]                                             # TASK pos
        return self.readout(last)                                     # B,1


class GateXformer(nn.Module):
    """Full model: token embedding + BoundCore. `embed(tokens)` returns the
    perturbable embedding tensor; `core` is what auto_LiRPA bounds."""

    def __init__(self, d_model=64, n_heads=4, n_layers=2, d_mlp=128, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.d_model = d_model
        self.tok_emb = nn.Embedding(V, d_model)
        pos = torch.zeros(L, d_model)                # no batch dim (auto_LiRPA)
        nn.init.normal_(pos, std=0.02)
        blocks = nn.ModuleList(Block(d_model, n_heads, d_mlp)
                               for _ in range(n_layers))
        self.core = BoundCore(pos, blocks, nn.LayerNorm(d_model),
                              nn.Linear(d_model, 1))
        # float32 throughout: auto_LiRPA's transformer bound ops (softmax,
        # LayerNorm, matmul) are float32; M1 needs no float64 (there is no Z3
        # comparison at this scale — credibility is M0's Z3 agreement + PGD).

    def embed(self, tokens):                                          # tokens: B,L
        return self.tok_emb(tokens)

    def forward(self, tokens):
        return self.core(self.embed(tokens))                         # B,1

    # -- edits as weight/behavior changes (a cloned model with masks set) -----
    def clone(self):
        import copy
        return copy.deepcopy(self)

    def ablate(self, layer_heads=None, layer_mlp=None):
        """Return an edited clone. layer_heads/layer_mlp: dict {layer_idx: (idxs)}."""
        m = self.clone()
        for li, blk in enumerate(m.core.blocks):
            if layer_heads and li in layer_heads:
                blk.ablate_heads = tuple(layer_heads[li])
            if layer_mlp and li in layer_mlp:
                blk.ablate_mlp = tuple(layer_mlp[li])
        return m

    def ablate_positions(self, positions):
        """Return an edited clone that firewalls `positions` from every block's
        attention (their value contribution is zeroed in ALL layers, so no block
        can move information out of them). This is the mechanistic edit M1
        certifies: knock out the readout's attention pathway to the quote
        positions, removing the quote skill while the bracket pathway is
        untouched."""
        m = self.clone()
        for blk in m.core.blocks:
            blk.ablate_kv_positions = tuple(positions)
            mask = torch.ones(1, 1, L, 1)          # built outside any forward
            for p in positions:
                mask[0, 0, p, 0] = 0.0
            with torch.no_grad():
                blk.kv_mask.copy_(mask)
        return m


def train(model, seqs, y, steps=3000, lr=3e-3, seed=1, verbose=False):
    """Full-batch BCE training on the enumerated task (a few hundred sequences)."""
    tok = torch.tensor(seqs)
    tgt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for t in range(steps):
        opt.zero_grad()
        out = model(tok)
        loss = lossf(out, tgt)
        loss.backward()
        opt.step()
        if verbose and t % 500 == 0:
            print(f"    step {t}: loss {loss.item():.4f}")
    return model


def skill_accuracy(model, seqs, y):
    """Per-skill accuracy (TQ = skill A, TB = skill B)."""
    with torch.no_grad():
        pred = (model(torch.tensor(seqs)).view(-1) > 0).float().numpy()
    task = seqs[:, -1]
    accA = float((pred[task == TQ] == y[task == TQ]).mean())
    accB = float((pred[task == TB] == y[task == TB]).mean())
    return accA, accB
