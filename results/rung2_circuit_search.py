"""
rung2_circuit_search.py — which seeds/configs of the threshold-gate subject
admit a removable skill-A circuit? (paper Fig. 3 claim; numeric, no solver)

For each (config, seed): train to acc, then characterize removability of
skill A by ABLATION on the clean A-positive sequences:
  - head ablation: A-pos worst logit (want <=0) and skill-B accuracy (want ~1)
  - all-MLP ablation: same
  - rt.find_circuit: does a test-passing ablation circuit exist?
Obstruction readout: if A-pos worst logit stays >0 even with everything
ablated -> RESTING BIAS (un-removable by silencing). If it goes <=0 only when
B accuracy collapses -> ENTANGLEMENT (shared trunk). If find_circuit succeeds ->
this seed/config is usable. For the certified seed 5, switching off the heads
alone, or the MLP alone, leaves skill A firing.

Configs kept solver-plausible: noise vars = L*d_model stays 48 (L6, d8);
we vary heads and MLP width (more ablation atoms / head specialization),
which add If-splits but not noise variables.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import transformer_model as _tm
_tm.set_seq_len(6)

import run_transformer as rt
from transformer_model import (TinyTransformer, train, sample_batch, accuracy,
                               Q1, Q2, ablate_head, ablate_mlp_neurons)
from verify_transformer import all_sequences

rt.EPS0 = 0.004                     # small noise gate for the numeric tests_pass

CONFIGS = [
    ("d8 h1 mlp8", dict(d_model=8, n_heads=1, d_head=4, d_mlp=8)),
    ("d8 h1 mlp16", dict(d_model=8, n_heads=1, d_head=4, d_mlp=16)),
    ("d8 h2 mlp8", dict(d_model=8, n_heads=2, d_head=4, d_mlp=8)),
]
SEEDS = range(6)


def build(cfg, seed):
    m = TinyTransformer(seed=seed, attn="gate", **cfg)
    m.score_act = "linear"
    train(m, steps=6000, verbose=False)
    m.harden()
    train(m, steps=1500, lr=0.01, seed=seed + 50, verbose=False)
    m.quantize(bits=12)
    return m


def a_pos_worst_and_B(model):
    """A-pos worst logit (removal target) and skill-B clean accuracy."""
    tA = all_sequences(Q1)
    tA_pos = tA[rt.CLAIMS["A pos"][0].numeric_mask(tA)]
    worstA = float(model.forward(tA_pos).max())
    tB = all_sequences(Q2)
    yB = np.array([1.0 if rt.CLAIMS["B pos"][0].numeric_mask(tB[i:i+1])[0]
                   else 0.0 for i in range(len(tB))])
    # B accuracy = fraction where sign(logit) matches which B-class it's in
    predB = (model.forward(tB) > 0).astype(float)
    yBtrue = rt.CLAIMS["B pos"][0].numeric_mask(tB).astype(float)
    accB = float((predB == yBtrue).mean())
    return worstA, accB


def main():
    rng = np.random.default_rng(123)
    for label, cfg in CONFIGS:
        print(f"\n=== {label} (L6, noise vars {6*cfg['d_model']}) ===",
              flush=True)
        for seed in SEEDS:
            m = build(cfg, seed)
            tok, y = sample_batch(8000, rng)
            accA, accB = accuracy(m, tok, y)
            if min(accA, accB) < 0.99:
                print(f"  seed {seed}: undertrained (A {accA:.3f} B {accB:.3f})",
                      flush=True)
                continue
            baseA, _ = a_pos_worst_and_B(m)
            # head ablation (all heads)
            mh = m
            for h in range(m.H):
                mh = ablate_head(mh, h)
            hA, hB = a_pos_worst_and_B(mh)
            # all-MLP ablation
            mm = ablate_mlp_neurons(m, list(range(m.m)))
            mA, mB = a_pos_worst_and_B(mm)
            # full pipeline circuit search
            circ, _ = rt.find_circuit(m)
            print(f"  seed {seed}: baseA_worst={baseA:+.2f} | "
                  f"heads_off A={hA:+.2f} B_acc={hB:.2f} | "
                  f"mlp_off A={mA:+.2f} B_acc={mB:.2f} | "
                  f"circuit={circ}", flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
