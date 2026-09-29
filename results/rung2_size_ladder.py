"""
rung2_size_ladder.py — how large can the threshold-gate subject be and still
certify its control claim tractably? (paper Table III)

The Boolean token encoding exceeds the time budget at every size; the hull
encoding is tractable on small configs. This sweeps configs from small to large
to locate the exact-certification frontier.

METHOD. Sweep configs small -> large. For each: train the gate subject to
>=0.99 on both skills (skip+note if it can't), validate the encoding, then
run the skill-A control in HULL mode at eps=0 (must be UNSAT/proved for the
config to be a valid control subject; measures unsat speed) and at a small
eps>0 (noise DOF active). A 300s per-query timeout flags a timeout. A config
is pipeline-viable if BOTH prove well under the timeout (a full 7-query
bisection is ~7x one query, so ~<120s/query keeps the pipeline < ~15 min).

Verdicts: PROVED = unsat (good); a fast SAT would mean the control fails at
that eps (a radius fact, still 'tractable'); unknown = timeout. Solver times
depend on the machine; the verdicts are what Table III reports. Output:
results/rung2_size_ladder.log.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import transformer_model as tm
from transformer_model import (TinyTransformer, train, sample_batch, accuracy,
                               Q1)
from run_transformer import validate_encoding
from verify_transformer import SequenceClass, prove_transformer

TIMEOUT_MS = 300_000
PROBE_EPS = (0.0, 0.005)

# (label, L, config kwargs). Ordered by roughly increasing If-split /
# noise-variable cost. The last is the original full-size subject (expect timeouts).
LADDER = [
    ("baseline  d8 h1 mlp8  L4", 4,
     dict(d_model=8, n_heads=1, d_head=4, d_mlp=8)),
    ("d8 h1 mlp8  L6", 6,
     dict(d_model=8, n_heads=1, d_head=4, d_mlp=8)),
    ("d8 h1 mlp8  L8", 8,
     dict(d_model=8, n_heads=1, d_head=4, d_mlp=8)),
    ("d12 h1 mlp12 L6", 6,
     dict(d_model=12, n_heads=1, d_head=4, d_mlp=12)),
    ("d16 h1 mlp16 L6", 6,
     dict(d_model=16, n_heads=1, d_head=8, d_mlp=16)),
    ("d16 h2 mlp32 L8 (orig)", 8,
     dict(d_model=16, n_heads=2, d_head=8, d_mlp=32)),
]


def build_subject(cfg, seed=0):
    m = TinyTransformer(seed=seed, attn="gate", **cfg)
    m.score_act = "linear"
    train(m, steps=6000, verbose=False)
    m.harden()
    train(m, steps=1500, lr=0.01, seed=seed + 50, verbose=False)
    m.quantize(bits=12)
    return m


def main():
    rng = np.random.default_rng(123)
    for label, L, cfg in LADDER:
        tm.set_seq_len(L)
        noise_vars = L * cfg["d_model"]
        print(f"\n=== {label}  (noise vars = {noise_vars}) ===", flush=True)
        t0 = time.time()
        model = build_subject(cfg)
        tokens, y = sample_batch(20000, rng)
        accA, accB = accuracy(model, tokens, y)
        print(f"  trained {time.time()-t0:.0f}s; acc A {accA:.4f} B {accB:.4f}",
              flush=True)
        if min(accA, accB) < 0.99:
            print("  -> did not learn both skills; skipping solver probe.",
                  flush=True)
            continue
        gap = validate_encoding(model, n=12)
        if gap >= 1e-9:
            print(f"  -> encoding gap {gap:.2e} too large; skipping.",
                  flush=True)
            continue
        print(f"  encoding gap {gap:.1e} OK", flush=True)
        for eps in PROBE_EPS:
            t0 = time.time()
            r = prove_transformer(model, SequenceClass(Q1, "pos"), "positive",
                                  eps, slack=1e-6, timeout_ms=TIMEOUT_MS,
                                  token_mode="hull")
            dt = time.time() - t0
            verdict = "PROVED" if r["proved"] else r["status"].split("(")[0]
            print(f"  hull A-pos eps={eps}: {verdict.strip()} in {dt:.1f}s",
                  flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
