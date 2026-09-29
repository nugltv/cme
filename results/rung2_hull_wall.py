"""
rung2_hull_wall.py — the hull encoding at the larger sizes: every query
                     exceeds a 30-minute budget (paper Table III note).
=============================================================================

Run it with:  python results/rung2_hull_wall.py    (takes ~2 hours; every
                                                    query burns its full
                                                    30-minute timeout)

WHAT THIS ASKS. The exact certificate is one claim about a small transformer
covering EVERY input sequence at once, with continuous nudges allowed on
every position. One candidate cost is the sequences: choosing a token per
position is a yes/no decision, and the solver has to consider all 5^(L-1)
combinations.

WHAT THIS TESTS. The "hull" encoding removes the yes/no choices entirely:
instead of picking a token per position, each position gets a continuous
blend of the tokens (mixture weights on the token simplex), with the skill's
label enforced as a mass gap. That region CONTAINS every real sequence, so
proving it would still prove the real claim (proposition P1) — and it
contains no Boolean branching at all.

WHAT IT FINDS (see rung2_hull_wall.log): every query times out, at both L = 6
and L = 8, at both noise sizes, on the larger model. So the sequence space is
not the bottleneck — the cost is the number of yes/no branches the attention
gates themselves create against ~100+ noise variables. The sizes that do
certify are in results/rung2_size_ladder.log.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import transformer_model as tm
from transformer_model import train_gate_subject, sample_batch, accuracy, Q1
from run_transformer import validate_encoding
from verify_transformer import SequenceClass, prove_transformer

for Lval in (6, 8):
    tm.set_seq_len(Lval)
    print(f"--- L={Lval}: training...", flush=True)
    m = train_gate_subject(seed=0, verbose=False)
    rng = np.random.default_rng(123)
    tokens, y = sample_batch(20000, rng)
    print(f"held-out acc: {accuracy(m, tokens, y)}", flush=True)
    gap = validate_encoding(m)          # hull mode with pinned integral lam
    print(f"encoding validation: max gap {gap:.2e}", flush=True)
    for eps in (0.01, 0.05):
        t0 = time.time()
        r = prove_transformer(m, SequenceClass(Q1, "pos"), "positive", eps,
                              slack=1e-6, timeout_ms=1800000)
        cx = r.get("counterexample")
        extra = "" if cx is None else f" (discrete={cx['discrete']})"
        print(f"HULL control A-pos L={Lval} eps={eps}: "
              f"{'PROVED' if r['proved'] else r['status']}{extra} "
              f"in {time.time()-t0:.1f}s", flush=True)
print("ALL DONE", flush=True)
