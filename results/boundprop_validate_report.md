# M0: bound propagation (auto_LiRPA) validated against exact Z3

Output of `run_boundprop_validate.py` (check M0). Before trusting the sound-but-incomplete bound-propagation path on a transformer Z3 cannot reach, we check it on the one subject where BOTH tools apply — the toy ReLU MLP. The gate: auto_LiRPA's certified radius must **under-approximate** the exact Z3 radius on every claim (never certify a larger wiggle room — that would be unsound).

Toy model accuracy: skill A 0.9992, skill B 1.0000. Skill-A circuit (ablated): [3]. Torch-vs-numpy edited-forward gap: 3.55e-15 (the two agree, so both tools bound the same function).

| claim | auto_LiRPA radius | Z3 exact radius | sound? |
|---|---|---|---|
| removal (A nonpositive over A-region) | >= 0.5 (cap) | >= 0.5 (cap) | yes |
| preservation (B positive over B-high) | 0.100 | 0.100 | yes |
| preservation (B nonpositive over B-low) | 0.100 | 0.100 | yes |

**Gate verdict:** PASS — auto_LiRPA under-approximates the exact Z3 radius on every claim; the pipeline is sound and ready for M1.

Reading it: auto_LiRPA (CROWN, backward mode) computes a sound linear relaxation of the edited network over each box; its certified radius is a lower bound on the true one, so it should sit at or below Z3's exact radius. Equality/closeness also means the relaxation is tight enough to be *useful*, not merely sound. With M0 passing, M1 runs the same pipeline on a softmax+LayerNorm transformer past the exact-Z3 frontier, where Z3 cannot follow and a PGD attack brackets the certified radius from above.