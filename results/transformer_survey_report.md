# Circuit survey: how common is a distributed, attention-load-bearing circuit?

Output of `run_transformer_survey.py` — numeric only (the same test gate `run_transformer.py` uses: all sequences enumerated + sampled embedding noise; NO certificates). Context for the certified transformer result (`transformer_report.md`, seed 5, whose circuit is cross-component): is that circuit typical, or the exception? Each model is the exact certified artifact: `train_gate_subject` (train → harden → fine-tune → quantize to 2^-12).

**Circuit classes.** *cross-component* = ≥ 1 attention head AND ≥ 1 MLP neuron (attention genuinely load-bearing — the certified seed's kind); *single head* / *single mlp* = exactly one atom; *multi heads* / *multi mlp* = several atoms of one type (distributed within a component); *none* = no test-passing circuit.

## L=5 (step down)  (config {'d_model': 8, 'n_heads': 2, 'd_head': 4, 'd_mlp': 8})

| seed | usable? | circuit | class |
|---|---|---|---|
| 0 | yes | None | none |
| 1 | yes | None | none |
| 2 | yes | None | none |
| 3 | yes | [('head', 0)] | single head |
| 4 | yes | None | none |
| 5 | yes | None | none |
| 6 | yes | None | none |
| 7 | yes | None | none |

**L=5 (step down): 8 usable seeds** — none: 7, single head: 1. Cross-component (attention load-bearing) circuits: **0 of 8**.

## L=6 (certified)  (config {'d_model': 8, 'n_heads': 2, 'd_head': 4, 'd_mlp': 8})

| seed | usable? | circuit | class |
|---|---|---|---|
| 0 | yes | [('mlp', 7), ('mlp', 3), ('mlp', 6), ('mlp', 0), ('mlp', 2)] | multi mlp |
| 1 | yes | [('head', 1)] | single head |
| 2 | yes | None | none |
| 3 | yes | [('head', 0)] | single head |
| 4 | yes | None | none |
| 5 | yes | [('head', 0), ('mlp', 1), ('mlp', 5), ('mlp', 2)] | cross-component |
| 6 | yes | None | none |
| 7 | yes | [('head', 0)] | single head |
| 8 | yes | None | none |
| 9 | yes | [('mlp', 1), ('mlp', 2), ('mlp', 4), ('mlp', 5), ('mlp', 0), ('mlp', 6)] | multi mlp |
| 10 | yes | None | none |
| 11 | yes | None | none |

**L=6 (certified): 12 usable seeds** — cross-component: 1, multi mlp: 2, none: 6, single head: 3. Cross-component (attention load-bearing) circuits: **1 of 12**.

## L=7 (step up)  (config {'d_model': 8, 'n_heads': 2, 'd_head': 4, 'd_mlp': 8})

| seed | usable? | circuit | class |
|---|---|---|---|
| 0 | yes | None | none |
| 1 | yes | [('head', 1)] | single head |
| 2 | yes | [('mlp', 6)] | single mlp |
| 3 | yes | None | none |
| 4 | yes | None | none |
| 5 | yes | None | none |
| 6 | yes | None | none |
| 7 | yes | None | none |

**L=7 (step up): 8 usable seeds** — none: 6, single head: 1, single mlp: 1. Cross-component (attention load-bearing) circuits: **0 of 8**.

## The population picture

Across all swept seeds and lengths: **1 of 28** usable seeds produce a cross-component (attention-load-bearing) circuit of the kind the certified seed-5 result showcases. Per length: L=5 (step down) 0/8; L=6 (certified) 1/12; L=7 (step up) 0/8.

Read this alongside `transformer_report.md`: the certified seed is a genuine, certified example of a distributed circuit — what this survey establishes is how *representative* that structure is, so the claim is worded to exactly what the population supports.

Total time 1033.2s on a laptop CPU (training dominates; no solver calls).