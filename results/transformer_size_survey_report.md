# Size survey: is a distributed circuit an architecture property?

Output of `run_transformer_size_survey.py` — numeric only (the same test gate `run_transformer.py` uses: all sequences enumerated + sampled embedding noise; NO certificates). This extends `run_transformer_survey.py` (a seed sweep at ONE shape) across the ARCHITECTURE ladder, to see whether the frequency of cross-component (attention-load-bearing) circuits is a property of the shape rather than a one-off. Each model is the exact certified artifact (`train_gate_subject`: train -> harden -> fine-tune -> quantize).

Sequence length fixed at L=6; 10 seeds per config; each config keeps n_heads x d_head = d_model.

**Circuit classes.** *cross-component* = >= 1 attention head AND >= 1 MLP neuron (attention genuinely load-bearing — the certified seed's kind); *single head* / *single mlp* = exactly one atom; *multi heads* / *multi mlp* = several atoms of one type; *none* = no test-passing circuit.

| config | usable | cross-component | breakdown |
|---|---|---|---|
| certified         d8/h2/mlp8 | 10 | **1** | cross-component: 1, multi mlp: 2, none: 4, single head: 3 |
| wider model     d12/h2/mlp8 | 10 | **2** | cross-component: 2, none: 8 |
| narrower model  d6/h2/mlp8 | 10 | **0** | multi mlp: 1, none: 6, single head: 1, single mlp: 2 |
| one head        d8/h1/mlp8 | 9 | **0** | none: 9 |
| more heads      d8/h4/mlp8 | 10 | **4** | cross-component: 4, multi heads: 2, none: 4 |
| wider mlp       d8/h2/mlp16 | 10 | **1** | cross-component: 1, none: 6, single head: 2, single mlp: 1 |
| narrow mlp      d8/h2/mlp4 | 10 | **2** | cross-component: 2, none: 7, single head: 1 |

## Per-config detail

### certified         d8/h2/mlp8  (config {'d_model': 8, 'n_heads': 2, 'd_head': 4, 'd_mlp': 8})

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

### wider model     d12/h2/mlp8  (config {'d_model': 12, 'n_heads': 2, 'd_head': 6, 'd_mlp': 8})

| seed | usable? | circuit | class |
|---|---|---|---|
| 0 | yes | None | none |
| 1 | yes | None | none |
| 2 | yes | None | none |
| 3 | yes | None | none |
| 4 | yes | [('head', 0), ('mlp', 3)] | cross-component |
| 5 | yes | [('head', 1), ('mlp', 5)] | cross-component |
| 6 | yes | None | none |
| 7 | yes | None | none |
| 8 | yes | None | none |
| 9 | yes | None | none |

### narrower model  d6/h2/mlp8  (config {'d_model': 6, 'n_heads': 2, 'd_head': 3, 'd_mlp': 8})

| seed | usable? | circuit | class |
|---|---|---|---|
| 0 | yes | None | none |
| 1 | yes | [('mlp', 2)] | single mlp |
| 2 | yes | [('head', 0)] | single head |
| 3 | yes | None | none |
| 4 | yes | None | none |
| 5 | yes | None | none |
| 6 | yes | [('mlp', 0), ('mlp', 5)] | multi mlp |
| 7 | yes | [('mlp', 3)] | single mlp |
| 8 | yes | None | none |
| 9 | yes | None | none |

### one head        d8/h1/mlp8  (config {'d_model': 8, 'n_heads': 1, 'd_head': 8, 'd_mlp': 8})

| seed | usable? | circuit | class |
|---|---|---|---|
| 0 | yes | None | none |
| 1 | yes | None | none |
| 2 | under-trained (A 1.000/B 0.840) | — | — |
| 3 | yes | None | none |
| 4 | yes | None | none |
| 5 | yes | None | none |
| 6 | yes | None | none |
| 7 | yes | None | none |
| 8 | yes | None | none |
| 9 | yes | None | none |

### more heads      d8/h4/mlp8  (config {'d_model': 8, 'n_heads': 4, 'd_head': 2, 'd_mlp': 8})

| seed | usable? | circuit | class |
|---|---|---|---|
| 0 | yes | None | none |
| 1 | yes | [('mlp', 0), ('head', 3)] | cross-component |
| 2 | yes | [('head', 1), ('mlp', 1), ('mlp', 7)] | cross-component |
| 3 | yes | [('head', 1), ('head', 0)] | multi heads |
| 4 | yes | None | none |
| 5 | yes | [('head', 0), ('head', 1), ('mlp', 1), ('mlp', 2)] | cross-component |
| 6 | yes | [('head', 3), ('head', 0)] | multi heads |
| 7 | yes | [('mlp', 6), ('head', 0)] | cross-component |
| 8 | yes | None | none |
| 9 | yes | None | none |

### wider mlp       d8/h2/mlp16  (config {'d_model': 8, 'n_heads': 2, 'd_head': 4, 'd_mlp': 16})

| seed | usable? | circuit | class |
|---|---|---|---|
| 0 | yes | None | none |
| 1 | yes | [('head', 1)] | single head |
| 2 | yes | [('mlp', 5)] | single mlp |
| 3 | yes | None | none |
| 4 | yes | None | none |
| 5 | yes | None | none |
| 6 | yes | None | none |
| 7 | yes | [('head', 0)] | single head |
| 8 | yes | [('head', 1), ('mlp', 2)] | cross-component |
| 9 | yes | None | none |

### narrow mlp      d8/h2/mlp4  (config {'d_model': 8, 'n_heads': 2, 'd_head': 4, 'd_mlp': 4})

| seed | usable? | circuit | class |
|---|---|---|---|
| 0 | yes | None | none |
| 1 | yes | [('mlp', 3), ('head', 1)] | cross-component |
| 2 | yes | None | none |
| 3 | yes | [('head', 0), ('mlp', 0)] | cross-component |
| 4 | yes | None | none |
| 5 | yes | None | none |
| 6 | yes | [('head', 1)] | single head |
| 7 | yes | None | none |
| 8 | yes | None | none |
| 9 | yes | None | none |

## The population picture

Across the whole ladder: **10 of 69** usable seeds produce a cross-component (attention-load-bearing) circuit. Per config: certified 1/10; wider 2/10; narrower 0/10; one 0/9; more 4/10; wider 1/10; narrow 2/10.

The question this answers is whether the certified seed-5 distributed circuit is a lucky exception or a predictable consequence of the architecture. Compare the columns: if cross-component frequency rises with head count or residual width, the distributed structure is a shape property; if it stays rare everywhere, the supported wording is that a distributed, attention-load-bearing circuit CAN occur and be certified on this architecture (the seed-sweep finding, now shown robust across shapes). This is population context for the certified result in `transformer_report.md`.

Total time 2673.2s on a laptop CPU (training dominates; no solver calls).