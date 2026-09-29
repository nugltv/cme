# Deep-model circuit survey: how common is the single-neuron funnel?

Output of `run_deep_survey.py` — numeric only (the practitioner's deployment tests; no proofs). Context for `deep_report.md` (whose subject's free search collapses onto one bottleneck neuron) and `deep_multi_report.md` (which certifies forced multi-neuron edits on that same subject).

| seed | usable? | free-search circuit | layer-1-only (greedy) | layer-1-only (exhaustive, smallest) |
|---|---|---|---|---|
| 0 | yes | None | None | None |
| 1 | yes | [(2, 6)] | None | [(1, 1), (1, 3), (1, 10), (1, 11)] |
| 2 | yes | [(1, 1), (2, 6), (1, 5)] | [(1, 1), (1, 11), (1, 10), (1, 5), (1, 0)] | [(1, 0), (1, 1), (1, 5), (1, 10)] |
| 3 | under-trained | — | — | — |
| 4 | yes | [(2, 7), (2, 2), (2, 6), (1, 7), (1, 5), (1, 8)] | [(1, 5), (1, 7), (1, 8), (1, 3)] | [(1, 3), (1, 5), (1, 7), (1, 8)] |
| 5 | yes | [(2, 0), (2, 3)] | None | None |
| 6 | yes | [(1, 6), (2, 2), (1, 9), (2, 1)] | None | None |
| 7 | under-trained | — | — | — |
| 8 | yes | [(2, 2), (1, 9), (1, 2)] | [(1, 9), (1, 2), (1, 3)] | [(1, 2), (1, 3), (1, 9)] |
| 9 | under-trained | — | — | — |

**The population picture (7 well-trained seeds):** 6 admit a test-passing circuit at all; of those, only 1 collapse to a single neuron while 4 are naturally CROSS-LAYER; and 4 admit a layer-1-only circuit of at most 5 neurons (multi-neuron by necessity — the XOR cannot live in one). The single-neuron funnel of `deep_report.md` is the exception in this population, not the rule. Note also that greedy misses some layer-1 circuits that exhaustive search finds: the XOR pieces cancel in pairs, so single knockouts look unhelpful — the same 'damage is not removal' family of traps documented in the paper's appendix.

Total time 419.7s on a laptop CPU (training dominates; no solver calls).