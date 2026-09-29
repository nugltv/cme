# The illusion search, by input dimension

Output of `run_illusion_dims.py`: the search protocol of `run_illusion_nd.py`, run one input dimension at a time with identical seeds (4 tidiness settings x 20 seeds per dimension).

| inputs | models trained | test-approved edits | proved removed | solver unknown | illusions |
|---|---|---|---|---|---|
| 2 | 80 | 49 | 49 | 0 | **0** |
| 3 | 80 | 100 | 100 | 0 | **0** |
| 4 | 80 | 92 | 90 | 0 | **2** |
| 5 | 80 | 106 | 105 | 0 | **1** |

**Reproduction check.** The d = 4 and d = 5 rows, pooled, reproduce the committed `illusion_nd_report.json` exactly (models, approved edits, proved removals, and the identity of every illusion).
