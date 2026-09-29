# Certified Mechanistic Edits — anonymous artifact

Anonymous code and results artifact accompanying the submission *"Certified
Mechanistic Edits: Behavioral Guarantees for Skill Removal and Preservation."*

> **Note:** this README is an AI-generated summary of the artifact. The paper is
> the authoritative description of the method and results; every number below
> is taken from the committed reports in `results/`.

## What this is

Mechanistic edits to a network — ablating a circuit, cutting weights, adding a
steering vector, clamping a sparse-autoencoder feature — are normally validated
by *testing* them on a finite sample of inputs. This artifact **proves** what an
edit does instead: that one skill is removed and a second skill preserved, for
**every** input in a continuous region at once. Exact certificates come from an
SMT solver (Z3) over exact rational arithmetic, so a proof is either `unsat`
(the claim holds everywhere in the region) or a concrete counterexample; each is
cross-checked against a brute-force grid, and each claim also carries a
*certified radius* — the largest input perturbation the guarantee survives,
found by bisecting proofs. The experiments cover a range of subjects: a
two-input ReLU MLP, a two-hidden-layer MLP with a distributed (XOR) circuit, a
threshold-gate transformer certified over all token sequences × a continuous
embedding ball in one query, a known-formula transformer (two independent
modular adders), and — past the exact solver's frontier — a **standard
softmax + LayerNorm transformer** certified by sound bound propagation (CROWN)
with every radius bracketed above by a PGD attack. The theory side is
machine-checked too: the propositions (including the impossibility result that
no finite deterministic black-box test protocol can certify removal) have
runnable witnesses. Pre-generated reports for every experiment are committed
under `results/`, so every number in the paper can be read without running
anything.

## Repository layout

```
tiny_model.py             two-skill ReLU MLP (any input dimension) + training + circuit search
deep_model.py             2-hidden-layer model; skill A is an XOR of thresholds (distributed)
transformer_model.py      threshold-gate decoder block (exactly SMT-encodable), quantized
mod_arith_model.py        known-formula subject: two modular adders on disjoint positions
boundprop_transformer.py  standard softmax + LayerNorm transformer (bound-propagation subject)
sae_model.py              sparse autoencoder over the toy model + its feature-clamp edit
edits.py                  edit types as weight changes: ablation, weight edit, two steering recipes
verify.py                 exact SMT encoding (rationals) + prove_forall / certified_radius
                          / prove_independence / certified_influence
verify_transformer.py     transformer prover: all sequences × continuous noise in one query
                          (token_mode "bool" = the discrete claim; "hull" = a relaxation whose
                          unsat still certifies it — the tractable path used)
verify_rung3.py           p-way two-copy (siamese) prover for the known-formula subject
run_*.py                  one experiment each; every script writes its own report to results/
figures/make_figures.py   regenerates the paper's figures from the committed reports
results/                  committed reports (.md human-readable, .json machine-readable)
```

## Setup — two deliberately separate dependency stacks

**Stack 1 — the exact (SMT) pipeline.** Pure `numpy` + `z3-solver`. CPU only, no
GPU, no cloud, no accelerator. Everything except the three `run_boundprop_*.py`
scripts runs here.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy, z3-solver (+ matplotlib for figures only)
python run_slice.py
```

**Stack 2 — bound propagation (isolated).** The `run_boundprop_*.py` scripts
need `torch` and `auto_LiRPA`, which is a much heavier toolchain with awkward
version constraints. It is kept in a *separate* virtual environment on purpose,
so that the exact pipeline — where the paper's exactness claims live — stays
pure `numpy` + `z3-solver` and cannot silently depend on a deep-learning stack.
`requirements-boundprop.txt` is documentation, not an installable file: a naive
`pip install -r` fails (the PyPI `auto_LiRPA` pins an old torch), so follow the
exact ordered commands in that file's header, which install CPU torch first and
`auto_LiRPA` from source into a Python 3.12 environment:

```bash
# see requirements-boundprop.txt for the full ordered recipe
uv venv --python 3.12 .venv-boundprop
uv pip install --python .venv-boundprop/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv-boundprop/bin/python numpy z3-solver
uv pip install --python .venv-boundprop/bin/python "git+https://github.com/Verified-Intelligence/auto_LiRPA.git"
.venv-boundprop/bin/python run_boundprop_validate.py
```

## Claim → command → report

Every number in the paper is regenerated by one command and is already
committed as a report. Section numbers refer to the paper. `$PY` below is `python` from Stack 1;
`$BP` is `.venv-boundprop/bin/python` from Stack 2. Runtimes are single-core
CPU wall-clock, taken from the committed reports where those record a total and
approximate otherwise.

### §V-A — an edit's behavioral effect proved over a whole input region

| Paper claim | Command | Report | Runtime |
|---|---|---|---|
| Softmax+LayerNorm transformer, 448 noise dims (≈9× the exact frontier): edit drives skill A to chance and keeps skill B at 100%; certified ≤ true ≤ attack for removal `0.0091 ≤ 0.03`, preservation `0.0079 ≤ 0.05` and `0.0098 ≤ 0.1` | `$BP run_boundprop_transformer.py` | `results/boundprop_transformer_report.md` | ~9 min |
| Exact certificate on the threshold-gate transformer over **all** sequences × a continuous embedding ball: removal radius `0.016`, preservation `0.013`; collateral quantified (skill B's radius `0.034 → 0.013`) | `$PY run_transformer.py` | `results/transformer_report.md` | ~37 min |
| Known-formula subject: removal certified as exact independence of the summands, radius `≥ 0.05`; unedited control correctly refuted | `$PY run_rung3.py` | `results/rung3_report.md` | ~3 min |
| Known-formula subject, the other side: exact-rational correctness on **all 625** clean sequences and noise-robust preservation to radius `≥ 0.002` | `$PY run_rung3_preservation.py` | `results/rung3_preservation_report.md` | ~10 min |

### §V-B — testing provably cannot deliver the guarantee

| Paper claim | Command | Report | Runtime |
|---|---|---|---|
| Constructed edit passes a `40,401`-point grid **with preservation proved**, yet the solver refutes removal with a concrete surviving input | `$PY run_illusion.py` | `results/illusion_report.md` | ~20 min |
| **Three naturally-trained illusions in 160 models** (4–5 inputs); each passed a full grid + 1,000 random inputs + preservation checks; survivor pockets: 0 hits in `2×10⁶` darts, 95% Clopper–Pearson ceiling `1.5×10⁻⁶` of the domain | `$PY run_illusion_nd.py` | `results/illusion_nd_report.md` | ~30 min |
| Illusions **per input dimension** (Fig. 1c): the same search run one dimension at a time, 80 models each — `0/49`, `0/100`, `2/92`, `1/106` (illusions / test-approved edits) for 2, 3, 4, 5 inputs; the 4- and 5-input rows reproduce the pooled 160-model run exactly | `$PY run_illusion_dims.py` | `results/illusion_dims_report.md` | ~28 min on 4 cores (`--dim` + `--merge`); ~95 min in sequence |

### §V-C — feature non-interference (certified influence)

| Paper claim | Command | Report | Runtime |
|---|---|---|---|
| On the transformer the certified influence of the forbidden content on the readout reaches **exactly 0**, robust to radius `≥ 0.05` — a *larger* radius than removal's own `0.016` | `$PY run_transformer_independence.py` | `results/transformer_independence_report.md` | ~2 min |
| Toy subject: influence `22 → <0.001`; entangled "messy" model: removal certifies yet influence only `63 → 10` (removal is not deafness); the illusion edit is refuted with a concrete input **pair**, influence ceiling `0.60` | `$PY run_independence.py` | `results/independence_report.md` | ~18 min |

### §V-D — certified radius and how sharp the claim can be

| Paper claim | Command | Report | Runtime |
|---|---|---|---|
| Certified edges reach within `0.0012` of the `0.5` rule for skill A and within `0.0039` at worst (Fig. 5) | `$PY run_margin_curve.py` | `results/margin_curve_report.md` | ~1.5 min |
| Certified radii per edit type, both subjects (the radius column referenced throughout) | `$PY run_robustness.py` | `results/robustness_report.md` | ~1.5 min |

### §V-E — surgery vs. steering, with certified collateral

| Paper claim | Command | Report | Runtime |
|---|---|---|---|
| Surgical edits certify removal at maximal radius; diff-of-means steering erodes the preserved margin on the separated model: `0.0996 → 0.0908` (dose 4) `→ 0.0656` (dose 16), until preservation breaks outright (Fig. 6a) | `$PY run_robustness.py` | `results/robustness_report.md` | ~1.5 min |
| The dose→collateral claim rests on a **full dose sweep (0.5–64)**, not two points (Fig. 6b) | `$PY run_collateral_sweep.py` | `results/collateral_sweep.json` | ~2.5 min |
| On the transformer, **no dose both removes A and preserves B** (certified rows per dose) | `$PY run_transformer_steering.py` | `results/transformer_steering_report.md` | ~64 min |
| An SAE feature clamp (skill A splits across three features) is certified by the same prover: removal `≥ 0.5`, preservation `0.064` against ablation's `0.100` | `$PY run_sae_clamp.py` | `results/sae_clamp_report.md` | ~1 min |
| Full edit-type × certified-radius table at transformer scale | `$PY run_transformer_edit_table.py` | `results/transformer_edit_table_report.md` | ~81 min |
| The same edit-type table on the known-formula subject | `$PY run_rung3_edit_table.py` | `results/rung3_edit_table_report.md` | ~4 min |

### §V-F — the exact frontier and its sound extension

| Paper claim | Command | Report | Runtime |
|---|---|---|---|
| The exact frontier is set by internal branching: the size ladder (Table III) and the Boolean/hull timeouts; the circuit search behind Fig. 3 | `$PY results/rung2_size_ladder.py`, `$PY results/rung2_hull_wall.py`, `$PY results/rung2_circuit_search.py` | `results/rung2_size_ladder.log`, `results/rung2_hull_wall.log`, `results/rung2_circuit_search.log` | ~35 min, ~2 h, ~10 min |
| Bound propagation **equals** the exact radius where both tools apply (the M0 soundness-and-tightness gate) | `$BP run_boundprop_validate.py` | `results/boundprop_validate_report.md` | ~2 min |
| A single token swap moves the embedding `≈324×` the certified ball, so the continuous guarantee is not disguised discrete-prompt robustness | `$BP run_boundprop_quantifier.py` | `results/boundprop_quantifier_report.md` | ~10 min |

### Framework, appendices and Table II

| Paper claim | Command | Report | Runtime |
|---|---|---|---|
| Machine-checked witnesses for P2–P4 and Corollary 1 (adaptive-tester transcript replay), App. E | `$PY run_theory_checks.py` | `results/theory_checks_report.md` | ~40 s |
| Deep MLP (Table II): the forced multi-neuron (layer-1-only) edit, fully certified | `$PY run_deep_multi.py` | `results/deep_multi_report.md` | ~38 min |
| App. B, the float gap: rigorous forward-error ceiling (`≈1.1×10⁻¹³` on the toy model) and the toy model's removal, preservation and control certificates re-proved with slack `10⁻⁹` (the threshold-gate transformer's certificates carry slack `10⁻⁶`) | `$PY run_float_gap.py` | `results/float_gap_report.md` | ~10 s |

### Supplementary experiments (not cited in the paper)

| What it shows | Command | Report | Runtime |
|---|---|---|---|
| How typical the certified circuit is across seeds / sizes (is the separable circuit a one-off?) | `$PY run_transformer_survey.py`, `$PY run_transformer_size_survey.py` | `results/transformer_survey_report.md`, `results/transformer_size_survey_report.md` | ~17 min, ~45 min |
| The vertical slice: removal + preservation proved on the toy MLP, each cross-checked by grid | `$PY run_slice.py` | `results/slice_report.md` | ~1 min |
| Deep (2-layer) subject: the free-search baseline, and the seed survey showing single-neuron circuits are the exception | `$PY run_deep.py`, `$PY run_deep_survey.py` | `results/deep_report.md`, `results/deep_survey_report.md` | ~46 min, ~7 min |
| Positioning: a *descriptive* certificate of the neighboring kind certifies the illusion edit, while the behavioral-over-a-region certificate refutes it | `$PY run_faithfulness_gap.py` | `results/faithfulness_gap_report.md` | ~1 min |
| Capacity check: the exact-encodable architecture learns modular addition at a certifiable size | `$PY run_rung3_spike.py` | `results/rung3_spike_report.md` | ~1.5 min |

### Figures

The paper's figures are rebuilt from the committed reports:

```bash
python figures/make_figures.py --paper        # every paper figure
python figures/make_figures.py --paper bp     # one of them
```

| key | output | what it shows | built from |
|---|---|---|---|
| `illusion` | `fig_illusion.pdf` | an edit every test approves and the solver refutes; illusions per input dimension | `illusion_report.json`, `illusion_nd_report.json`, `illusion_dims_report.json` |
| `bp` | `fig_bp.pdf` | bound-propagation radii on the softmax transformer, each bracketed above by an attack | `boundprop_transformer_report.json` |
| `circuit` | `fig_circuit.pdf` | the distributed skill-A circuit of the exact transformer subject | `transformer_report.json` |
| `influence` | `fig_influence.pdf` | the two-copy query that certifies influence (feature non-interference) | the encoding in `verify.py::prove_independence` |
| `edits` | `fig_edits.pdf` | skill B's certified radius per edit type, and against steering dose | `robustness_report.json`, `collateral_sweep.json` |
| `radius` | `fig_radius.pdf` | the certified edge against the edited model's own decision boundary (entangled model, ablation); retrains the subject, ~15 s | `robustness_report.json` |

They are written to `paper/figures/fig_<role>.pdf`, authored at their final
printed size (3.45 in for a column, 7.05 in for a full-width figure) in the
paper's typefaces. If PyMuPDF is installed, each build also measures its printed
text sizes and fails on any label below the floor; without PyMuPDF that check is
skipped. Rebuilds are byte-identical.

`make_figures.py` without `--paper` builds the on-screen variants instead: wider,
self-titled versions of the same results. With no arguments it builds all of
them; a few of the exploratory ones retrain their subject first and so take
minutes.

## Notes on reproducibility

- **Fixed seeds everywhere.** Re-running a script reproduces its committed report.
- **A timed-out solver probe is treated as a failure**, never as a proof, so
  every reported radius is a proved lower bound. Because near-boundary probes
  can time out non-deterministically on a loaded machine, *natural* (unedited)
  radii may come back slightly smaller on a slow run; the edited-model radii the
  paper quotes are stable.
- **Runtimes** above are single-core CPU wall-clock from the committed reports.
  Total cost of the full suite is several hours, dominated by
  `run_transformer_edit_table.py`, `run_transformer_steering.py`,
  `results/rung2_hull_wall.py` (a deliberate 2×30-minute timeout measurement)
  and `run_transformer.py`. The quickest meaningful end-to-end check is
  `run_slice.py` followed by `run_theory_checks.py` (under two minutes total).
- **Reports are both `.md` and `.json`.** The `.md` file is written to be read on
  its own; the `.json` holds the same numbers for the figure scripts.
- **Self-contained.** Comments and reports point only at files in this
  artifact or at sections of the paper.
- **Timing-sensitive scripts.** `run_rung3_preservation.py` gives each solver
  query 20 s, the transformer scripts 120–300 s. On a slower or heavily loaded
  machine a query can time out, which is counted as a failure and lowers a
  radius; raise the per-query budget (for example `PER_QUERY_MS`) if that
  happens. It changes only how long the solver may search, never what it
  proves. The Table III solver times depend on the machine; its verdicts do not.

## Scope and limitations (matching the paper)

- **Small models.** The subjects are deliberately small: tens of neurons for the
  MLPs, and for the exact transformer certificate `d_model 8`, 2 heads,
  sequence length 6. This is a *measured* frontier, not an arbitrary choice: the
  exact encoding's cost is exponential in the model's internal case splits, and
  exceeds the time budget at every larger size we measured
  (`results/rung2_size_ladder.log`).
  Exact input-side certification also forced a threshold-gate attention variant,
  because soft attention's application is bilinear in the perturbation.
- **Embedding-space regions, not prompt-edit distances.** The continuous axis of
  every guarantee is noise in embedding space. In the exact cases the certificate
  covers every discrete prompt in the model's input space *and* a continuous
  neighborhood of each — a superset — but a certified radius is still not a
  discrete prompt-edit distance. We measure the gap rather than gloss it: the
  nearest in-vocabulary token swap is `≈324×` the certified ball.
- **The decidable-skill assumption.** Every skill certified here has a decidable
  ground truth (a computable property such as "more `«` than `»`", or
  `(a+b) mod p`). That is exactly what makes "removed" a crisp, checkable claim —
  and it does not transfer to real harmful capabilities for free, since whether
  an output "exhibits a harmful skill" is generally not decidable. This artifact
  is a proof of concept, **not** a certificate over a real dangerous capability,
  and nothing here is a frontier-scale or real-harm guarantee.
- **Bound propagation is sound, not exact.** The CROWN results over-approximate:
  a certified radius is a lower bound on the true one, so a wide gap to the
  attack line means prover looseness, never unsoundness (an attack succeeding
  *below* a certified radius would signal a bug; none does). Those radii are
  reported over balls of radius `≥ 1e-4`, because CROWN's softmax/LayerNorm
  relaxation is degenerate exactly at `ε = 0`.
- **The softmax-model edit is a position path-patch, not a component ablation.**
  A from-scratch standard softmax transformer does not organise the two skills
  into separable heads or neurons — greedy component search finds no single atom
  that removes A while preserving B — so that edit zeroes the value contribution
  of the relevant positions instead.
- **What the radius does not cover.** The certificates fix the edited weights and
  quantify over inputs. Attacks that change weights (for example fine-tuning
  recovery) are outside what any certificate here promises.
- **Exact-rational vs. float.** Exact certificates describe the ideal
  rational-arithmetic network, while the code executes float64. The gap is
  bounded and the transfer argument is run, not asserted
  (`results/float_gap_report.md`): the forward-error ceiling is `≈1.1×10⁻¹³`, the toy
  model's removal, preservation and control certificates re-prove with slack
  `10⁻⁹`, and the threshold-gate transformer's certificates carry slack `10⁻⁶`. A *zero-margin* claim does not
  transfer, and the report demonstrates exactly that boundary case rather than
  claiming the trap is simply avoided.

## License

MIT, with the copyright holder withheld for double-blind review (see `LICENSE`);
attribution will be restored in the public release.
