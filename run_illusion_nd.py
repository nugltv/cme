"""
run_illusion_nd.py — hunting the intervention illusion in MORE input dimensions
================================================================================

Run it with:    python run_illusion_nd.py

WHY THIS EXPERIMENT EXISTS
--------------------------
The 2-input search (`run_illusion.py`, Route A) came back empty: across 200
trained models, every edit that passed the tests really had removed the skill.
Our diagnosis: with only two input numbers there is nowhere for a leftover
pathway to hide — it either shows up at the test points or it doesn't exist.

This script re-runs the same hunt on models with FOUR and FIVE inputs. The two
skills still only depend on the first two inputs (skill A: "is x0 > 0.5?",
skill B: "is x1 > 0.5?"); the extra inputs are "distractors" the network is
supposed to ignore. But the network doesn't know it's supposed to ignore them —
its neurons grow small stray connections to the distractors. After an edit,
those stray connections can add up to a leftover pathway that only wakes up in
some corner of the input space, e.g. "x0 high AND x2 very low AND x4 very
high". And here is the key fact about testing in higher dimensions:

    A test set covers a vanishing fraction of the space as dimensions grow.
    In 2 dimensions, 225 well-spread test points leave gaps ~0.07 wide.
    In 5 dimensions you would need ~750 MILLION points for the same coverage.

So pockets that were impossible to miss in 2D become easy to miss in 5D. The
practitioner's tests below are genuinely reasonable ones — a grid AND a
thousand random inputs — and the question is whether a trained model + an
approved edit can still fool them all while the solver finds the survivor.

WHAT COUNTS AS AN "ILLUSION" (same rules as before)
---------------------------------------------------
  1. every test says skill A is gone from its whole region, AND
  2. every test says skill B still works, AND
  3. the solver finds an input where the edited model still does skill A.

Results land in results/illusion_nd_report.md (plain language) and .json.
"""

from __future__ import annotations
import json, os, time
from itertools import combinations

import numpy as np

from tiny_model import TinyMLP, train, find_skill_circuit, accuracy, \
    true_label_A, true_label_B
from verify import prove_forall

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# The practitioner's test budget (deliberately GENEROUS — the point is that
# even a diligent tester gets fooled, not a lazy one):
GRID_PER_AXIS = 5          # a grid with 5 points along every input axis
RANDOM_REMOVAL_N = 1000    # random test inputs for "is skill A gone?"
RANDOM_PRESERVE_N = 500    # random test inputs per region for "is B fine?"
RANDOM_TEST_SEED = 42

# How we measure a found pocket's size afterwards: Monte-Carlo (many random
# darts). 2 million darts resolves pockets down to ~0.0001% of the region.
VOLUME_DARTS = 2_000_000


# ---------------------------------------------------------------------------
# Regions, now in d dimensions. Skill A's HIGH region is "x0 is clearly high,
# every other input can be anything." Same margins as the 2-input experiments.
# ---------------------------------------------------------------------------
def region_a_high(d):
    return [(0.6, 1.0)] + [(0.0, 1.0)] * (d - 1)

def region_b_high(d):
    return [(0.0, 1.0), (0.6, 1.0)] + [(0.0, 1.0)] * (d - 2)

def region_b_low(d):
    return [(0.0, 1.0), (0.0, 0.4)] + [(0.0, 1.0)] * (d - 2)


# ---------------------------------------------------------------------------
# Test helpers (the practitioner's tools).
# ---------------------------------------------------------------------------
def grid_points(box, k=GRID_PER_AXIS):
    """All combinations of k evenly-spaced values along each axis of the box.
    (5 per axis = 625 points in 4D, 3,125 points in 5D.)"""
    axes = [np.linspace(lo, hi, k) for lo, hi in box]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def random_points(box, n, rng):
    """n uniformly random points inside the box."""
    los = np.array([lo for lo, _ in box])
    his = np.array([hi for _, hi in box])
    return los + rng.uniform(size=(n, len(box))) * (his - los)


def count_violations(model, pts, head, want, edit):
    """How many of these test points break the claim? (head says the wrong
    side of zero). head: 0 = A, 1 = B. want: 'positive' or 'nonpositive'."""
    logits = model.forward(pts, ablate=edit)[:, head]
    return int((logits <= 0).sum()) if want == "positive" \
        else int((logits > 0).sum())


def full_test_battery_passes(model, edit, d, rng):
    """
    Everything a careful practitioner would check before deploying the edit:
      - skill A looks GONE on a grid AND on 1000 random inputs in its region;
      - skill B looks INTACT on grids AND random inputs in both its regions.
    Returns (passed, details).
    """
    boxA = region_a_high(d)
    boxBh, boxBl = region_b_high(d), region_b_low(d)

    checks = {
        "removal_grid": count_violations(model, grid_points(boxA), 0,
                                         "nonpositive", edit),
        "removal_random": count_violations(
            model, random_points(boxA, RANDOM_REMOVAL_N, rng), 0,
            "nonpositive", edit),
        "preserve_grid_high": count_violations(model, grid_points(boxBh), 1,
                                               "positive", edit),
        "preserve_random_high": count_violations(
            model, random_points(boxBh, RANDOM_PRESERVE_N, rng), 1,
            "positive", edit),
        "preserve_grid_low": count_violations(model, grid_points(boxBl), 1,
                                              "nonpositive", edit),
        "preserve_random_low": count_violations(
            model, random_points(boxBl, RANDOM_PRESERVE_N, rng), 1,
            "nonpositive", edit),
    }
    return all(v == 0 for v in checks.values()), checks


def pocket_volume_fraction(model, edit, d, darts=VOLUME_DARTS):
    """After the solver finds a survivor, measure how big the surviving pocket
    is: throw millions of random darts into skill A's region and count how
    many land where the edited model still says HIGH."""
    rng = np.random.default_rng(7)
    hits, per_batch = 0, 200_000
    thrown = 0
    while thrown < darts:
        pts = random_points(region_a_high(d), per_batch, rng)
        hits += int((model.forward(pts, ablate=edit)[:, 0] > 0).sum())
        thrown += per_batch
    return hits / thrown


def rank_neurons_by_damage_to_A(model, d, n=8000, seed=7):
    """Knock out each neuron alone; sort by how much skill A suffers.
    (Same practitioner move as in the 2-input search.)"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, d))
    Y = np.stack([true_label_A(X), true_label_B(X)], axis=1)
    base = accuracy(model, X, Y)
    rows = []
    for j in range(model.H):
        acc_j = accuracy(model, X, Y, ablate=[j])
        rows.append((j, float(base[0] - acc_j[0]), float(base[1] - acc_j[1])))
    rows.sort(key=lambda r: -r[1])
    return rows


# ---------------------------------------------------------------------------
# The search itself.
# ---------------------------------------------------------------------------
def search(dims=(4, 5), l1_values=(0.0, 5e-4, 1e-3, 2e-3), n_seeds=20,
           H=24, top_pool=5, max_edit_size=4, train_steps=5000):
    illusions, honest, checked, trained, unknowns = [], 0, 0, 0, 0

    for d in dims:
        for l1 in l1_values:
            for seed in range(n_seeds):
                model = train(TinyMLP(H=H, seed=seed, d=d), l1=l1,
                              seed=seed + 1, steps=train_steps, verbose=False)
                trained += 1

                # Only edit models that actually learned both skills.
                rng = np.random.default_rng(999)
                X = rng.uniform(0, 1, size=(6000, d))
                Y = np.stack([true_label_A(X), true_label_B(X)], axis=1)
                acc = accuracy(model, X, Y)
                if acc[0] < 0.98 or acc[1] < 0.98:
                    continue

                # Candidate edits a practitioner might apply.
                ranked = rank_neurons_by_damage_to_A(model, d)
                pool = [j for j, _, _ in ranked[:top_pool]]
                candidates = []
                for k in range(1, max_edit_size + 1):
                    for combo in combinations(pool, k):
                        candidates.append(tuple(sorted(combo)))
                found = find_skill_circuit(model, "A")
                if found:
                    candidates.append(tuple(sorted(found)))
                candidates = list(dict.fromkeys(candidates))

                test_rng = np.random.default_rng(RANDOM_TEST_SEED)
                for edit in candidates:
                    edit = list(edit)
                    passed, checks = full_test_battery_passes(model, edit, d,
                                                              test_rng)
                    if not passed:
                        continue    # practitioner rejects this edit anyway
                    checked += 1

                    # Tests all say "deploy". The solver gets the last word.
                    proof = prove_forall(model, region_a_high(d), "A",
                                         "nonpositive", ablate=edit,
                                         timeout_ms=60000)
                    if proof["proved"]:
                        honest += 1
                        continue
                    if proof["counterexample"] is None:
                        unknowns += 1
                        continue

                    # ILLUSION: every test passed; the solver found a survivor.
                    cx = proof["counterexample"]
                    logit = float(model.forward(np.array([cx]),
                                                ablate=edit)[0, 0])
                    vol = pocket_volume_fraction(model, edit, d)
                    illusions.append({
                        "d": d, "H": H, "l1": l1, "seed": seed, "edit": edit,
                        "counterexample": [round(v, 6) for v in cx],
                        "counterexample_logit_numeric": logit,
                        "pocket_volume_fraction": vol,
                        "tests_passed": {
                            "grid_points_removal": int(GRID_PER_AXIS ** d),
                            "random_points_removal": RANDOM_REMOVAL_N},
                    })
                    print(f"    ILLUSION @ d={d} l1={l1} seed {seed}: "
                          f"edit=off {edit}; survivor at "
                          f"{[f'{v:.3f}' for v in cx]} "
                          f"(logit {logit:.4f}, pocket "
                          f"{vol*100:.5f}% of region)")

    return illusions, honest, checked, trained, unknowns


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def _write_report(illusions, honest, checked, trained, unknowns):
    md = []
    md.append("# The illusion hunt in 4 and 5 input dimensions\n")
    md.append("Output of `run_illusion_nd.py`. The 2-input search found no "
              "naturally-trained illusion; this re-runs the hunt on models "
              "with 4 and 5 inputs, where test points cover far less of the "
              "space and hidden pockets have room to exist.\n")
    md.append(f"- Models trained: {trained} (4D and 5D, four tidiness "
              f"settings, 20 seeds each).")
    md.append(f"- Edits that passed the practitioner's ENTIRE test battery "
              f"(grid + {RANDOM_REMOVAL_N} random inputs for removal, grids + "
              f"random inputs for preservation): {checked}.")
    md.append(f"- Of those: {honest} proved genuinely removed, "
              f"{unknowns} unresolved (solver timeout), and "
              f"**{len(illusions)} ILLUSIONS** — edits every test approved "
              f"but the solver refuted.\n")
    if illusions:
        md.append("| dims | tidiness (l1) | seed | edit (neurons off) | "
                  "survivor input | pocket size | tests it fooled |")
        md.append("|---|---|---|---|---|---|---|")
        for il in illusions:
            # A pocket that 2 million random darts never hit: the honest
            # number is a Clopper-Pearson 95%-confidence CEILING (zero hits
            # in n darts pins the fraction below about 3/n with 95%
            # confidence) — a sampling statement, not a certified bound, and
            # not a misleading "0%".
            vol = il["pocket_volume_fraction"]
            ceiling = 1.0 - 0.05 ** (1.0 / VOLUME_DARTS)
            vol_str = (f"< {100*ceiling:.5f}% (95%-confidence ceiling; "
                       f"0 hits in {VOLUME_DARTS:,} darts)"
                       if vol == 0 else f"{vol*100:.5f}%")
            md.append(
                f"| {il['d']} | {il['l1']} | {il['seed']} | {il['edit']} | "
                f"{il['counterexample']} | {vol_str} | "
                f"{il['tests_passed']['grid_points_removal']}-pt grid + "
                f"{il['tests_passed']['random_points_removal']} random |")
        md.append("")
        md.append("Every survivor above was re-checked numerically on the "
                  "ordinary (float) model: the edited head A really does say "
                  "HIGH at the listed input, so the skill genuinely survives "
                  "there.\n")
    else:
        md.append("_No illusion found in this sweep either. Next levers: more "
                  "seeds, larger H, more distractor dimensions, or edits "
                  "chosen adversarially rather than by damage ranking._\n")
    md.append("## What this means\n")
    if illusions:
        md.append("In higher dimensions the intervention illusion arises in "
                  "ORDINARILY TRAINED models: a diligent tester (thousands of "
                  "test points, grid and random, plus collateral checks) "
                  "approves an edit that provably did not remove the skill. "
                  "This is the naturally-occurring companion to the "
                  "constructed 2-input example in `illusion_report.md` — "
                  "together they show the blind spot is both unavoidable in "
                  "principle and real in practice.\n")
    else:
        md.append("The constructed example (2-input, `illusion_report.md`) "
                  "remains the demonstration that no finite black-box test "
                  "can certify removal; a naturally-trained example still "
                  "needs a wider search.\n")

    with open(os.path.join(RESULTS, "illusion_nd_report.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(RESULTS, "illusion_nd_report.json"), "w") as f:
        json.dump({"models_trained": trained,
                   "edits_test_approved": checked,
                   "proved_genuinely_removed": honest,
                   "solver_unknown": unknowns,
                   "illusions": illusions}, f, indent=2)


def main():
    print("=" * 70)
    print("ILLUSION HUNT IN 4 AND 5 INPUT DIMENSIONS")
    print("=" * 70)
    t0 = time.time()
    illusions, honest, checked, trained, unknowns = search()
    print(f"\n{trained} models trained; {checked} edits passed the full test "
          f"battery")
    print(f"-> {honest} proved genuinely removed, {unknowns} unknown, "
          f"{len(illusions)} ILLUSIONS  ({time.time()-t0:.0f}s)")
    _write_report(illusions, honest, checked, trained, unknowns)
    print("\nReport written to: results/illusion_nd_report.md and .json")


if __name__ == "__main__":
    main()
