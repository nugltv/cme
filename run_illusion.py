"""
run_illusion.py — "the intervention illusion" (paper Fig. 1a,b)
================================================================

Run it with:    python run_illusion.py

THE IDEA, IN PLAIN WORDS
------------------------
run_slice.py shows we can PROVE what an edit does. This experiment shows why that
matters: it builds a situation where ordinary TESTING says an edit worked, but
the solver PROVES it didn't — and hands us the exact input where the "removed"
skill is still alive.

Picture it like checking a field for weeds by walking along straight rows.
If a thin strip of weeds grows BETWEEN two rows, you can walk the whole field
and see nothing. Testing a model on a grid (or list) of inputs has the same
blind spot: a narrow strip of inputs where the skill survives can hide between
the points you tested. The solver doesn't walk rows — it reasons about every
point at once, so it cannot miss the strip.

We demonstrate this in TWO ways:

ROUTE A — "search":  train many models the messy way (no tidiness penalty, so
    skill A gets smeared across several neurons), switch off only the neurons
    an interpretability researcher would identify, and check: does the edit
    LOOK complete on a test set, while the solver finds a surviving input?
    This shows the illusion can arise naturally in trained models.

ROUTE B — "construct": build a small network BY HAND whose skill A survives
    (after the edit) only in a sliver of inputs deliberately placed between
    the test grid's lines. This is a guaranteed demonstration of the blind
    spot, and doubles as the seed of a formal claim: NO finite test can rule
    out such a sliver — only a proof can.

Route A is the more convincing story ("this happens for real"); Route B is the
insurance and the sharpest illustration ("this is WHY testing can never be
enough"). We report both, honestly labelled.

WHAT COUNTS AS AN "ILLUSION" HERE
---------------------------------
All three of these must hold at once for the same edit:
  1. The coarse test says skill A is GONE (zero surviving test points in the
     region where A should have said HIGH), and
  2. the coarse test says skill B still WORKS (zero failures on B's regions) —
     so a practitioner would happily deploy this edit, and
  3. the solver FINDS an input in A's region where the edited model still says
     HIGH — the skill is provably NOT fully removed.

Everything is written to results/illusion_report.md (plain language) and
results/illusion_report.json (the same facts as data).
"""

from __future__ import annotations
import json, os, time
import numpy as np

from tiny_model import TinyMLP, train, find_skill_circuit, accuracy, \
    true_label_A, true_label_B
from verify import prove_forall, grid_check

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# Same regions as run_slice.py: we make claims away from the fuzzy 0.5 border.
REGION_A_HIGH = (0.6, 1.0, 0.0, 1.0)   # skill A should say HIGH here (x0 >= 0.6)
REGION_B_HIGH = (0.0, 1.0, 0.6, 1.0)   # skill B should say HIGH here (x1 >= 0.6)
REGION_B_LOW  = (0.0, 1.0, 0.0, 0.4)   # skill B should say LOW here  (x1 <= 0.4)

# The "coarse test": a 15x15 grid of test points per region (225 points).
# That's the size of a plausible real-world test suite, and it is exactly the
# kind of net the illusion slips through.
COARSE_N = 15
# A second style of test: a few hundred RANDOM inputs, like a sampled test set.
RANDOM_TEST_N = 300
RANDOM_TEST_SEED = 42


# ---------------------------------------------------------------------------
# Small helpers for "testing the way a practitioner would".
# ---------------------------------------------------------------------------
def coarse_test_passes(model, edit) -> dict:
    """
    Run the whole practitioner-style test battery on an edited model.

    Returns a dict with the three verdicts:
      removal_looks_done : True if NO test point in A's HIGH region still says HIGH
      b_looks_preserved  : True if B is correct at every test point in B's regions
      passed             : both of the above (i.e. "deploy it")
    """
    # grid_check counts VIOLATIONS of a claim on sampled points. For removal we
    # claim "logit_A <= 0" (says LOW); a violation = a test point where the
    # skill visibly survives.
    removal = grid_check(model, REGION_A_HIGH, "A", "nonpositive",
                         ablate=edit, n=COARSE_N)
    b_high = grid_check(model, REGION_B_HIGH, "B", "positive",
                        ablate=edit, n=COARSE_N)
    b_low = grid_check(model, REGION_B_LOW, "B", "nonpositive",
                       ablate=edit, n=COARSE_N)
    removal_ok = removal["violations_found"] == 0
    b_ok = b_high["violations_found"] == 0 and b_low["violations_found"] == 0
    return {"removal_looks_done": removal_ok, "b_looks_preserved": b_ok,
            "passed": removal_ok and b_ok,
            "test_points_per_region": removal["grid_points"]}


def random_test_survivors(model, edit, n=RANDOM_TEST_N, seed=RANDOM_TEST_SEED):
    """
    The other common testing style: n RANDOM inputs in skill A's HIGH region.
    Returns how many of them still say HIGH after the edit (survivors seen).
    """
    rng = np.random.default_rng(seed)
    lo0, hi0, lo1, hi1 = REGION_A_HIGH
    X = np.stack([rng.uniform(lo0, hi0, n), rng.uniform(lo1, hi1, n)], axis=1)
    logits = model.forward(X, ablate=edit)
    return int((logits[:, 0] > 0).sum())


def survivor_area_fraction(model, edit, n=1001) -> float:
    """
    Estimate how BIG the surviving pocket is: sweep a very fine grid (about a
    million points) over A's HIGH region and return the fraction of points
    where the edited model still says HIGH. This is for reporting only — it
    tells the reader "the survivor covers just X% of the region", which is
    exactly why coarser tests miss it. (Done in chunks to keep memory low.)
    """
    lo0, hi0, lo1, hi1 = REGION_A_HIGH
    xs0 = np.linspace(lo0, hi0, n)
    xs1 = np.linspace(lo1, hi1, n)
    total, hits = 0, 0
    for i in range(0, n, 50):                     # 50 rows of the grid at a time
        G0, G1 = np.meshgrid(xs0[i:i + 50], xs1)
        pts = np.stack([G0.ravel(), G1.ravel()], axis=1)
        logits = model.forward(pts, ablate=edit)
        hits += int((logits[:, 0] > 0).sum())
        total += pts.shape[0]
    return hits / total


def rank_neurons_by_damage_to_A(model, n=8000, seed=7):
    """
    Which neurons matter most for skill A? Knock each one out alone and measure
    how much skill A's accuracy drops. Returns a list of (neuron, drop_for_A,
    drop_for_B), sorted with the most A-critical neuron first. This mimics how
    an interpretability researcher would pick what to switch off.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    Y = np.stack([true_label_A(X), true_label_B(X)], axis=1)
    base = accuracy(model, X, Y)
    rows = []
    for j in range(model.H):
        acc_j = accuracy(model, X, Y, ablate=[j])
        rows.append((j, float(base[0] - acc_j[0]), float(base[1] - acc_j[1])))
    rows.sort(key=lambda r: -r[1])
    return rows


# ---------------------------------------------------------------------------
# ROUTE A — search for the illusion in ordinarily-trained models.
# ---------------------------------------------------------------------------
def route_A_search(n_seeds=25, l1_values=(0.0, 5e-4, 1e-3, 2e-3),
                   hidden_sizes=(16, 24), top_pool=5, max_edit_size=4,
                   train_steps=5000):
    """
    Search many ordinarily-trained models for a naturally-arising illusion.

    Why the search knobs are what they are:
      * l1_values — the "tidiness" penalty. At high tidiness (as in run_slice.py) skills sit
        in single neurons and edits are clean. At zero, skills smear widely and
        a partial edit usually fails the test VISIBLY. The sweet spot for an
        illusion is in between: a faint leftover pathway, too weak to show up
        at test points, just strong enough to peek above zero in some pocket.
        So we sweep several values between "perfectly tidy" and "fully messy".
      * hidden_sizes — more neurons = more chances for redundant pathways.
      * candidate edits — every small subset (up to max_edit_size) of the
        top_pool most A-critical neurons, plus the knock-out circuit-finder's
        answer. These are the edits a practitioner could plausibly arrive at.

    For each candidate edit, the gate is the practitioner's full test battery
    (coarse grid: skill A looks gone AND skill B looks fine). Only edits the
    tests would APPROVE go to the solver. The solver then either proves the
    removal genuine, or hands back a surviving input = an illusion.

    Returns (illusions_found, honest_removals, candidates_checked, models_trained).
    """
    from itertools import combinations

    illusions, honest, checked, trained = [], 0, 0, 0

    for H in hidden_sizes:
        for l1 in l1_values:
            for seed in range(n_seeds):
                model = train(TinyMLP(H=H, seed=seed), l1=l1, seed=seed + 1,
                              steps=train_steps, verbose=False)
                trained += 1

                # Skip models that never learned both skills properly — an
                # edit to a broken model proves nothing interesting.
                rng = np.random.default_rng(999)
                X = rng.uniform(0, 1, size=(6000, 2))
                Y = np.stack([true_label_A(X), true_label_B(X)], axis=1)
                acc = accuracy(model, X, Y)
                if acc[0] < 0.98 or acc[1] < 0.98:
                    continue

                # Candidate edits: all small subsets of the most A-critical
                # neurons, plus the knock-out circuit-finder's answer. De-duped.
                ranked = rank_neurons_by_damage_to_A(model)
                pool = [j for j, _, _ in ranked[:top_pool]]
                candidates = []
                for k in range(1, max_edit_size + 1):
                    for combo in combinations(pool, k):
                        candidates.append(tuple(sorted(combo)))
                found = find_skill_circuit(model, "A")
                if found:
                    candidates.append(tuple(sorted(found)))
                candidates = list(dict.fromkeys(candidates))

                for edit in candidates:
                    edit = list(edit)
                    verdicts = coarse_test_passes(model, edit)
                    if not verdicts["passed"]:
                        continue   # a practitioner would reject this edit anyway
                    checked += 1

                    # The tests say "deploy". Ask the solver for the truth.
                    proof = prove_forall(model, REGION_A_HIGH, "A",
                                         "nonpositive", ablate=edit)
                    if proof["proved"]:
                        honest += 1   # the test's verdict was actually right
                        continue
                    if proof["counterexample"] is None:
                        continue      # solver said "unknown" — no verdict

                    # ILLUSION: the test approved the edit, the solver refutes.
                    cx = proof["counterexample"]
                    # Double-check the counterexample on the float model.
                    logit = float(model.forward(np.array([cx]),
                                                ablate=edit)[0, 0])
                    area = survivor_area_fraction(model, edit)
                    rand_hits = random_test_survivors(model, edit)
                    illusions.append({
                        "seed": seed, "H": H, "l1": l1, "edit": edit,
                        "counterexample": {"x0": cx[0], "x1": cx[1]},
                        "counterexample_logit_numeric": logit,
                        "survivor_area_fraction": area,
                        "coarse_test_points": verdicts["test_points_per_region"],
                        "random_test_survivors": rand_hits,
                        "random_test_points": RANDOM_TEST_N,
                    })
                    print(f"    ILLUSION @ H={H} l1={l1} seed {seed}: "
                          f"edit=off neurons {edit}, survivor at "
                          f"x0={cx[0]:.4f} x1={cx[1]:.4f} "
                          f"(pocket = {area*100:.4f}% of region, "
                          f"random test saw {rand_hits}/{RANDOM_TEST_N})")

    return illusions, honest, checked, trained


# ---------------------------------------------------------------------------
# ROUTE B — construct the illusion by hand (guaranteed to exist).
# ---------------------------------------------------------------------------
# The constructed network, in plain words:
#   * Two "main" neurons do skill A the obvious way (is x0 above 0.5?).
#   * Two more neurons do skill B the same way for x1. These are never touched.
#   * Three extra neurons form a BACKUP for skill A — but a weird one: they
#     only push head A above zero inside a sliver of inputs around x0 = 0.815,
#     a strip about 0.0006 wide. Think of it as a redundant pathway that only
#     wakes up for very specific inputs.
#   * The sliver's position (0.815) is chosen to fall BETWEEN the lines of
#     both our 15x15 coarse test grid and the much finer 201x201 grid that
#     run_slice.py uses as a cross-check. So after the edit, every one of those
#     40,000+ test points says "skill A is gone" — and yet it isn't.
#
# The EDIT switches off the two main skill-A neurons — exactly what a
# researcher would do after (correctly!) identifying them as "the skill A
# circuit". The backup sliver is what the edit misses.
#
# How the sliver works (the only slightly clever part): three ReLU neurons
# with the same input weight but staggered offsets combine into a "tent": a
# function that is zero everywhere except a narrow triangular bump around the
# centre. Head A adds (bump * BETA) and subtracts a constant BIAS_A, so head A
# only goes above zero where the bump is tall enough — the middle of the tent.
SLIVER_CENTER = 0.815     # midway between grid lines of BOTH test grids
TENT_HALFWIDTH = 0.0006   # the tent is nonzero on (center +/- this)
BETA = 1000.0             # how strongly the tent feeds head A
BIAS_A = 0.3              # the hurdle; head A > 0 only where BETA*tent > this


def build_constructed_model() -> tuple[TinyMLP, list[int]]:
    """Hand-set every weight of a 7-neuron TinyMLP. Returns (model, edit)."""
    m = TinyMLP(H=7, seed=0)
    c, w = SLIVER_CENTER, TENT_HALFWIDTH

    # W1 row = [weight on x0, weight on x1]; b1 = the neuron's offset.
    m.W1 = np.array([
        [ 20.0,   0.0],   # neuron 0: main A+  -> ReLU(20*(x0 - 0.5))
        [-20.0,   0.0],   # neuron 1: main A-  -> ReLU(20*(0.5 - x0))
        [  1.0,   0.0],   # neuron 2: tent part 1 -> ReLU(x0 - (c - w))
        [  1.0,   0.0],   # neuron 3: tent part 2 -> ReLU(x0 - c)
        [  1.0,   0.0],   # neuron 4: tent part 3 -> ReLU(x0 - (c + w))
        [  0.0,  20.0],   # neuron 5: main B+  -> ReLU(20*(x1 - 0.5))
        [  0.0, -20.0],   # neuron 6: main B-  -> ReLU(20*(0.5 - x1))
    ])
    m.b1 = np.array([-10.0, 10.0, -(c - w), -c, -(c + w), -10.0, 10.0])

    # Head A (row 0): main pathway (n0 - n1), plus the tent combination
    # (n2 - 2*n3 + n4) scaled by BETA, minus the hurdle BIAS_A.
    # Head B (row 1): just its own pathway (n5 - n6).
    m.W2 = np.array([
        [1.0, -1.0, BETA, -2.0 * BETA, BETA, 0.0, 0.0],
        [0.0,  0.0, 0.0,   0.0,        0.0,  1.0, -1.0],
    ])
    m.b2 = np.array([-BIAS_A, 0.0])

    edit = [0, 1]   # switch off the two MAIN skill-A neurons (the honest find)
    return m, edit


def route_B_constructed():
    """Run the full battery on the constructed model and collect the facts."""
    model, edit = build_constructed_model()

    # Sanity: before the edit, both skills work (proved, not just tested).
    control_A = prove_forall(model, REGION_A_HIGH, "A", "positive", ablate=None)

    # The practitioner's tests on the EDITED model:
    verdicts = coarse_test_passes(model, edit)
    fine = grid_check(model, REGION_A_HIGH, "A", "nonpositive",
                      ablate=edit, n=201)          # the base slice's fine cross-check
    rand_hits = random_test_survivors(model, edit)

    # The solver's verdict on removal:
    removal = prove_forall(model, REGION_A_HIGH, "A", "nonpositive", ablate=edit)

    # And on preservation of skill B (these should genuinely PROVE):
    pres_high = prove_forall(model, REGION_B_HIGH, "B", "positive", ablate=edit)
    pres_low = prove_forall(model, REGION_B_LOW, "B", "nonpositive", ablate=edit)

    # If the solver found the survivor, double-check it numerically and
    # measure the pocket.
    cx, logit, area = None, None, None
    if removal["counterexample"] is not None:
        cx = removal["counterexample"]
        logit = float(model.forward(np.array([cx]), ablate=edit)[0, 0])
        area = survivor_area_fraction(model, edit)

    return {
        "edit": edit,
        "control_A_proved_before_edit": control_A["proved"],
        "coarse_test": verdicts,
        "fine_grid_201x201_survivors": fine["violations_found"],
        "fine_grid_201x201_points": fine["grid_points"],
        "random_test_survivors": rand_hits,
        "random_test_points": RANDOM_TEST_N,
        "solver_removal_proved": removal["proved"],
        "solver_status": removal["status"],
        "counterexample": None if cx is None else {"x0": cx[0], "x1": cx[1]},
        "counterexample_logit_numeric": logit,
        "survivor_area_fraction": area,
        "preservation_B_proved": pres_high["proved"] and pres_low["proved"],
        "sliver_center": SLIVER_CENTER,
        "sliver_active_halfwidth": TENT_HALFWIDTH - BIAS_A / BETA,
    }


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def _write_report(routeA, honest, checked, n_seeds, routeB):
    md = []
    md.append("# The intervention illusion\n")
    md.append("Output of `run_illusion.py`. The question: can an edit LOOK "
              "successful under ordinary testing while the solver PROVES the "
              "skill survives? Answer below, two ways.\n")

    md.append("## Route B — the constructed example (guaranteed)\n")
    b = routeB
    md.append(f"- We built a 7-neuron model by hand. Before the edit, skill A "
              f"provably works (proved: {b['control_A_proved_before_edit']}).")
    md.append(f"- The edit switches off the two main skill-A neurons — the "
              f"correct, obvious circuit. A hidden backup pathway survives in "
              f"a sliver of inputs ~{2*b['sliver_active_halfwidth']:.4f} wide "
              f"around x0 = {b['sliver_center']}.")
    md.append(f"- **Coarse test ({b['coarse_test']['test_points_per_region']} "
              f"points): skill A looks GONE** "
              f"(passed: {b['coarse_test']['passed']}).")
    md.append(f"- **Fine grid ({b['fine_grid_201x201_points']} points): "
              f"{b['fine_grid_201x201_survivors']} survivors seen** — even "
              f"40,000+ test points miss the sliver.")
    # Honesty about the random test: unlike the
    # grids, which the sliver dodges BY CONSTRUCTION, a random test misses it
    # only with some probability — report that probability, not just the pass.
    miss_p = (1.0 - b["survivor_area_fraction"]) ** b["random_test_points"]
    md.append(f"- Random test ({b['random_test_points']} points): "
              f"{b['random_test_survivors']} survivors seen. (Honest footnote: "
              f"at this pocket size a {b['random_test_points']}-point random "
              f"test misses the sliver with probability only "
              f"{miss_p:.0%} — this pass is seed luck. The grids are dodged by "
              f"construction, and the recipe behind this model — proposition "
              f"P3 — shrinks the pocket below any target test size; the "
              f"in-principle claim rests on those, not on this coin flip.)")
    md.append(f"- **Solver: removal NOT proved** — counterexample at "
              f"x0={b['counterexample']['x0']:.6f}, "
              f"x1={b['counterexample']['x1']:.6f} "
              f"(numeric check: edited head A logit = "
              f"{b['counterexample_logit_numeric']:.4f} > 0, skill alive).")
    md.append(f"- The surviving pocket covers only "
              f"**{b['survivor_area_fraction']*100:.3f}%** of the region — "
              f"small enough to hide from tests, impossible to hide from the "
              f"solver.")
    md.append(f"- Skill B, meanwhile, is PROVED preserved "
              f"({b['preservation_B_proved']}) — so every test a practitioner "
              f"would run says this edit is a clean success.\n")

    md.append("## Route A — searched in ordinarily-trained models\n")
    md.append(f"- Trained {n_seeds} models across a sweep of tidiness "
              f"penalties and hidden-layer sizes; {checked} candidate edits "
              f"passed the practitioner's full test battery.")
    md.append(f"- Of those, {honest} were PROVED genuinely removed (the test's "
              f"verdict was right), and **{len(routeA)} were illusions** (the "
              f"test approved an edit the solver refuted).\n")
    if routeA:
        md.append("| H | tidiness (l1) | seed | edit (neurons off) | survivor "
                  "found at | pocket size | random test saw |")
        md.append("|---|---|---|---|---|---|---|")
        for il in routeA:
            md.append(f"| {il['H']} | {il['l1']} | {il['seed']} | {il['edit']} | "
                      f"x0={il['counterexample']['x0']:.4f}, "
                      f"x1={il['counterexample']['x1']:.4f} | "
                      f"{il['survivor_area_fraction']*100:.4f}% | "
                      f"{il['random_test_survivors']}/{il['random_test_points']} |")
        md.append("")
    else:
        md.append("_No naturally-arising illusion in this batch of seeds — "
                  "the constructed example (Route B) still demonstrates the "
                  "blind spot; widen the search (more seeds, other edit "
                  "choices) to find a trained one._\n")

    md.append("## What this means, in one sentence\n")
    md.append("An edit can pass every test a careful practitioner would run — "
              "coarse grid, fine grid, random samples, collateral checks — "
              "while the skill it was supposed to remove provably survives in "
              "a pocket of inputs the tests never touch; only a proof over the "
              "WHOLE region (or any white-box method — the point is opening "
              "the model, not sampling it) can tell the difference.\n")

    with open(os.path.join(RESULTS, "illusion_report.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(RESULTS, "illusion_report.json"), "w") as f:
        json.dump({"route_B_constructed": routeB,
                   "route_A_search": {"seeds_searched": n_seeds,
                                      "candidates_test_approved": checked,
                                      "proved_genuinely_removed": honest,
                                      "illusions": routeA}}, f, indent=2)


def main():
    print("=" * 70)
    print("THE INTERVENTION ILLUSION")
    print("=" * 70)

    print("\n[1] Route B — constructed example (guaranteed illusion):")
    t0 = time.time()
    routeB = route_B_constructed()
    ct = routeB["coarse_test"]
    print(f"    coarse test ({ct['test_points_per_region']} pts): "
          f"removal looks done = {ct['removal_looks_done']}, "
          f"B looks preserved = {ct['b_looks_preserved']}")
    print(f"    fine grid ({routeB['fine_grid_201x201_points']} pts): "
          f"{routeB['fine_grid_201x201_survivors']} survivors seen")
    print(f"    random test ({routeB['random_test_points']} pts): "
          f"{routeB['random_test_survivors']} survivors seen")
    print(f"    solver: removal proved = {routeB['solver_removal_proved']}"
          + ("" if routeB["counterexample"] is None else
             f" — counterexample x0={routeB['counterexample']['x0']:.6f}, "
             f"x1={routeB['counterexample']['x1']:.6f} "
             f"(logit {routeB['counterexample_logit_numeric']:.4f} > 0)"))
    print(f"    surviving pocket: {routeB['survivor_area_fraction']*100:.3f}% "
          f"of the region; skill B preservation proved = "
          f"{routeB['preservation_B_proved']}  ({time.time()-t0:.1f}s)")

    print("\n[2] Route A — searching trained models (sweep of tidiness "
          "penalties and sizes):")
    t0 = time.time()
    routeA, honest, checked, trained = route_A_search()
    print(f"    {trained} models trained; {checked} candidate edits passed "
          f"the full test battery")
    print(f"    -> {honest} proved genuinely removed, "
          f"{len(routeA)} ILLUSIONS found  ({time.time()-t0:.1f}s)")

    _write_report(routeA, honest, checked, trained, routeB)

    print("\n" + "=" * 70)
    print("HEADLINE:")
    ok_B = (not routeB["solver_removal_proved"]) and routeB["coarse_test"]["passed"]
    print(f"  Constructed illusion demonstrated (Route B) : {ok_B}")
    print(f"  Naturally-trained illusions found (Route A) : {len(routeA)}")
    print("=" * 70)
    print("\nReport written to: results/illusion_report.md and .json")


if __name__ == "__main__":
    main()
