"""
run_collateral_sweep.py — a DENSE diff-of-means dose sweep (paper Fig. 6b).

The robustness report (`run_robustness.py`) measures diff-of-means at only two doses
per model (the minimal-passing dose and 4x). This runs the SAME certified
preservation-radius computation over a fine grid of doses, so the dose->collateral
curve is drawn from many points, not two — showing P4's linearity empirically and
locating where skill B breaks. Same models, same regions, same solver params as
`run_robustness.py` (imported from it), so shared doses reproduce that report.

Records, per dose per model: skill B's certified preservation radius (min over
B's two regions; None = refuted = skill B broken) AND skill A's removal radius
(None/0 = the dose is too weak to even remove A). Output:
results/collateral_sweep.json — consumed by figures/make_figures.py (fig_p3b).
"""
from __future__ import annotations
import json
import os
import time

from tiny_model import TinyMLP, train
from edits import apply_steering, apply_ablation, diff_of_means_vector
from verify import certified_radius
import run_robustness as RR

DOSES = [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def _pres_radius(model):
    hi = certified_radius(model, RR.REGION_B_HIGH, "B", "positive",
                          eps_max=RR.EPS_MAX, tol=RR.TOL)
    lo = certified_radius(model, RR.REGION_B_LOW, "B", "nonpositive",
                          eps_max=RR.EPS_MAX, tol=RR.TOL)
    if hi["radius"] is None or lo["radius"] is None:
        return None
    return min(hi["radius"], lo["radius"])


def _removal_radius(model):
    r = certified_radius(model, RR.REGION_A_HIGH, "A", "nonpositive",
                         eps_max=RR.EPS_MAX, tol=RR.TOL)
    return r["radius"]           # None = removal not certified even at eps=0


def sweep(model, label):
    rows = []
    for d in DOSES:
        m = model if d == 0 else apply_steering(model, diff_of_means_vector(model, d))
        pres = _pres_radius(m)
        rem = _removal_radius(m)
        rows.append({"dose": d, "preservation_radius": pres,
                     "removal_radius": rem,
                     "removes_A": rem is not None})
        print(f"  {label} dose {d:>4}: preservation "
              f"{'REFUTED' if pres is None else round(pres, 4)} | "
              f"removes_A {rem is not None}", flush=True)
    return rows


def main():
    t0 = time.time()
    print("Dense diff-of-means collateral sweep")
    print("[1] tidy (well-separated) model...")
    tidy = train(TinyMLP(H=16, seed=0), verbose=False)

    print("[2] messy (entangled) model (replicating run_robustness's selection)...")
    messy = None
    for seed in range(10):
        cand = train(TinyMLP(H=16, seed=seed), l1=0.0, seed=seed + 1, verbose=False)
        ranked = RR.rank_neurons_by_damage_to_A(cand)
        for k in range(1, 6):
            if RR.tests_pass(apply_ablation(cand, ranked[:k])):
                messy = cand
                break
        if messy is not None:
            print(f"    using seed {seed}")
            break

    print("[3] tidy sweep:")
    tidy_rows = sweep(tidy, "tidy")
    print("[4] messy sweep:")
    messy_rows = sweep(messy, "messy")

    out = {"doses": DOSES, "tidy": tidy_rows, "messy": messy_rows,
           "eps_max": RR.EPS_MAX, "tol": RR.TOL, "seconds": time.time() - t0}
    with open(os.path.join(RESULTS, "collateral_sweep.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote results/collateral_sweep.json  ({out['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
