"""
run_illusion_dims.py -- the intervention-illusion search, broken down by input
dimension.

What and why: run_illusion_nd.py pools four- and five-input models into one
tally (160 models, 3 illusions). This driver re-runs that SAME search protocol
one input dimension at a time -- d = 2, 3, 4, 5 -- with identical seeds,
tidiness grid, test battery, circuit search and solver, and records the counts
per dimension. It lets the rate at which ordinary training produces an
illusion be read against dimension (paper Fig. 1c).

Each (dimension, tidiness, seed) model is trained and tested with its own fixed
random streams, so running one dimension alone gives exactly the same per-model
results as the pooled run. The d = 4 and d = 5 rows therefore reproduce the
committed results/illusion_nd_report.json; the script checks this and says so.
The committed nd report is left untouched.

Usage:
    python run_illusion_dims.py              # d = 2, 3, 4, 5 in sequence, then merge
    python run_illusion_dims.py --dim 3      # one dimension -> results/illusion_dims_d3.json
    python run_illusion_dims.py --merge      # combine the per-dimension parts

The per-dimension form exists so the four dimensions can run as parallel
processes (each run is single-threaded); --merge then produces the same report
the sequential form does, and removes the part files.

Writes results/illusion_dims_report.{json,md}.
"""
import json
import os
import sys
import time

from run_illusion_nd import search, VOLUME_DARTS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
L1_VALUES = (0.0, 5e-4, 1e-3, 2e-3)       # as run_illusion_nd.search
N_SEEDS = 20


def run(dims):
    rows = []
    for d in dims:
        t0 = time.time()
        print(f"--- d = {d}: training {len(L1_VALUES) * N_SEEDS} models ...",
              flush=True)
        illusions, honest, checked, trained, unknowns = search(
            dims=(d,), l1_values=L1_VALUES, n_seeds=N_SEEDS)
        rows.append({
            "d": d, "models_trained": trained, "edits_test_approved": checked,
            "proved_genuinely_removed": honest, "solver_unknown": unknowns,
            "illusions": illusions, "n_illusions": len(illusions),
            "seconds": round(time.time() - t0, 1)})
        print(f"    d = {d}: {trained} models, {checked} test-approved edits, "
              f"{honest} proved removed, {unknowns} unknown, "
              f"{len(illusions)} illusions  ({rows[-1]['seconds']} s)",
              flush=True)
    return rows


def reproduction_check(rows):
    """Compare the d=4 and d=5 rows against the committed pooled nd run."""
    path = os.path.join(RESULTS, "illusion_nd_report.json")
    with open(path) as f:
        ref = json.load(f)
    by_d = {r["d"]: r for r in rows}
    if not {4, 5} <= set(by_d):
        return None
    pooled = [by_d[4], by_d[5]]
    got = {
        "models_trained": sum(r["models_trained"] for r in pooled),
        "edits_test_approved": sum(r["edits_test_approved"] for r in pooled),
        "proved_genuinely_removed": sum(r["proved_genuinely_removed"]
                                        for r in pooled),
        "illusions": sorted((i["d"], i["seed"], i["l1"], tuple(i["edit"]))
                            for r in pooled for i in r["illusions"])}
    want = {
        "models_trained": ref["models_trained"],
        "edits_test_approved": ref["edits_test_approved"],
        "proved_genuinely_removed": ref["proved_genuinely_removed"],
        "illusions": sorted((i["d"], i["seed"], i["l1"], tuple(i["edit"]))
                            for i in ref["illusions"])}
    return {"matches": got == want, "reproduced": got, "committed": want}


def write_report(rows, check):
    out = {"protocol": "run_illusion_nd.search, one input dimension at a time",
           "l1_values": list(L1_VALUES), "n_seeds": N_SEEDS,
           "volume_darts": VOLUME_DARTS, "by_dimension": rows,
           "reproduces_committed_nd_run": check}
    with open(os.path.join(RESULTS, "illusion_dims_report.json"), "w") as f:
        json.dump(out, f, indent=2, default=list)
    md = ["# The illusion search, by input dimension\n",
          "Output of `run_illusion_dims.py`: the search protocol of "
          "`run_illusion_nd.py`, run one input dimension at a time with "
          f"identical seeds ({len(L1_VALUES)} tidiness settings x {N_SEEDS} "
          "seeds per dimension).\n",
          "| inputs | models trained | test-approved edits | proved removed "
          "| solver unknown | illusions |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['d']} | {r['models_trained']} | "
                  f"{r['edits_test_approved']} | "
                  f"{r['proved_genuinely_removed']} | {r['solver_unknown']} | "
                  f"**{r['n_illusions']}** |")
    md.append("")
    if check is not None:
        md.append("**Reproduction check.** The d = 4 and d = 5 rows, pooled, "
                  + ("reproduce the committed `illusion_nd_report.json` "
                     "exactly (models, approved edits, proved removals, and the "
                     "identity of every illusion)." if check["matches"] else
                     "do **not** match the committed `illusion_nd_report.json`; "
                     "see the JSON for both tallies.") + "\n")
    with open(os.path.join(RESULTS, "illusion_dims_report.md"), "w") as f:
        f.write("\n".join(md))
    print("wrote results/illusion_dims_report.{json,md}")


def _part_path(d):
    return os.path.join(RESULTS, f"illusion_dims_d{d}.json")


def finish(rows):
    rows = sorted(rows, key=lambda r: r["d"])
    check = reproduction_check(rows)
    if check is not None:
        print("reproduction of committed nd run:",
              "EXACT MATCH" if check["matches"] else "MISMATCH")
    write_report(rows, check)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--dim"]:
        (row,) = run([int(args[1])])
        with open(_part_path(row["d"]), "w") as f:
            json.dump(row, f, indent=2, default=list)
        print(f"wrote {os.path.relpath(_part_path(row['d']), HERE)}")
    elif args[:1] == ["--merge"]:
        parts = sorted(f for f in os.listdir(RESULTS)
                       if f.startswith("illusion_dims_d") and f.endswith(".json"))
        rows = []
        for name in parts:
            with open(os.path.join(RESULTS, name)) as f:
                rows.append(json.load(f))
        finish(rows)
        for name in parts:
            os.remove(os.path.join(RESULTS, name))
    else:
        finish(run([int(a) for a in args] or [2, 3, 4, 5]))
