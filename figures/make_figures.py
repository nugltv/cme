"""
figures/make_figures.py — reproducible figures for the paper and artifact.

Each figure reads from `results/*.json` (or computes from the model files), so
re-running regenerates it from the real data — no hand-placed numbers. Output is
white-background, print-safe vector PDF plus a PNG for quick viewing, written
next to this file in `figures/`.

Palette: the dataviz reference (light/print). Colours are assigned by the job
they do, not by taste — test points (all passing) are the calm blue backdrop;
the survivor pocket and the solver's counterexample are the one critical-red
accent. The blue/red pair is validated (CVD ΔE 23.8, normal-vision 31.6, both
>= 3:1 on the surface); the counterexample also carries a distinct star marker
and a direct label, so identity never rests on colour alone.

Usage:
    python figures/make_figures.py            # build every figure
    python figures/make_figures.py p1         # build just P1
    python figures/make_figures.py --paper    # build every PAPER figure
    python figures/make_figures.py --paper bp # build just the paper fig:bp

Two families of figure live here. The p*/u* builders are on-screen and outreach
variants: wide, self-titled and self-captioned. The paper_* builders at the end
produce the submission figures, authored at final IEEEtran size (3.45 in column,
7.05 in full width) with no in-image title or caption, written to
paper/figures/fig_<role>.pdf. Paper figures must never be rescaled in LaTeX.

On-screen figures (the paper's own figures are the paper_* builders at the
end, built with --paper):
    p1   the intervention illusion
    u1   skill A's output surface, before and after the edit (removal)
    u1b  skill B's output surface, before and after the edit (preservation)
    p2   certified preservation radius by edit type, one model (toy models)
    p2b  the same, both models grouped by colour (toy models)
    p3   diff-of-means steering: the dose-collateral curve (toy models)
    p3b  the dose-collateral curve over a DENSE dose grid, heavily labelled
         (companion to p3; data from run_collateral_sweep.py)
    p4   the dimensional arc: testing fails as dimension grows
    p5   the certified mechanistic edit on a transformer (threshold-gate)
    p6   the exact-certification frontier / size ladder (threshold-gate)
    u4   the tent gadget: why no finite test catches the illusion
    u2   the collateral on the surface: surgical vs steering edit
    u5   certified influence: does the edited head still read x0?
    u5b  the two-copy (siamese) mechanism behind the certified influence
    u3   the certified radius, geometrically: growing a claimed region
    p7   a known-formula edit, both sides certified over noise (modular adders)
    p8   bound propagation validated against exact Z3 (M0): CROWN = exact
    p9   the certified edit past the exact frontier: certified <= true <= attack
         on a softmax+LayerNorm transformer (bound propagation, M1)
    p10  discrete closure vs the continuous ball: the quantifier boundary
"""

import json
import re
import os
import sys

import numpy as np
import logging
import matplotlib
matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)  # OTF header timestamp noise
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)

# ---- palette (dataviz reference, light surface) ---------------------------
SURFACE = "#ffffff"     # print white
INK = "#0b0b0b"         # primary text
INK2 = "#52514e"        # secondary text
MUTED = "#898781"       # axis / muted labels
GRIDL = "#e1e0d9"       # hairline gridlines
BASELINE = "#c3c2b7"    # axis / baseline
TEST = "#2a78d6"        # categorical blue: a test point (all pass) — the backdrop
CRIT = "#d03b3b"        # critical red: the survivor pocket + the counterexample
# diverging map for a signed output: blue = negative (skill inactive), grey = 0,
# red = positive (skill active). Poles are the palette's blue/red diverging pair.
DIVERGING = LinearSegmentedColormap.from_list(
    "cm_div", ["#2a78d6", "#eef0ee", "#d03b3b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 9.5,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
})


def _save(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(HERE, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")


# ---------------------------------------------------------------------------
# P1 — the intervention illusion
# ---------------------------------------------------------------------------
def fig_p1():
    """The whole thesis in one image: an edit that passes every test while the
    solver proves a thin band of inputs where the 'removed' skill still fires."""
    from run_illusion import REGION_A_HIGH, COARSE_N, SLIVER_CENTER

    with open(os.path.join(RESULTS, "illusion_report.json")) as f:
        d = json.load(f)["route_B_constructed"]
    cx = d["counterexample"]                       # the solver's survivor
    cx0, cx1 = cx["x0"], cx["x1"]
    half = d["sliver_active_halfwidth"]            # where skill A still fires
    center = d["sliver_center"]
    pocket_pct = d["survivor_area_fraction"] * 100
    n_coarse = d["coarse_test"]["test_points_per_region"]
    n_fine = d["fine_grid_201x201_points"]
    n_rand = d["random_test_points"]

    lo0, hi0, lo1, hi1 = REGION_A_HIGH             # skill A's removal region
    # the two test grids' column spacings (as run by run_illusion)
    coarse_dx = (hi0 - lo0) / (COARSE_N - 1)       # 15 cols over [0.6,1.0]
    fine_dx = (hi0 - lo0) / (201 - 1)              # 201 cols (40,401 pts)

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(9.2, 4.3), gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- Panel A: the full region, every test passing --------------------
    gx = np.linspace(lo0, hi0, COARSE_N)
    gy = np.linspace(lo1, hi1, COARSE_N)
    GX, GY = np.meshgrid(gx, gy)
    axA.scatter(GX, GY, s=11, c=TEST, alpha=0.55, linewidths=0,
                zorder=2, label=f"test point (all {n_coarse}×3 pass)")
    # the survivor sliver — a hairline at this scale (that is the point)
    axA.axvline(center, color=CRIT, lw=1.1, zorder=3)
    # the solver's counterexample
    axA.scatter([cx0], [cx1], marker="*", s=200, c=CRIT,
                edgecolors="white", linewidths=1.1, zorder=5)
    # zoom window shown in panel B
    zlo, zhi = 0.810, 0.820
    axA.add_patch(Rectangle((zlo, lo1), zhi - zlo, hi1 - lo1, fill=False,
                            ec=INK2, ls=(0, (4, 3)), lw=1.0, zorder=4))
    axA.annotate("surviving region\n(width 0.0006)", xy=(center, 0.62),
                 xytext=(0.66, 0.78), color=CRIT, fontsize=8.5, ha="left",
                 arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.0))
    axA.annotate("detail in (b) →", xy=(zhi, 0.5), xytext=(0.835, 0.5),
                 color=INK2, fontsize=8.0, va="center", ha="left")
    axA.set_xlim(lo0, hi0)
    axA.set_ylim(lo1, hi1)
    axA.set_xlabel("x₀")
    axA.set_ylabel("x₁")
    axA.set_title("(a)  Testing reports the skill removed")
    axA.set_xticks(np.round(np.arange(lo0, hi0 + 1e-9, 0.1), 1))
    axA.set_yticks([0, 0.5, 1.0])

    # ---- Panel B: the zoom, where the grid steps over the survivor -------
    # fine-grid columns falling inside the zoom window (the 40,401-pt grid)
    k0 = int(np.ceil((zlo - lo0) / fine_dx))
    k1 = int(np.floor((zhi - lo0) / fine_dx))
    fine_cols = lo0 + fine_dx * np.arange(k0, k1 + 1)
    for xc in fine_cols:
        axB.axvline(xc, color=GRIDL, lw=0.8, zorder=1)
        axB.scatter([xc] * 5, np.linspace(0.1, 0.9, 5), s=10, c=TEST,
                    alpha=0.7, linewidths=0, zorder=2)
    # the survivor band (where the edited model's skill A logit is > 0)
    axB.axvspan(center - half, center + half, color=CRIT, alpha=0.18, zorder=1)
    axB.axvline(center, color=CRIT, lw=1.0, alpha=0.5, zorder=3)
    axB.scatter([cx0], [cx1], marker="*", s=260, c=CRIT,
                edgecolors="white", linewidths=1.2, zorder=6,
                label="solver's survivor")
    axB.annotate(f"surviving input\nx₀ = {cx0:.4f}", xy=(cx0, cx1),
                 xytext=(zlo + 0.0009, 0.32), color=CRIT, fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.0))
    axB.text(zlo + 0.0005, 0.95,
             "grid spacing 0.002;\nsurviving region width 0.0006",
             color=INK2, fontsize=8.0, va="top")
    axB.set_xlim(zlo, zhi)
    axB.set_ylim(lo1, hi1)
    axB.set_xlabel("x₀  (detail)")
    axB.set_title("(b)  Verification returns a surviving input")
    axB.set_xticks([0.810, 0.812, 0.814, 0.816, 0.818, 0.820])
    axB.set_yticks([0, 0.5, 1.0])
    axB.tick_params(labelleft=False)

    # ---- shared caption --------------------------------------------------
    cap = (f"A constructed two-input model and a mechanistic edit. A "
           f"{n_coarse}-point grid, a {n_fine:,}-point grid, and {n_rand} "
           f"random inputs all report skill A removed.\nFormal verification "
           f"refutes removal, returning a surviving input at x₀ = {cx0:.4f} — "
           f"a region occupying {pocket_pct:.1f}% of the domain, located "
           f"between the grid lines.")
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=8.5, color=INK2)
    fig.suptitle("The intervention illusion: exhaustive testing accepts an "
                 "edit that verification refutes",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(w_pad=2.0)
    _save(fig, "p1_intervention_illusion")


def _style_3d(ax):
    """Quiet the default 3-D chrome for a print-clean surface."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(SURFACE)
        axis.pane.set_edgecolor(GRIDL)
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"].update(color=GRIDL, linewidth=0.5)
    ax.tick_params(colors=MUTED, labelsize=7.5)


# ---------------------------------------------------------------------------
# U1 / U1b — the output surface of a skill, before and after the edit
# ---------------------------------------------------------------------------
def _surface3d(ax, X0, X1, Z, norm, zmin, zmax, zlabel, title):
    """Draw one output surface (with a translucent zero plane) on a 3-D axis."""
    n = Z.shape[0]
    ax.plot_surface(X0, X1, Z, facecolors=DIVERGING(norm(Z)),
                    rcount=n, ccount=n, linewidth=0, antialiased=True,
                    shade=False, zorder=2)
    ax.plot_surface(X0, X1, np.zeros_like(Z), color=MUTED, alpha=0.12,
                    rcount=2, ccount=2, linewidth=0, shade=False, zorder=1)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("x₀", labelpad=4)
    ax.set_ylabel("x₁", labelpad=4)
    ax.set_zlabel(zlabel, labelpad=6, fontsize=8.5, color=INK2)
    ax.set_title(title, y=1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis="x", pad=-2)
    ax.tick_params(axis="y", pad=-2)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1, 1, 0.72))
    _style_3d(ax)


def _output_surface_pair(head_idx, zlabel, suptitle, caption_fn, outname):
    """Plot one head's output over the input square, before and after skill
    A's circuit is switched off. `caption_fn(max_change)` returns the figure
    caption (so the preserved-skill figure can report the tiny change)."""
    from tiny_model import TinyMLP, train, find_skill_circuit

    model = train(TinyMLP(H=16, seed=0), verbose=False)
    circuit = find_skill_circuit(model, "A")          # the edit both figures use

    n = 90
    g = np.linspace(0.0, 1.0, n)
    X0, X1 = np.meshgrid(g, g)
    pts = np.column_stack([X0.ravel(), X1.ravel()])
    z_before = model.forward(pts, ablate=None)[:, head_idx].reshape(n, n)
    z_after = model.forward(pts, ablate=circuit)[:, head_idx].reshape(n, n)
    max_change = float(np.abs(z_after - z_before).max())

    zmin = float(min(z_before.min(), z_after.min()))
    zmax = float(max(z_before.max(), z_after.max()))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=zmin, vmax=zmax)

    fig = plt.figure(figsize=(9.4, 4.7))
    for i, (Z, title) in enumerate([(z_before, "(a)  Original model"),
                                    (z_after, "(b)  After the edit")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        _surface3d(ax, X0, X1, Z, norm, zmin, zmax, zlabel, title)

    fig.text(0.5, 0.02, caption_fn(max_change), ha="center", va="top",
             fontsize=8.5, color=INK2)
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.0)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.16, wspace=0.05)
    _save(fig, outname)


def fig_u1():
    """Skill A (the removed skill): its output rises above zero where the
    skill is active, then lies below zero everywhere after the edit."""
    cap = lambda _: (
        "Skill A's output over the two-input square (positive = the skill "
        "is active). In the original model (a) the output rises above zero "
        "where x₀ > 0.5.\nAfter switching off skill A's circuit (b) the "
        "output lies below zero throughout — verified for every input in "
        "the region — while skill B is unchanged.")
    _output_surface_pair(
        0, "skill A output",
        "A mechanistic edit, seen on the model's output surface",
        cap, "u1_output_surface")


def fig_u1b():
    """Skill B (the preserved skill): the same edit leaves its output surface
    essentially unchanged — the visual companion to U1's removal."""
    cap = lambda mc: (
        "Skill B's output over the two-input square (positive = the skill is "
        "active). The edit that removes skill A leaves skill B's output "
        "essentially unchanged:\nthe two surfaces coincide (largest change "
        f"{mc:.4f}, against a range spanning tens of units), and preservation "
        "is verified for every input in skill B's region.")
    _output_surface_pair(
        1, "skill B output",
        "The same edit preserves the other skill",
        cap, "u1b_output_surface_skillB")


# ---------------------------------------------------------------------------
# P2 / P3 — the edit-comparison and the dose-collateral story (toy models)
# ---------------------------------------------------------------------------
CLEAN = "#2a78d6"       # surgical / targeted edit — no collateral
COLLAT = "#eb6834"      # diff-of-means steering — collateral grows with dose
# status palette (dataviz reference) — always paired with a marker shape + label
GOOD = "#0ca30c"        # proves
WARN = "#e08a00"        # relaxation leaks (a darker amber for white-surface contrast)
BAD = "#d03b3b"         # times out


def _nice_edit(name):
    return (name.replace("steering targeted", "targeted steering")
                .replace("steering diff-of-means", "diff-of-means steering"))


def _dose_of(name):
    m = re.search(r"dose (\d+)", name)
    return int(m.group(1)) if m else 0


def fig_p2():
    """Every edit removes skill A at the maximal radius; they differ only in
    the certified preservation radius they leave for skill B (one model)."""
    with open(os.path.join(RESULTS, "robustness_report.json")) as f:
        subj = json.load(f)["tidy"]
    ctrl = next(r for r in subj["rows"] if r["edit"].startswith("control"))
    ctrl_pres = ctrl["preservation_radius"]
    rows = [r for r in subj["rows"] if not r["edit"].startswith("control")]

    labels = [_nice_edit(r["edit"]) for r in rows]
    vals = [r["preservation_radius"] or 0.0 for r in rows]
    is_dom = ["diff-of-means" in r["edit"] for r in rows]
    colors = [COLLAT if dom else CLEAN for dom in is_dom]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(rows))[::-1]                    # first edit at top
    ax.barh(y, vals, height=0.62, color=colors, zorder=3)
    for yi, v in zip(y, vals):                        # value labels at bar end
        ax.text(v + 0.0015, yi, f"{v:.3f}", va="center", ha="left",
                fontsize=8.5, color=INK2)
    ax.axvline(ctrl_pres, color=MUTED, ls=(0, (4, 3)), lw=1.2, zorder=2)
    ax.text(ctrl_pres, len(rows) - 0.4,
            f"  skill B's natural margin\n  (no edit): {ctrl_pres:.3f}",
            color=INK2, fontsize=8.0, va="top", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("skill B — certified preservation radius")
    ax.set_xlim(0, ctrl_pres * 1.28)
    ax.set_title("The cost of an edit: how much of the other skill it keeps")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRIDL, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    # legend below the plot (above the caption), bars left in place
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CLEAN, label="surgical / targeted edit"),
                       Patch(color=COLLAT, label="diff-of-means steering")],
              loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
              frameon=False, fontsize=9)
    cap = ("Every edit certifies removal of skill A at the maximal radius "
           "(≥ 0.6, the entire input\ndomain); the bars show what each leaves "
           "of skill B. Surgical and targeted edits\npreserve the full "
           "natural margin (zero collateral); diff-of-means steering erodes\n"
           "it, more at higher dose. Well-separated model.")
    ax.text(0.5, -0.28, cap, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color=INK2, linespacing=1.5)
    _save(fig, "p2_edit_comparison")


def _edit_kind(name):
    if name.startswith("ablation"):
        return "ablation"
    if name.startswith("weight edit"):
        return "weight edit"
    if "targeted" in name:
        return "targeted steering"
    if "diff-of-means" in name:
        return "diff-of-means steering"
    return None


def _p2_by_type(rows):
    """Per edit type, the certified preservation radius at the MINIMAL dose
    that passes the tests (for steering); ablation/weight-edit have no dose."""
    best = {}
    for r in rows:
        if r["edit"].startswith("control"):
            continue
        k = _edit_kind(r["edit"])
        pr = r["preservation_radius"]
        if k is None or pr is None or not r["tests_pass"]:
            continue
        dose = _dose_of(r["edit"])
        if k not in best or dose < best[k][1]:
            best[k] = (pr, dose)
    return best


def fig_p2b():
    """Certified preservation radius for skill B, per edit type, on both the
    well-separated and the entangled model (two bars per edit)."""
    with open(os.path.join(RESULTS, "robustness_report.json")) as f:
        d = json.load(f)
    order = ["ablation", "weight edit", "targeted steering",
             "diff-of-means steering"]
    models = [("tidy", CLEAN, "well-separated model"),
              ("messy", COLLAT, "entangled model")]
    by_model = {s: _p2_by_type(d[s]["rows"]) for s, _, _ in models}
    margins = {s: next(r["preservation_radius"] for r in d[s]["rows"]
                       if r["edit"].startswith("control")) for s, _, _ in models}

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y = np.arange(len(order))[::-1]                   # first edit at top
    h = 0.36
    for mi, (subj, color, name) in enumerate(models):
        off = h / 2 if mi == 0 else -h / 2
        vals = [by_model[subj].get(t, (0.0, None))[0] for t in order]
        ax.barh(y + off, vals, height=h, color=color, zorder=3, label=name)
        for yi, v in zip(y + off, vals):
            ax.text(v + 0.0015, yi, f"{v:.3f}", va="center", ha="left",
                    fontsize=8, color=INK2)
    # the models' natural margins (no edit) as a shaded reference band
    lo_m, hi_m = min(margins.values()), max(margins.values())
    ax.axvspan(lo_m, hi_m, color=MUTED, alpha=0.16, zorder=1)
    ax.text(hi_m, len(order) - 0.42, "  natural margin\n  (no edit)",
            color=INK2, fontsize=8.0, va="top", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9.5)
    ax.set_xlabel("skill B — certified preservation radius")
    ax.set_xlim(0, hi_m * 1.30)
    ax.set_title("The cost of an edit: how much of skill B it keeps, "
                 "on two models")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRIDL, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    # legend below the plot (above the caption)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
              frameon=False, fontsize=9)
    cap = ("Certified preservation radius for skill B after each edit, on a "
           "well-separated and an\nentangled model. Every edit certifies "
           "removal of skill A at the maximal radius (≥ 0.6).\n"
           "Surgical and targeted edits reach each model's natural margin — "
           "zero collateral.\nDiff-of-means steering (minimal deployment dose) "
           "falls short, more on the entangled model,\nand at higher dose it "
           "breaks skill B entirely (see the dose–collateral figure).")
    ax.text(0.5, -0.30, cap, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color=INK2, linespacing=1.5)
    _save(fig, "p2b_edit_comparison_both_models")


def fig_p3():
    """The diff-of-means dose-response: the preserved skill's certified margin
    falls with steering dose, and on the entangled model it breaks."""
    with open(os.path.join(RESULTS, "robustness_report.json")) as f:
        d = json.load(f)

    def series(subj):
        rows = d[subj]["rows"]
        ctrl = next(r for r in rows if r["edit"].startswith("control"))
        pts = [(0, ctrl["preservation_radius"], True)]
        for r in rows:
            if "diff-of-means" in r["edit"]:
                pr = r["preservation_radius"]
                ok = (pr is not None) and r["tests_pass"]
                pts.append((_dose_of(r["edit"]), pr if pr else 0.0, ok))
        return sorted(pts)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    broke_seen = False
    for subj, color, name in [("tidy", CLEAN, "well-separated model"),
                              ("messy", COLLAT, "entangled model")]:
        pts = series(subj)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=2.0, marker="o", ms=6,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=1.0, zorder=3, label=name)
        for bx, by, ok in pts:                          # the break: an ✕ marker
            if not ok:
                ax.scatter([bx], [by], marker="X", s=150, color=color,
                           edgecolors="white", linewidths=1.2, zorder=5)
                broke_seen = True

    # identities (incl. the ✕) go in the legend, in the clear upper-right
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=CLEAN, lw=2, marker="o", markerfacecolor=CLEAN,
               markeredgecolor="white", label="well-separated model"),
        Line2D([0], [0], color=COLLAT, lw=2, marker="o", markerfacecolor=COLLAT,
               markeredgecolor="white", label="entangled model"),
    ]
    if broke_seen:
        handles.append(Line2D([0], [0], color=COLLAT, lw=0, marker="X", ms=9,
                              markerfacecolor=COLLAT, markeredgecolor="white",
                              label="preservation refuted (skill B broken)"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5)
    ax.set_xlabel("diff-of-means steering dose")
    ax.set_ylabel("skill B — certified preservation radius")
    ax.set_ylim(-0.004, 0.11)
    ax.set_xlim(-1.5, 34)
    ax.axhline(0, color=BASELINE, lw=1.0, zorder=1)
    ax.set_title("Realistic steering erodes the preserved skill, dose by dose")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRIDL, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    cap = ("Dose 0 is the unedited model's natural margin. As the "
           "diff-of-means steering\ndose rises, skill B's certified "
           "preservation radius falls; on the entangled\nmodel it reaches "
           "zero — the edit removes skill A but destroys skill B.\nSurgical "
           "and targeted edits stay flat at the natural margin "
           "(see the edit-comparison figure).")
    fig.tight_layout()
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=8.5, color=INK2,
             linespacing=1.5)
    _save(fig, "p3_dose_collateral")


# ---------------------------------------------------------------------------
# P5 / P6 — the certified transformer edit and the exact-certification frontier
# ---------------------------------------------------------------------------
def fig_p5():
    """The threshold-gate result: a transformer with skill A's DISTRIBUTED circuit
    (an attention head + MLP neurons) highlighted, and the certified removal /
    preservation radii it achieves over all sequences × embedding noise."""
    from run_transformer import SUBJECT_CONFIG, SUBJECT_L
    with open(os.path.join(RESULTS, "transformer_report.json")) as f:
        d = json.load(f)
    circuit = [tuple(c) for c in d["circuit"]]
    circ_heads = {i for k, i in circuit if k == "head"}
    circ_mlp = {i for k, i in circuit if k == "mlp"}
    edit = d["edits"][0]
    r_removal = edit["removal_radius"]["radius"]
    r_pres = edit["preservation_radius"]["radius"]
    r_ctrl = d["control_radii"]["preservation"]["radius"]
    H, M = SUBJECT_CONFIG["n_heads"], SUBJECT_CONFIG["d_mlp"]

    fig, (axS, axR) = plt.subplots(
        1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})

    # ---- (a) the schematic: a one-block transformer, circuit highlighted --
    axS.set_xlim(0, 10)
    axS.set_ylim(0, 10)
    axS.axis("off")
    from matplotlib.patches import FancyBboxPatch

    def stage(x, w, y, h, label, fill="#eef2f7", ec=BASELINE):
        axS.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,"
                     "rounding_size=0.12", fc=fill, ec=ec, lw=1.0, zorder=2))
        axS.text(x + w / 2, y + h + 0.28, label, ha="center", va="bottom",
                 fontsize=8.5, color=INK2)

    def arrow(x0, x1, y=5.0):
        axS.annotate("", xy=(x1, y), xytext=(x0, y),
                     arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))

    # stage 1: embeddings + noise
    stage(0.2, 1.7, 3.2, 3.6, "embeddings\n+ ε noise")
    arrow(1.9, 2.6)
    # stage 2: attention heads
    stage(2.6, 2.0, 2.6, 4.8, "attention")
    for h in range(H):
        yy = 6.0 - h * 1.7
        active = h in circ_heads
        axS.add_patch(FancyBboxPatch((2.85, yy), 1.5, 1.2,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     fc=BAD if active else "#cfe0f5",
                     ec="white", lw=1.0, zorder=3))
        axS.text(3.6, yy + 0.6, f"head {h}", ha="center", va="center",
                 fontsize=8, color="white" if active else INK2, zorder=4)
    arrow(4.6, 5.3)
    # stage 3: MLP neurons
    stage(5.3, 2.2, 1.6, 6.8, f"MLP ({M} neurons)")
    for j in range(M):
        yy = 7.7 - j * 0.78
        active = j in circ_mlp
        axS.add_patch(FancyBboxPatch((5.95, yy), 0.9, 0.6,
                     boxstyle="round,pad=0.01,rounding_size=0.05",
                     fc=BAD if active else "#cfe0f5", ec="white", lw=0.8,
                     zorder=3))
        axS.text(6.4, yy + 0.3, str(j), ha="center", va="center",
                 fontsize=7, color="white" if active else INK2, zorder=4)
    arrow(7.5, 8.2)
    # stage 4: readout
    stage(8.2, 1.6, 3.6, 2.8, "readout\n→ skill A")
    axS.set_title("(a)  A transformer, with skill A's circuit highlighted",
                  fontsize=10)
    axS.text(5.0, 0.4, "circuit = attention head "
             f"{sorted(circ_heads)[0]} + MLP neurons "
             f"{sorted(circ_mlp)} — distributed across attention and the MLP",
             ha="center", va="bottom", fontsize=8.0, color=BAD)

    # ---- (b) the certified radii -----------------------------------------
    bars = [("removal\n(skill A gone)", r_removal, BAD),
            ("preservation\n(skill B kept)", r_pres, CLEAN)]
    y = np.arange(len(bars))[::-1]
    axR.barh(y, [b[1] for b in bars], height=0.5,
             color=[b[2] for b in bars], zorder=3)
    for yi, (_, v, _) in zip(y, bars):
        axR.text(v + 0.0007, yi, f"{v:.3f}", va="center", ha="left",
                 fontsize=9, color=INK2)
    axR.axvline(r_ctrl, color=MUTED, ls=(0, (4, 3)), lw=1.2, zorder=2)
    axR.text(r_ctrl, 0.5, f"  skill B,\n  no edit:\n  {r_ctrl:.3f}",
             color=INK2, fontsize=8.0, va="center", ha="left")
    axR.set_yticks(y)
    axR.set_yticklabels([b[0] for b in bars], fontsize=8.5)
    axR.set_xlim(0, r_ctrl * 1.62)
    axR.set_xlabel("certified radius (embedding noise)")
    axR.set_title("(b)  What the edit certifies", fontsize=10)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    axR.grid(axis="x", color=GRIDL, lw=0.6, zorder=0)
    axR.set_axisbelow(True)

    cap = ("A trained one-block transformer (2 attention heads + MLP, d_model "
           f"{SUBJECT_CONFIG['d_model']}, sequence length {SUBJECT_L}). "
           "Switching off skill A's distributed circuit certifies its removal "
           "over every\nsequence and all embedding-space noise (radius "
           f"{r_removal:.3f}), while skill B is preserved (radius {r_pres:.3f}); "
           f"the edit cuts skill B's margin from {r_ctrl:.3f} to {r_pres:.3f}.")
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=8.5, color=INK2,
             linespacing=1.5)
    fig.suptitle("A certified mechanistic edit on a real transformer",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, "p5_transformer_edit")


def _parse_size_ladder():
    text = open(os.path.join(RESULTS, "rung2_size_ladder.log")).read()
    out = []
    for block in text.split("=== "):
        m = re.match(r"(.+?)\s*\(noise vars = (\d+)\)", block)
        if not m:
            continue
        label, nv = m.group(1).strip(), int(m.group(2))
        verdicts = re.findall(r"hull A-pos eps=[\d.]+: (\w+)", block)
        if verdicts and all(v == "PROVED" for v in verdicts):
            outcome = "proves"
        elif any(v == "relaxation" for v in verdicts):
            outcome = "leaks"
        else:
            outcome = "walls"
        out.append((nv, label, outcome))
    return sorted(out)


def fig_p6():
    """The measured frontier: how large the model can be before EXACT
    input-side certification stops — it proves, then the relaxation leaks,
    then the solver times out, as the number of continuous noise variables grows."""
    ladder = _parse_size_ladder()
    style = {"proves": (GOOD, "o", "proves (exact certificate)"),
             "leaks": (WARN, "^", "relaxation leaks (inconclusive)"),
             "walls": (BAD, "X", "solver times out (300 s)")}

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    y = np.arange(len(ladder))[::-1]                  # smallest at top
    for yi, (nv, label, outcome) in zip(y, ladder):
        color, marker, _ = style[outcome]
        ax.plot([0, nv], [yi, yi], color=GRIDL, lw=1.2, zorder=1)  # stem
        ax.scatter([nv], [yi], marker=marker, s=150, color=color,
                   edgecolors="white", linewidths=1.2, zorder=3)
        ax.text(nv + 3, yi, f"{nv}", va="center", ha="left", fontsize=8.5,
                color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in ladder], fontsize=9)
    ax.set_xlabel("number of continuous noise variables  (model size)")
    ax.set_xlim(0, max(nv for nv, _, _ in ladder) * 1.16)
    ax.set_title("How far exact certification reaches, and where it stops")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRIDL, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    # mark the certified subject (the 2-head variant at the same 48 vars)
    ship_y = next(yi for yi, (nv, _, _) in zip(y, ladder) if nv == 48)
    ax.annotate("the certified subject (2-head\nvariant) sits here — proves",
                xy=(48, ship_y), xytext=(66, ship_y + 0.75), fontsize=8.5,
                color=INK2, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], lw=0, marker=m, ms=9, color=c,
                      markeredgecolor="white", label=lab)
               for c, m, lab in style.values()]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5)
    cap = ("Each row is a trained subject; the marker is the outcome of the "
           "all-sequences × embedding-noise certificate. Exact certification "
           "holds up to ~48 noise\nvariables (the certified subject); at 64 the "
           "continuous relaxation leaks; beyond that the noise-robust claim "
           "times out (the noise-free one still proves at 72 and 96). Mapping "
           "this frontier is itself a contribution.")
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=8.5, color=INK2,
             linespacing=1.5)
    fig.tight_layout()
    _save(fig, "p6_certification_frontier")


# ---------------------------------------------------------------------------
# P4 — the dimensional arc
# ---------------------------------------------------------------------------
def fig_p4():
    """Testing is reliable in two dimensions and fails from four up: the same
    search finds no illusions in 2-input models but several in 4-5-input ones,
    because test coverage of a region collapses as dimension grows."""
    with open(os.path.join(RESULTS, "illusion_report.json")) as f:
        r2 = json.load(f)["route_A_search"]
    with open(os.path.join(RESULTS, "illusion_nd_report.json")) as f:
        rn = json.load(f)

    rows = [
        ("2 inputs\n(200 models)", r2["candidates_test_approved"],
         r2["candidates_test_approved"] - r2["proved_genuinely_removed"]),
        ("4–5 inputs\n(160 models)", rn["edits_test_approved"],
         len(rn["illusions"])),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, total, illus) in zip(y, rows):
        genuine = total - illus
        ax.barh(yi, genuine, height=0.5, color=TEST, zorder=3)
        ax.barh(yi, illus, left=genuine, height=0.5, color=CRIT, zorder=3)
        if illus:
            ax.annotate(f"{illus} illusions\n(solver refutes)",
                        xy=(total, yi), xytext=(total + 18, yi + 0.5),
                        fontsize=8.5, color=CRIT, ha="left", va="center",
                        arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.0))
        else:
            ax.text(total + 6, yi, f"{total} approved — 0 illusions",
                    va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("test-approved edits (grid + random tests all pass)")
    ax.set_xlim(0, 320)
    ax.set_title("Testing catches every fake removal in 2 dimensions — "
                 "not from 4 up")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRIDL, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TEST, label="genuinely removed "
                             "(test agrees with the proof)"),
                       Patch(color=CRIT, label="illusion (test approves, "
                             "proof refutes)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
              frameon=False, fontsize=8.5)
    cap = ("The same automated search over trained models, run in 2 inputs and "
           "in 4–5 inputs.\nIn 2 dimensions every test-approved edit is a "
           "genuine removal; from 4 dimensions up, some\nedits pass a full grid "
           "and 1,000 random inputs yet the solver refutes them — the surviving "
           "regions\nlie below 1.5×10⁻⁶ of the domain (too small for 2 million "
           "random samples to find). Real models have thousands of dimensions.")
    ax.text(0.5, -0.42, cap, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color=INK2, linespacing=1.5)
    _save(fig, "p4_dimensional_arc")


# ---------------------------------------------------------------------------
# U4 — the tent gadget (the mechanism behind the illusion)
# ---------------------------------------------------------------------------
def fig_u4():
    """Why no finite test catches the illusion: after the edit, the model's
    skill-A output sits below zero everywhere except a sharp spike that pokes
    just above zero in a sliver narrower than the test-grid spacing."""
    from run_illusion import (build_constructed_model, COARSE_N, SLIVER_CENTER,
                              TENT_HALFWIDTH)
    model, edit = build_constructed_model()
    c = SLIVER_CENTER
    lo0, hi0 = 0.6, 1.0

    def curve(xs):
        return model.forward(np.column_stack([xs, np.zeros_like(xs)]),
                             ablate=edit)[:, 0]
    coarse = lo0 + (hi0 - lo0) / (COARSE_N - 1) * np.arange(COARSE_N)
    fine = lo0 + (hi0 - lo0) / 200 * np.arange(201)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 4.2))

    # ---- (a) the whole spike (coarse test grid steps over it) ------------
    xlo_a, xhi_a = 0.788, 0.842
    xs = np.linspace(xlo_a, xhi_a, 8000)
    axA.plot(xs, curve(xs), color=INK2, lw=1.6, zorder=3)
    axA.axhline(0, color=BASELINE, lw=1.0, zorder=2)
    cc = coarse[(coarse >= xlo_a) & (coarse <= xhi_a)]
    axA.scatter(cc, curve(cc), s=34, color=TEST, edgecolors="white",
                linewidths=0.8, zorder=4)
    zx = (c - 0.0016, c + 0.0016)
    axA.add_patch(Rectangle((zx[0], -0.012), zx[1] - zx[0], 0.024, fill=False,
                            ec=INK2, ls=(0, (4, 3)), lw=1.0, zorder=5))
    axA.annotate("test points:\nall below zero\n(skill 'removed')",
                 xy=(cc[0], curve(cc[:1])[0]), xytext=(xlo_a + 0.001, -0.15),
                 fontsize=8.5, color=TEST, ha="left",
                 arrowprops=dict(arrowstyle="->", color=TEST, lw=1.0))
    axA.annotate("detail in (b) →", xy=(zx[1], 0.0),
                 xytext=(c + 0.006, -0.09), fontsize=8.0, color=INK2, ha="left")
    axA.set_xlim(xlo_a, xhi_a)
    axA.set_xlabel("x₀")
    axA.set_ylabel("skill A output (after the edit)")
    axA.set_title("(a)  A sharp spike the tests step over")
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # ---- (b) the tip, crossing zero --------------------------------------
    xz = np.linspace(c - 0.0012, c + 0.0012, 4000)
    yz = curve(xz)
    axB.plot(xz, yz, color=INK2, lw=1.8, zorder=3)
    axB.axhline(0, color=BASELINE, lw=1.0, zorder=2)
    axB.fill_between(xz, 0, yz, where=(yz > 0), color=CRIT, alpha=0.30,
                     zorder=1)
    ff = fine[(fine >= c - 0.0012) & (fine <= c + 0.0012)]
    axB.scatter(ff, curve(ff), s=40, color=TEST, edgecolors="white",
                linewidths=0.9, zorder=4)
    axB.annotate("output > 0 here:\nthe skill survives", xy=(c, max(yz) * 0.6),
                 xytext=(c + 0.0003, max(yz) * 0.9), fontsize=8.5, color=CRIT,
                 ha="left")
    axB.annotate("nearest grid points\n(0.002 apart) miss it",
                 xy=(ff[0], curve(ff[:1])[0]), xytext=(c - 0.0011,
                 min(yz) * 0.6), fontsize=8.0, color=TEST, ha="left")
    axB.set_xlim(c - 0.0012, c + 0.0012)
    axB.set_xlabel("x₀  (detail)")
    axB.set_title("(b)  The tip crosses zero, between grid points")
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    cap = ("The constructed model's skill-A output along x₀ after the edit. It "
           "lies below zero at every test point (a), so testing reports the "
           "skill removed;\nbut a spike (a 'tent' of three ReLU units) pokes "
           "above zero in a sliver ~0.0006 wide (b) — narrower than the grid "
           "spacing — so the skill survives there.")
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=8.5, color=INK2,
             linespacing=1.5)
    fig.tight_layout()
    _save(fig, "u4_tent_gadget")


# ---------------------------------------------------------------------------
# U5 — certified influence: does the edited head still "listen" to x0?
# ---------------------------------------------------------------------------
def fig_u5():
    """A removed skill can still leave the head listening to its input. The
    certified influence — the largest change in the head's output from moving
    x0 alone — quantifies it: near zero for a genuine removal, sizeable when
    the removal is only skin-deep."""
    with open(os.path.join(RESULTS, "independence_report.json")) as f:
        d = json.load(f)

    def inf(rows, edit):
        return next(r["influence_A"]["influence"] for r in rows
                    if r["edit"].startswith(edit))
    rows = [
        ("well-separated model, no edit", inf(d["tidy"], "control"), "base"),
        ("well-separated model, after edit", inf(d["tidy"], "ablation"),
         "deaf"),
        ("entangled model, no edit", inf(d["messy"], "control"), "base"),
        ("entangled model, after edit", inf(d["messy"], "ablation"), "listen"),
        ("illusion edit (passed 40,401 tests)",
         d["illusion"]["influence_A"]["influence"], "listen"),
    ]
    cmap = {"base": MUTED, "deaf": TEST, "listen": CRIT}

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], height=0.6,
            color=[cmap[r[2]] for r in rows], zorder=3)
    for yi, (_, v, _) in zip(y, rows):
        ax.text(v * 1.15, yi, f"{v:.3g}", va="center", ha="left", fontsize=8.5,
                color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 3e2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("certified influence of x₀ on skill A's output  (log scale)")
    ax.set_title("Does the edited model still 'listen' to the removed skill's "
                 "input?")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRIDL, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=MUTED, label="no edit (the skill listens)"),
        Patch(color=TEST, label="genuine removal (head goes deaf to x₀)"),
        Patch(color=CRIT, label="removal certified, yet still listening")],
        loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1,
        frameon=False, fontsize=8.5)
    cap = ("The largest change in skill A's output produced by moving x₀ alone "
           "(proved via a two-copy encoding). A genuine removal drives it to "
           "~0.0007 — the\nhead no longer reads x₀. But the entangled model "
           "certifies removal while still listening (10.3), and the illusion "
           "edit still listens (0.60) despite passing every test.")
    ax.text(0.5, -0.40, cap, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color=INK2, linespacing=1.5)
    _save(fig, "u5_certified_influence")


# ---------------------------------------------------------------------------
# U5b — the two-copy (siamese) MECHANISM behind the certified influence
# ---------------------------------------------------------------------------
def fig_u5b():
    """U5 shows the certified-influence NUMBERS; this shows HOW they are proved.
    The edited model is described to the solver twice, over two inputs pinned to
    agree on everything except x0; the solver hunts for a pair whose head-A
    outputs differ by more than kappa. No such pair (unsat) proves the head
    cannot move when only x0 changes. The illusion edit is the worked example:
    a concrete pair the 'removed' head still tells apart. Numbers from the JSON."""
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.gridspec as gridspec
    with open(os.path.join(RESULTS, "independence_report.json")) as f:
        d = json.load(f)
    ill = d["illusion"]
    (px, _), (py, _) = ill["independence_pair"]          # x0 of each copy
    gap = ill["independence_gap"]                         # how far the head moves
    ceil = ill["influence_A"]["influence"]               # bisected ceiling 0.60
    deaf = next(r for r in d["tidy"]
                if r["edit"] == "ablation")["influence_A"]["influence"]

    REDL = "#fbe7e7"        # light red fill for the "free" (may-differ) cells
    BLUEL = "#eef2f7"       # neutral fill for tied cells / model boxes

    fig = plt.figure(figsize=(9.4, 5.6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.62, 0.62], hspace=0.10)
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 10); ax.set_ylim(0.2, 9.9)
    ax.axis("off")
    axr = fig.add_subplot(gs[1]); axr.set_xlim(0, 10); axr.set_ylim(0, 2.4)
    axr.axis("off")

    def cell(x, y, w, h, label, ec, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     ec=ec, fc=fc, lw=1.4, zorder=3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=9.5, color=INK, zorder=4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))

    # ---- input pair (left) ----
    ax.text(1.72, 9.15, "Two inputs, identical except x₀", ha="center",
            va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    cw, chh = 1.15, 0.95
    ax.text(0.40, 8.22, "copy x", fontsize=8.2, color=INK2, va="bottom",
            ha="left")
    cell(0.35, 7.15, cw, chh, "x₀ = a", CRIT, REDL)
    cell(1.72, 7.15, cw, chh, "x₁ = t", MUTED, BLUEL)
    ax.text(0.40, 5.82, "copy y", fontsize=8.2, color=INK2, va="bottom",
            ha="left")
    cell(0.35, 4.75, cw, chh, "x₀ = b", CRIT, REDL)
    cell(1.72, 4.75, cw, chh, "x₁ = t", MUTED, BLUEL)
    # the "tied" bracket on the right of the x1 cells
    xb = 3.02
    ax.plot([xb, xb], [5.22, 7.62], color=INK2, lw=1.3, zorder=3)
    ax.plot([2.87, xb], [7.62, 7.62], color=INK2, lw=1.3, zorder=3)
    ax.plot([2.87, xb], [5.22, 5.22], color=INK2, lw=1.3, zorder=3)
    ax.text(xb + 0.12, 6.42, "=  tied\n(all other\ncoords too)", fontsize=7.8,
            color=INK2, va="center", ha="left")
    # the "free" callout on the x0 cells
    ax.text(0.93, 4.52, "free — may differ", fontsize=7.8, color=CRIT,
            ha="center", va="top")

    # ---- the edited model, applied to each copy (same weights) ----
    for cy in (7.62, 5.22):
        ax.add_patch(FancyBboxPatch((4.35, cy - 0.62), 1.75, 1.24,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     ec=BASELINE, fc=BLUEL, lw=1.2, zorder=3))
        ax.text(5.22, cy, "edited\nmodel", ha="center", va="center",
                fontsize=8.8, color=INK2, zorder=4)
    ax.text(5.22, 6.42, "(same weights,\ndescribed twice)", ha="center",
            va="center", fontsize=7.6, color=MUTED)
    arrow(4.05, 7.62, 4.35, 7.62)
    arrow(4.05, 5.22, 4.35, 5.22)

    # ---- the comparator (head-A gap) ----
    ax.add_patch(FancyBboxPatch((7.55, 5.72), 2.2, 1.4,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 ec=INK2, fc=SURFACE, lw=1.3, zorder=3))
    ax.text(8.65, 6.42, "gap =\n|A(x) − A(y)|", ha="center", va="center",
            fontsize=9, color=INK, zorder=4)
    arrow(6.10, 7.62, 7.55, 6.75)
    arrow(6.10, 5.22, 7.55, 6.10)

    # ---- the solver's question banner ----
    ax.add_patch(FancyBboxPatch((0.5, 0.55), 9.0, 2.35,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 ec=GRIDL, fc="#f5f7f9", lw=1.0, zorder=1))
    ax.text(5.0, 2.42, "Solver asks:  does any such pair make  gap > κ ?",
            ha="center", va="center", fontsize=9.3, color=INK, zorder=2)
    ax.text(5.0, 1.66, "no such pair  (unsat)   ⇒   the head cannot move when "
            "only x₀ changes   ⇒   certified influence of x₀ on head A  ≤  κ",
            ha="center", va="center", fontsize=8.6, color=INK2, zorder=2)
    ax.text(5.0, 1.02, "κ is found by bisection — the same way as a certified "
            "radius; κ = 0 means the head provably ignores x₀ entirely, with no "
            "region to pick.",
            ha="center", va="center", fontsize=7.8, color=MUTED, zorder=2)

    # ---- result strip: the two edits the certificate separates ----
    axr.text(0.06, 2.15, "The certificate then separates two edits that BOTH "
             "pass every ordinary test:", fontsize=8.4, color=INK, va="top")
    axr.text(0.30, 1.30, "✓", fontsize=13, color=TEST, ha="center", va="center")
    axr.text(0.62, 1.30, f"Genuine removal (ablation): no pair moves head A by "
             f"more than {deaf:.4f}  —  provably deaf to x₀.",
             fontsize=8.5, color=INK, va="center", ha="left")
    axr.text(0.30, 0.45, "✗", fontsize=13, color=CRIT, ha="center", va="center")
    axr.text(0.62, 0.45, f"Illusion edit (passes 40,401 tests): x₀ = {px:.4f} vs "
             f"{py:.4f} (x₁ = 0) still moves head A by {gap:.2f}; certified "
             f"ceiling {ceil:.2f}  —  still listening.",
             fontsize=8.5, color=INK, va="center", ha="left")

    fig.suptitle("How the stronger removal claim is proved: the two-copy "
                 "(siamese) certificate", fontsize=11.5, fontweight="bold",
                 y=0.99)
    cap = ("The edited model is described to the solver twice, over two inputs "
           "pinned to agree on every coordinate except x₀; the solver searches "
           "for a pair whose head-A outputs differ by more than κ. Finding none "
           "proves the head's output\ncannot move when only x₀ changes — the "
           "certified influence κ, bisected like a certified radius. Influence "
           "near zero means the head provably ignores x₀ (a genuine removal); a "
           "positive influence is a proved ceiling on the leftover pathway a "
           "region-based removal claim leaves unstated.")
    fig.text(0.5, 0.015, cap, ha="center", va="top", fontsize=8.3, color=INK2,
             linespacing=1.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.14)
    _save(fig, "u5b_two_copy_mechanism")


def fig_u2():
    """The collateral, on the surface: two edits that both remove skill A, but
    a surgical one leaves skill B's output untouched while realistic steering
    visibly pulls it down."""
    from tiny_model import TinyMLP, train, find_skill_circuit
    from edits import apply_ablation, apply_steering, diff_of_means_vector

    model = train(TinyMLP(H=16, seed=0), verbose=False)
    circuit = find_skill_circuit(model, "A")
    surgical = apply_ablation(model, circuit)
    steered = apply_steering(model, diff_of_means_vector(model, strength=32))

    n = 90
    g = np.linspace(0.0, 1.0, n)
    X0, X1 = np.meshgrid(g, g)
    pts = np.column_stack([X0.ravel(), X1.ravel()])
    zB_surg = surgical.forward(pts)[:, 1].reshape(n, n)   # skill B, surgical
    zB_steer = steered.forward(pts)[:, 1].reshape(n, n)    # skill B, steering

    zmin = float(min(zB_surg.min(), zB_steer.min()))
    zmax = float(max(zB_surg.max(), zB_steer.max()))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=zmin, vmax=zmax)

    fig = plt.figure(figsize=(9.4, 4.7))
    panels = [(zB_surg, "(a)  After a surgical edit"),
              (zB_steer, "(b)  After realistic steering")]
    for i, (Z, title) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        _surface3d(ax, X0, X1, Z, norm, zmin, zmax, "skill B output", title)

    cap = ("Skill B's output after two different edits that both remove skill "
           "A. A surgical edit (a) leaves skill B exactly as it was; realistic "
           "diff-of-means steering (b)\npulls the whole surface down — the "
           "certified margin near the decision boundary shrinks, the collateral "
           "the robustness figures measure, here visible directly.")
    fig.text(0.5, 0.02, cap, ha="center", va="top", fontsize=8.5, color=INK2,
             linespacing=1.5)
    fig.suptitle("The collateral of a realistic edit, on the model's surface",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.18, wspace=0.05)
    _save(fig, "u2_both_skills_collateral")


def fig_u3():
    """The certified radius, geometrically: the region where a skill is
    claimed can be grown outward by ε until it reaches the decision boundary;
    just past that a counterexample appears. The largest safe ε is the radius."""
    with open(os.path.join(RESULTS, "robustness_report.json")) as f:
        row = next(r for r in json.load(f)["tidy"]["rows"]
                   if r["edit"] == "ablation")
    ph = row["preservation_high"]
    radius, first_fail = ph["radius"], ph["first_failure"]
    cx = ph["counterexample_at_failure"]
    region_lo = 0.6          # skill-B-HIGH claim region: x1 in [0.6, 1]
    boundary = 0.5           # skill B flips at x1 = 0.5
    infl = region_lo - radius

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.34, 1.0)
    # skill B's HIGH side (shaded) and the decision boundary, labelled once in
    # the otherwise-empty lower region so nothing crowds the narrow band where
    # the certified edge meets the boundary.
    ax.axhspan(boundary, 1.0, color=TEST, alpha=0.05, zorder=0)
    ax.axhline(boundary, ls=(0, (5, 3)), color=MUTED, lw=1.2, zorder=2)
    ax.text(0.015, 0.435, "decision boundary: skill B flips at x₁ = 0.5\n"
            "(above it HIGH, below it LOW)", fontsize=8, color=MUTED,
            va="center")
    # the certified (inflated) region
    ax.add_patch(Rectangle((0, infl), 1, 1 - infl, fc=GOOD, ec=GOOD,
                           alpha=0.16, lw=0, zorder=1))
    ax.axhline(infl, color=GOOD, lw=1.4, zorder=3)
    ax.text(0.985, infl + 0.01, "edge of the certified region", fontsize=8,
            color="#0a7d0a", ha="right", va="bottom")
    # the original claim region
    ax.add_patch(Rectangle((0, region_lo), 1, 1 - region_lo, fill=False,
                           ec=INK2, lw=1.4, ls=(0, (2, 2)), zorder=3))
    ax.text(0.5, 0.83, "region where skill B is claimed HIGH", fontsize=8.5,
            color=INK2, ha="center")
    # the inflation arrow (ε*)
    ax.annotate("", xy=(0.12, infl), xytext=(0.12, region_lo),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(0.145, (region_lo + infl) / 2,
            f"ε* = {radius:.3f}  (certified radius)", fontsize=8.5, color=INK,
            va="center")
    # the counterexample just past the radius
    ax.scatter([cx[0]], [cx[1]], marker="*", s=220, color=CRIT,
               edgecolors="white", linewidths=1.2, zorder=5)
    ax.annotate(f"first counterexample (ε = {first_fail:.3f}):\n"
                "skill B is LOW just below the boundary",
                xy=(cx[0], cx[1]), xytext=(0.44, 0.39), fontsize=8.5,
                color=CRIT, ha="left",
                arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.0))
    ax.set_xlabel("x₀")
    ax.set_ylabel("x₁")
    ax.set_title("The certified radius: how far a claimed region expands before "
                 "verification returns a counterexample")
    cap = ("Skill B is claimed HIGH throughout the dashed region. Expanding that "
           "region outward by ε keeps the claim provable until ε reaches "
           f"{radius:.3f} —\nwhere the region reaches the decision boundary; just "
           "beyond it, verification returns a point across the boundary where "
           "skill B is LOW. That largest provable ε is the certified radius.")
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=8.5, color=INK2,
             linespacing=1.5)
    fig.tight_layout()
    _save(fig, "u3_certified_radius")


# ---------------------------------------------------------------------------
# P3b — the DENSE dose->collateral curve (heavily labelled companion to P3)
# ---------------------------------------------------------------------------
def fig_p3b():
    """A denser, fully-labelled version of P3, from run_collateral_sweep.py: skill
    B's certified preservation radius vs diff-of-means dose over a fine dose grid,
    both models. Every marker is one certified proof; open markers = dose too weak
    to even remove skill A; the X on the axis = preservation refuted (B broken).
    Does NOT overwrite p3_dose_collateral (the 2-points-per-model paper version)."""
    from matplotlib.lines import Line2D
    with open(os.path.join(RESULTS, "collateral_sweep.json")) as f:
        S = json.load(f)

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    series = [("tidy", "well-separated model", CLEAN),
              ("messy", "entangled model", COLLAT)]
    for key, label, col in series:
        rows = S[key]
        pos = [(r["dose"], r["preservation_radius"], r["removes_A"])
               for r in rows if r["preservation_radius"] is not None]
        xs = [d for d, _, _ in pos]
        ys = [v for _, v, _ in pos]
        ax.plot(xs, ys, "-", color=col, lw=2, zorder=3, label=label)
        # filled markers where skill A is actually removed; open where it isn't
        for d, v, rem in pos:
            ax.plot(d, v, marker="o", ms=4.5, zorder=4,
                    color=(col if rem else SURFACE), mec=col, mew=1.2)
        # the break: first refuted dose -> X on the axis, connector down to it
        ref = [r["dose"] for r in rows if r["preservation_radius"] is None]
        if ref:
            dref = min(ref)
            ax.plot([xs[-1], dref], [ys[-1], 0.0], ls=(0, (2, 2)), color=col,
                    lw=1.3, zorder=2)
            ax.plot(dref, 0.0, marker="X", ms=11, color=CRIT, mec=SURFACE,
                    mew=1.0, zorder=6)
            ax.annotate(f"refuted @ {dref:g}", (dref, 0.0),
                        textcoords="offset points", xytext=(0, -13),
                        fontsize=7.8, color=col, ha="center", va="top")

    # natural-margin note (top-left; the legend sits top-right)
    ax.text(0.015, 0.995, "dose 0 = natural margin (no edit)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
            color=INK2)
    # "linear (P4)" annotation between the two curves
    # ax.annotate("≈ linear in dose (Proposition P4)", (13, 0.049),
    #             fontsize=8.4, color=INK2, rotation=-20, rotation_mode="anchor")

    ax.set_xlim(-1.5, 50)
    ax.set_ylim(-0.018, 0.108)
    ax.set_xlabel("diff-of-means steering dose")
    ax.set_ylabel("skill B — certified preservation radius\n(0 = broken)")
    ax.set_title("How diff-of-means steering erodes the preserved skill "
                 "(dense dose sweep)", pad=26)
    ax.text(0.5, 1.03, "each cricular marker along both lines is one certified proof · shared doses "
            "reproduce the robustness report · exact Z3",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.0,
            color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRIDL, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    legend = [
        Line2D([0], [0], color=CLEAN, lw=2, marker="o", ms=5, mec=CLEAN,
               label="well-separated"),
        Line2D([0], [0], color=COLLAT, lw=2, marker="o", ms=5, mec=COLLAT,
               label="entangled"),
        Line2D([0], [0], color=MUTED, lw=0, marker="o", ms=5, mfc=SURFACE,
               mec=MUTED, mew=1.2, label="open = too weak to remove A"),
        Line2D([0], [0], color=CRIT, lw=0, marker="X", ms=8, mec=SURFACE,
               label="✗ = skill B broken"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=7.5, frameon=True,
              facecolor=SURFACE, edgecolor="none", framealpha=0.9,
              borderpad=0.4, labelspacing=0.35, bbox_to_anchor=(1.0, 1.0))
    cap = ("Skill B's certified preservation radius as the diff-of-means steering "
           "dose rises, over a fine grid of doses (each a full solver proof). It "
           "falls **linearly** — as Proposition P4 predicts — until it hits zero\n"
           "and preservation is refuted (X): the entangled model breaks at dose "
           "32, the well-separated one holds longer (to ~40). Filled markers are "
           "doses that certifiably remove skill A; open markers (low doses) are "
           "too weak to\nremove it at all. Surgical and targeted edits (not shown) "
           "would be flat lines pinned at the natural margin. Companion to the "
           "2-point paper figure `p3_dose_collateral`.")
    ax.text(0.5, -0.20, cap.replace("**", ""), transform=ax.transAxes,
            ha="center", va="top", fontsize=8.0, color=INK2, linespacing=1.5)
    fig.tight_layout()
    _save(fig, "p3b_dose_collateral_dense")


# ---------------------------------------------------------------------------
# P8 — bound propagation validated against the exact truth (M0)
# ---------------------------------------------------------------------------
def fig_p8():
    """M0: auto_LiRPA (CROWN) vs exact Z3 on the toy ReLU MLP — the one subject
    where both apply. Every claim lands on the identity line: CROWN equals the
    exact radius (sound AND tight). Nothing sits above the line (that would be
    over-certifying = unsound). This licenses trusting CROWN past the frontier
    (P9), where Z3 cannot follow."""
    with open(os.path.join(RESULTS, "boundprop_validate_report.json")) as f:
        rep = json.load(f)
    rows = rep["rows"]

    def nice(claim):
        if claim.startswith("removal"):
            return "removal (A)"
        return "preservation (B > 0)" if "B positive" in claim \
            else "preservation (B ≤ 0)"

    z3 = [r["z3"] for r in rows]
    al = [r["alirpa"] for r in rows]
    names = [nice(r["claim"]) for r in rows]
    lim = max(max(z3), max(al)) * 1.12

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    # unsound zone: above the identity line (CROWN > exact) — never occurs
    ax.fill_between([0, lim], [0, lim], [lim, lim], color=CRIT, alpha=0.06,
                    zorder=0, lw=0)
    ax.text(lim * 0.30, lim * 0.82, "unsound zone\n(CROWN > exact —\nnever occurs)",
            color=CRIT, fontsize=8.2, ha="center", va="center")
    # identity line
    ax.plot([0, lim], [0, lim], color=MUTED, ls=(0, (5, 4)), lw=1.2, zorder=1)
    ax.text(lim * 0.62, lim * 0.62, "CROWN = exact", color=INK2, fontsize=8.8,
            rotation=45, rotation_mode="anchor", ha="center", va="bottom")
    # points, de-duplicating the two coincident preservation claims
    seen = {}
    for x, yv, nm in zip(z3, al, names):
        seen.setdefault((round(x, 4), round(yv, 4)), []).append(nm)
    ax.scatter([k[0] for k in seen], [k[1] for k in seen], s=95, color=CLEAN,
               edgecolor=SURFACE, linewidth=1.3, zorder=4)
    for (x, yv), nms in seen.items():
        lbl = " /\n".join(dict.fromkeys(nms))
        capped = abs(x - 0.5) < 1e-6                 # removal saturates eps_max
        head = f"{'≥ ' if capped else ''}{x:.3f}{'  (capped)' if capped else ''}"
        ax.annotate(f"{lbl}\n{head}", (x, yv), textcoords="offset points",
                    xytext=(11, -4), ha="left", va="top", fontsize=8.2,
                    color=INK2)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("exact Z3 certified radius")
    ax.set_ylabel("auto_LiRPA (CROWN) certified radius")
    ax.set_title("Bound propagation matches the exact truth (M0)", pad=26)
    ax.text(0.5, 1.02, f"toy ReLU MLP — both tools apply · forward agreement "
            f"{rep['forward_gap']:.0e}", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8.2, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRIDL, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    cap = ("The exact solver gives the true certified radius; a sound bound "
           "method must never exceed it\n(soundness) and ideally equals it "
           "(tightness). On every claim the two coincide — CROWN is sound\n"
           "AND tight where both apply, which licenses its use on the "
           "softmax+LayerNorm transformer (P9),\nwhere the exact solver cannot "
           "follow.")
    ax.text(0.5, -0.20, cap, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.3, color=INK2, linespacing=1.5)
    fig.tight_layout()
    _save(fig, "p8_boundprop_validate")


# ---------------------------------------------------------------------------
# P7 — a known-formula edit, both sides certified (modular adders)
# ---------------------------------------------------------------------------
def fig_p7():
    """A known-formula edit, BOTH sides certified over continuous noise.
    Two modular adders on disjoint positions [a1 b1 | a2 b2 | TASK]; ablating
    skill A's single head certifies exact summand-independence (radius ≥ 0.05,
    control refuted), and skill B is preserved — certified exactly over all 625
    sequences at ε=0 and noise-robustly to radius ≥ 0.002."""
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    with open(os.path.join(RESULTS, "rung3_report.json")) as f:
        r3 = json.load(f)
    with open(os.path.join(RESULTS, "rung3_preservation_report.json")) as f:
        pr = json.load(f)
    p = r3["p"]
    rem = 0.05                              # r3 removal radius (≥ 0.05, capped)
    pres = pr["radius"]                     # 0.002
    n_sub = pr["n_sub"]

    fig = plt.figure(figsize=(8.8, 5.0))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.35], hspace=0.62)
    ax_s = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # --- (a) schematic: the sequence and the disjoint skills ---
    toks = ["a₁", "b₁", "a₂", "b₂", "TASK"]
    face = [CLEAN, CLEAN, COLLAT, COLLAT, MUTED]
    for i, (t, col) in enumerate(zip(toks, face)):
        ax_s.add_patch(FancyBboxPatch((i + 0.08, 0.30), 0.84, 0.5,
                       boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.4, edgecolor=col,
                       facecolor=col, alpha=0.16, zorder=2))
        ax_s.text(i + 0.5, 0.55, t, ha="center", va="center", fontsize=11,
                  color=INK, zorder=3)
    ax_s.text(1.0, 1.04, "skill A  (removed)", ha="center", va="bottom",
              fontsize=9.2, color=CLEAN, fontweight="bold")
    ax_s.text(3.0, 1.04, "skill B  (preserved)", ha="center", va="bottom",
              fontsize=9.2, color=COLLAT, fontweight="bold")
    ax_s.plot([0.08, 1.92], [0.16, 0.16], color=CLEAN, lw=2.2)
    ax_s.plot([2.08, 3.92], [0.16, 0.16], color=COLLAT, lw=2.2)
    ax_s.set_xlim(-0.1, 5.1)
    ax_s.set_ylim(0, 1.35)
    ax_s.axis("off")

    # --- (b) the two certified radii ---
    y = [1, 0]
    vals = [rem, pres]
    cols = [CLEAN, COLLAT]
    labs = ["Removal\nskill A: (a₁+b₁) mod %d" % p,
            "Preservation\nskill B: (a₂+b₂) mod %d" % p]
    ax_b.barh(y, vals, height=0.5, color=cols, zorder=3)
    ax_b.text(rem, 1, f"  ≥ {rem:g}", va="center", ha="left", fontsize=9,
              color=INK)
    ax_b.text(pres, 0, f"  ≥ {pres:g}", va="center", ha="left", fontsize=9,
              color=INK)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labs, fontsize=9)
    ax_b.set_xlim(0, rem * 1.32)
    ax_b.set_ylim(-0.6, 1.6)
    ax_b.set_xlabel("certified radius — embedding-space noise (L∞), over all "
                    "sequences × continuous noise")
    for s in ("top", "right", "left"):
        ax_b.spines[s].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    ax_b.grid(axis="x", color=GRIDL, lw=0.6, zorder=0)
    ax_b.set_axisbelow(True)

    fig.suptitle("A known-formula edit, both sides certified over noise "
                 "(modular adders)", fontsize=11.5, fontweight="bold", y=0.99)
    cap = ("Removal is the strong claim: the edited output is provably "
           "independent of the summands (a₁,b₁) over all sequences × "
           "noise, so the model\nno longer reads the numbers it should add — and "
           "the unedited control is correctly refuted. Preservation is certified "
           f"two ways: exact-rational\ncorrectness over all {n_sub} clean "
           "sequences (ε=0) and noise-robustness to radius ≥ %g. Its smaller "
           "radius is an exact-solver frontier (it\nlacks removal's two-copy "
           "noise-cancellation), not fragility." % pres)
    ax_b.text(0.5, -0.42, cap, transform=ax_b.transAxes, ha="center", va="top",
              fontsize=8.3, color=INK2, linespacing=1.5)
    _save(fig, "p7_adders_both_sides")


# ---------------------------------------------------------------------------
# P9 — the certified edit past the exact frontier (bound propagation, M1)
# ---------------------------------------------------------------------------
def fig_p9():
    """M1, the scale headline: a certified edit on a standard softmax+LayerNorm
    transformer that the exact solver cannot encode at all, at 448 noise
    variables (~9x the exact-Z3 frontier). Each claim is a SOUND certified radius
    (auto_LiRPA / CROWN) bracketed above by a PGD attack — read as
    certified <= true <= attack. The wide gaps are honest bound-propagation
    looseness; no attack breaks below a certified radius (nothing unsound)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    with open(os.path.join(RESULTS, "boundprop_transformer_report.json")) as f:
        rep = json.load(f)
    rows, cfg = rep["rows"], rep["config"]

    def nice(claim):
        if claim.startswith("removal"):
            return "Removal\nskill A ≤ 0  (all quote inputs)"
        if "> 0" in claim:
            return "Preservation\nskill B > 0  (bracket: more open)"
        return "Preservation\nskill B ≤ 0  (bracket: more close)"

    labels = [nice(r["claim"]) for r in rows]
    cert = [r["certified"] for r in rows]
    pgd = [r["pgd"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.9, 4.3))
    y = np.arange(len(rows))[::-1]
    xmax = max(pgd) * 1.14
    for yi, c, p in zip(y, cert, pgd):
        # unproven gap: certified -> attack (true radius lies in here)
        ax.plot([c, p], [yi, yi], color=MUTED, lw=1.4, ls=(0, (1, 1.6)),
                zorder=3)
        # certified-safe (sound) region: solid blue bar anchored at 0
        ax.barh(yi, c, height=0.46, color=CLEAN, zorder=4)
        # the PGD attack break: critical-red X (distinct marker + label)
        ax.plot(p, yi, marker="X", ms=11, color=CRIT, mec=SURFACE, mew=0.9,
                zorder=5)
        # bare-number direct labels (the legend carries blue=certified / red=attack)
        ax.text(0.0, yi + 0.30, f"{c:.4f}", ha="left", va="bottom",
                fontsize=8.2, color=INK)
        ax.text(p, yi + 0.30, f"{p:g}", ha="center", va="bottom",
                fontsize=8.2, color=CRIT)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("embedding-space perturbation radius  ε  (L∞)")
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_title("A certified edit past the exact-solver frontier "
                 "(softmax + LayerNorm)", pad=32)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRIDL, lw=0.6, zorder=1)
    ax.set_axisbelow(True)

    # subtitle strip, centred below the (raised) title
    ax.text(0.5, 1.03, f"standard transformer · {cfg['noise_vars']} noise "
            f"variables (~9× the exact-Z3 frontier) · sound CROWN, PGD-bracketed",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2,
            color=INK2)

    legend = [
        Patch(color=CLEAN, label="certified safe — sound lower bound (CROWN)"),
        Line2D([0], [0], color=MUTED, lw=1.4, ls=(0, (1, 1.6)),
               label="unproven gap — the true radius lies here"),
        Line2D([0], [0], color=CRIT, marker="X", ms=9, lw=0, mec=SURFACE,
               mew=0.8, label="first PGD attack success — upper bound"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False, fontsize=8.3, handletextpad=0.5,
              columnspacing=1.4)
    cap = ("Each row reads certified ≤ true ≤ attack: the solid bar is the "
           "sound certified radius (nothing in that\nembedding ball breaks the "
           "claim, for every sequence of the class); the red mark is where a PGD "
           "attack first\nsucceeds. The wide gap is bound-propagation looseness "
           "through softmax/LayerNorm — expected, and the\nreason the exact "
           "pipeline is worth building where it reaches; no attack breaks below a "
           "certified radius.")
    ax.text(0.5, -0.32, cap, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.3, color=INK2, linespacing=1.5)
    fig.tight_layout()
    _save(fig, "p9_boundprop_bracket")


# ---------------------------------------------------------------------------
# P10 — discrete closure vs the continuous ball (the honest quantifier boundary)
# ---------------------------------------------------------------------------
def fig_p10():
    """The two complementary guarantees of the M1 edit, honestly to scale.
    Removal is certified over the ENTIRE TQ class (any in-vocabulary content
    rewriting) — a discrete guarantee — AND over a continuous embedding ball of
    radius 0.0091. The smallest in-vocabulary token substitution is 2.94 in the
    same L∞ metric, ~324x the certified radius: the continuous ball is not a
    disguised discrete-robustness claim."""
    with open(os.path.join(RESULTS, "boundprop_quantifier_report.json")) as f:
        rep = json.load(f)
    cert = rep["rows"][0]["certified"]            # removal over all TQ sequences
    sep = rep["token_separation"]
    ratio = rep["ratio"]

    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    y0 = 0.55
    # the continuous certified ball: a marker at its reach on the log axis
    ax.plot([cert], [y0], marker="o", ms=12, color=CLEAN, mec=SURFACE, mew=1.4,
            zorder=5)
    ax.text(cert * 1.35, y0 + 0.14, f"certified continuous radius {cert:.4f}\n"
            "(sound; over all 64 TQ sequences)", ha="left", va="bottom",
            fontsize=8.6, color=CLEAN)
    # the in-vocabulary token swaps: a band from the nearest to the farthest pair
    ax.hlines(y0, sep["min"], sep["max"], lw=9, color=CRIT, zorder=4,
              capstyle="round")
    ax.plot([sep["min"]], [y0], marker="o", ms=9, color=CRIT, mec=SURFACE,
            mew=1.2, zorder=5)
    ax.text(np.sqrt(sep["min"] * sep["max"]), y0 + 0.14,
            f"in-vocabulary token swaps\n(nearest {sep['min']:.2f}, of "
            f"{sep['pairs']} pairs)", ha="center", va="bottom", fontsize=8.6,
            color=CRIT)
    # the gap between them
    ax.annotate("", xy=(sep["min"], y0 - 0.13), xytext=(cert, y0 - 0.13),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.4))
    ax.text(np.sqrt(cert * sep["min"]), y0 - 0.30, f"≈ {ratio:.0f}×",
            ha="center", va="top", fontsize=11, color=INK, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(cert * 0.45, sep["max"] * 1.8)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("embedding-space distance  ε  (L∞, log scale)")
    ax.set_title("The certified ball and a discrete token substitution, to scale",
                 pad=22)
    ax.text(0.5, 1.04, "removal is separately certified over the entire TQ class "
            "(any content rewriting); the continuous ball is a much smaller, "
            "distinct guarantee", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8.2, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRIDL, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    cap = ("Two complementary guarantees. Discrete: removal holds for every one "
           "of the 64 TQ sequences — any in-vocabulary rewriting of the "
           "content.\nContinuous: a certified embedding ball of radius "
           f"{cert:.4f}. The smallest in-vocabulary token substitution is "
           f"{sep['min']:.2f} in the same metric —\nabout {ratio:.0f}× the "
           "certified radius — so the continuous ball is not a disguised "
           "discrete-robustness claim; neither is oversold.")
    ax.text(0.5, -0.40, cap, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.3, color=INK2, linespacing=1.5)
    fig.tight_layout()
    _save(fig, "p10_discrete_vs_continuous")



# ===========================================================================
# PAPER MODE — submission figures, authored at final size
# ---------------------------------------------------------------------------
# The wide figures above render their text at 8.5-12 pt on a 6-10 in canvas;
# scaled into a 3.5 in IEEEtran column that text lands at ~3 pt. The builders
# below are laid out directly at column or full width with 7.5-8 pt text, and
# are saved at exactly that canvas size (no tight bbox), so the size in the PDF
# is the size on the page. All prose belongs in \caption, not in the image.
# ===========================================================================
PAPER_COLUMN = 3.45          # in; IEEEtran \columnwidth is ~3.49 in
PAPER_WIDE = 7.05            # in; IEEEtran \textwidth is ~7.16 in
PAPER_DIR = os.path.join(ROOT, "paper", "figures")
PAPER_TEXT_MIN_PT = 6.5      # floor for any text (Nimbus Roman) at final size
PAPER_NOTE_PT = 6.5          # data-value labels: smaller than the 7.5-8 pt
                             #   labels/ticks, and the smallest used for print
PAPER_SCRIPT_MIN_PT = 5.5    # floor for math glyphs: TeX sets sub/superscripts
                             #   at ~70% of base, as in the paper's 8 pt captions
PAPER_LINEWIDTH_IN = 3.49    # IEEEtran \columnwidth, for the size check
PAPER_TEXTWIDTH_IN = 7.16    # IEEEtran \textwidth
PAPER_RC = {
    # Typeface matches the paper: IEEEtran body text is URW Nimbus Roman (the
    # Times clone the PDF embeds as NimbusRomNo9L) and its math is Computer
    # Modern. STIXGeneral (Times-like, bundled with matplotlib) is the per-glyph
    # fallback. Size follows IEEEtran's 8 pt captions, not the 10 pt body --
    # figure text at body size is not the convention and would crowd a column.
    "font.family": ["Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "cm",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "axes.titleweight": "normal",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "axes.labelpad": 2.5,
    "lines.linewidth": 1.0,
    "xtick.color": INK2,         # darker than the on-screen MUTED: 7 pt in print
    "ytick.color": INK2,         #   needs the contrast, esp. categorical labels
    "pdf.fonttype": 42,          # embed TrueType, keeps text selectable/measurable
}


def paper_figure(fn):
    """Decorator: build a paper figure under the paper rc settings."""
    def wrapped():
        with plt.rc_context(PAPER_RC):
            fn()
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    return wrapped


def _check_paper_pdf(path, include_width_in):
    """Acceptance check on a written paper figure: measure every text span in
    the PDF at the size it will print, given the width LaTeX includes it at.
    Text (the Nimbus Roman face) must be >= PAPER_TEXT_MIN_PT; math glyphs
    (Computer Modern) >= PAPER_SCRIPT_MIN_PT. Raises if either floor fails.
    Needs PyMuPDF; without it the check is skipped (it verifies figures, it
    proves nothing), so building figures only ever requires matplotlib."""
    try:
        import fitz
    except ImportError:
        return "skipped (PyMuPDF not installed)"
    page = fitz.open(path)[0]
    scale = include_width_in / (page.rect.width / 72.0)
    text_min = math_min = float("inf")
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if not span["text"].strip():
                    continue
                size = span["size"] * scale
                if span["font"].lower().startswith("cm"):
                    math_min = min(math_min, size)
                else:
                    text_min = min(text_min, size)
    ok = (text_min >= PAPER_TEXT_MIN_PT) and (math_min >= PAPER_SCRIPT_MIN_PT)
    fmt = lambda v: "-" if v == float("inf") else f"{v:.2f}"
    msg = (f"text min {fmt(text_min)} pt (floor {PAPER_TEXT_MIN_PT}), "
           f"math min {fmt(math_min)} pt (floor {PAPER_SCRIPT_MIN_PT})")
    if not ok:
        raise RuntimeError(f"paper figure {path} fails size check: {msg}")
    return msg


def _save_paper(fig, name, wide=False):
    """Save at the exact canvas size (the paper size) plus a PNG preview, then
    verify the printed text sizes. wide=True for figure* (full text width)."""
    os.makedirs(PAPER_DIR, exist_ok=True)
    pdf = os.path.join(PAPER_DIR, f"{name}.pdf")
    fig.savefig(pdf, metadata={"CreationDate": None})   # byte-stable rebuilds
    fig.savefig(os.path.join(HERE, f"paper_{name}.png"), dpi=300)
    w, h = fig.get_size_inches()
    plt.close(fig)
    check = _check_paper_pdf(
        pdf, PAPER_TEXTWIDTH_IN if wide else PAPER_LINEWIDTH_IN)
    status = "" if check.startswith("skipped") else "PASS: "
    print(f"  wrote paper/figures/{name}.pdf  ({w:.2f} x {h:.2f} in)"
          f"  + figures/paper_{name}.png\n  size check {status}{check}")


@paper_figure
def paper_bp():
    """fig:bp -- the headline. Certified removal and preservation on a standard
    softmax+LayerNorm transformer (M1, 448 noise variables). Encoding, explained
    in the caption rather than in a legend: the solid bar is the sound certified
    radius from bound propagation, the X is the first PGD attack success, and
    the dotted span between them is where the true radius lies."""
    with open(os.path.join(RESULTS, "boundprop_transformer_report.json")) as f:
        rows = json.load(f)["rows"]

    def lab(claim):
        if claim.startswith("removal"):
            return r"removal ($A \leq 0$)"
        return (r"preserve ($B > 0$)" if "> 0" in claim
                else r"preserve ($B \leq 0$)")

    fig = plt.figure(figsize=(PAPER_COLUMN, 1.6), layout="constrained")
    ax = fig.add_subplot()
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        c, p = r["certified"], r["pgd"]
        ax.plot([c, p], [yi, yi], color=MUTED, lw=0.9, ls=(0, (1, 1.5)),
                zorder=3)
        ax.barh(yi, c, height=0.42, color=CLEAN, zorder=4)
        ax.plot(p, yi, marker="X", ms=6.5, color=CRIT, mec=SURFACE, mew=0.6,
                zorder=5)
        ax.text(0, yi + 0.27, f"{c:.4f}", ha="left", va="bottom", color=INK)
        ax.text(p, yi + 0.27, f"{p:g}", ha="center", va="bottom", color=CRIT)
    ax.set_yticks(y, [lab(r["claim"]) for r in rows])
    ax.set_xlim(0, max(r["pgd"] for r in rows) * 1.12)
    ax.set_ylim(-0.55, len(rows) - 0.15)
    ax.set_xlabel(r"embedding perturbation radius $\varepsilon$  ($L_\infty$)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRIDL, lw=0.5, zorder=1)
    ax.set_axisbelow(True)
    _save_paper(fig, "fig_bp")


@paper_figure
def paper_illusion():
    """fig:illusion -- Fig. 1, the paper's hook and Proposition 3's witness.
    Full width, three panels. (a) Skill A's removal region with the passing test
    grid and the solver's surviving input. (b) The edited model's skill-A output
    along x0 through the survivor, against the fine test grid: the tent gadget of
    Proposition 3 (folded in from the former fig:tent), showing a survivor band
    narrower than the grid spacing. (c) One automated search over trained models,
    80 per input dimension (run_illusion_dims.py): no illusion at two or three
    inputs, three at four and five, so the illusion is not an artefact of hand
    construction. The edited output depends
    on x0 alone, so the slice in (b) holds for every x1."""
    from run_illusion import REGION_A_HIGH, COARSE_N, build_constructed_model
    with open(os.path.join(RESULTS, "illusion_report.json")) as f:
        rep = json.load(f)
    con, two = rep["route_B_constructed"], rep["route_A_search"]
    with open(os.path.join(RESULTS, "illusion_nd_report.json")) as f:
        nd = json.load(f)
    cx0, cx1 = con["counterexample"]["x0"], con["counterexample"]["x1"]
    c, half = con["sliver_center"], con["sliver_active_halfwidth"]
    lo0, hi0, lo1, hi1 = REGION_A_HIGH
    fine_dx = (hi0 - lo0) / (201 - 1)          # the 201 x 201 grid's spacing
    model, edit = build_constructed_model()

    def out(xs):
        xs = np.asarray(xs, dtype=float)
        return model.forward(np.column_stack([xs, np.full_like(xs, cx1)]),
                             ablate=edit)[:, 0]

    fig = plt.figure(figsize=(PAPER_WIDE, 2.05), layout="constrained")
    axA, axB, axC = fig.subplots(
        1, 3, gridspec_kw={"width_ratios": [1.0, 1.3, 0.9]})
    zlo, zhi = c - 0.004, c + 0.004           # the x0 range drawn in (b)

    # ---- (a) the region: every test point reports the skill removed -------
    g0, g1 = np.meshgrid(np.linspace(lo0, hi0, COARSE_N),
                         np.linspace(lo1, hi1, COARSE_N))
    axA.scatter(g0, g1, s=4, c=TEST, alpha=0.65, linewidths=0, zorder=2)
    axA.axvline(c, color=CRIT, lw=0.8, zorder=4)   # the survivor band, a hairline
    axA.scatter([cx0], [cx1], marker="*", s=60, c=CRIT, edgecolors=SURFACE,
                linewidths=0.5, zorder=5, clip_on=False)
    axA.annotate("survivor", xy=(cx0, cx1), xytext=(cx0 + 0.035, 0.16),
                 color=CRIT, ha="left", va="center",
                 bbox=dict(fc=SURFACE, ec="none", pad=0.4),
                 arrowprops=dict(arrowstyle="-", color=CRIT, lw=0.6))
    axA.set_xlim(lo0 - 0.012, hi0 + 0.012)
    axA.set_ylim(lo1 - 0.05, hi1 + 0.05)
    axA.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    axA.set_yticks([0, 0.5, 1])
    axA.set_xlabel(r"$x_0$")
    axA.set_ylabel(r"$x_1$")
    axA.set_title("(a)", loc="left")

    # ---- (b) the tent: the survivor band falls between grid points ------
    xs = np.linspace(zlo, zhi, 8000)
    ys = out(xs)
    axB.axhline(0, color=BASELINE, lw=0.6, zorder=1)
    axB.fill_between(xs, 0, ys, where=ys > 0, color=CRIT, alpha=0.28, lw=0,
                     zorder=2)
    axB.plot(xs, ys, color=INK2, lw=0.9, zorder=3)
    k = np.arange(np.ceil((zlo - lo0) / fine_dx),
                  np.floor((zhi - lo0) / fine_dx) + 1)
    fx = lo0 + fine_dx * k
    axB.scatter(fx, out(fx), s=14, c=TEST, edgecolors=SURFACE,
                linewidths=0.5, zorder=4)
    ylo, yhi = float(ys.min()), float(ys.max())
    span = yhi - ylo
    # dimension line: two neighboring grid points, straddling the survivor
    g_left = fx[fx < c].max()
    y_dim = ylo - 0.30 * span
    axB.annotate("", xy=(g_left, y_dim), xytext=(g_left + fine_dx, y_dim),
                 arrowprops=dict(arrowstyle="<->", color=TEST, lw=0.6,
                                 shrinkA=0, shrinkB=0))
    axB.text(g_left + fine_dx / 2, y_dim - 0.06 * span,
             f"grid spacing {fine_dx:g}", color=TEST, ha="center", va="top")
    axB.text(c + 1.6 * half, yhi * 0.62,
             f"survivor band, width {2 * half:g}", color=CRIT, ha="left",
             va="center")
    axB.set_xlim(zlo, zhi)
    axB.set_ylim(ylo - 0.62 * span, yhi + 0.18 * span)
    axB.set_xticks(np.round(np.arange(zlo, zhi + 1e-12, 0.002), 3))
    axB.set_yticks([0])
    axB.set_xlabel(r"$x_0$")
    axB.set_ylabel(r"skill $A$ output after edit")
    for side in ("top", "right"):
        axB.spines[side].set_visible(False)
    axB.set_title("(b)", loc="left")

    # ---- (c) trained models, one bar per input dimension ------------------
    # One protocol for every bar: run_illusion_dims.py re-runs the nd search a
    # dimension at a time (80 models each); its d=4,5 rows reproduce the
    # committed illusion_nd_report.json exactly.
    with open(os.path.join(RESULTS, "illusion_dims_report.json")) as f:
        dims = json.load(f)["by_dimension"]
    xd = [r["d"] for r in dims]
    n_ill = [r["n_illusions"] for r in dims]
    axC.bar(xd, n_ill, width=0.55, color=CRIT, zorder=3)
    for xi, r in zip(xd, dims):
        axC.text(xi, r["n_illusions"] + 0.08,
                 f"{r['n_illusions']}/{r['edits_test_approved']}",
                 ha="center", va="bottom", color=INK2)
    axC.set_xticks(xd)
    axC.set_xlim(min(xd) - 0.6, max(xd) + 0.6)
    axC.set_ylim(0, max(n_ill) + 0.75)
    axC.set_yticks(range(0, max(n_ill) + 1))
    axC.set_xlabel("inputs")
    axC.set_ylabel("illusions found")
    for side in ("top", "right"):
        axC.spines[side].set_visible(False)
    axC.grid(axis="y", color=GRIDL, lw=0.5, zorder=1)
    axC.set_axisbelow(True)
    axC.set_title("(c)", loc="left")
    _save_paper(fig, "fig_illusion", wide=True)


@paper_figure
def paper_edits():
    """fig:edits -- A2, surgery vs steering with certified collateral (P4).
    Full width, two panels, on the separated and entangled toy models. The two
    sources describe the same models: robustness_report.json's unedited radii
    and every dose it shares with collateral_sweep.json agree exactly.
    (a) Skill B's certified preservation radius after each edit type. Ablation,
    weight edit and targeted steering land on each model's unedited radius
    (zero collateral, which P4(d) forces for a circuit disjoint from head B);
    diff-of-means falls short even at the smallest dose that certifiably
    removes skill A. A dot plot on a zoomed axis, because the values differ by
    thousandths and bars drawn from zero would look equal.
    (b) The same radius against diff-of-means dose, on a linear axis (dose 0 is
    the unedited model, and the claim is a straight line). Every marker is a
    separate proof: filled where the dose also removes skill A, open where it is
    too weak; a cross where preservation is refuted. Lines are least-squares
    fits to the certified points, not the theorem's prediction: P4(d) makes the
    logit shift linear in dose, and the certified radius tracks it.
    Model identity is carried by marker shape and line style as well as colour,
    so the figure survives greyscale print."""
    from matplotlib.lines import Line2D
    with open(os.path.join(RESULTS, "robustness_report.json")) as f:
        rob = json.load(f)
    with open(os.path.join(RESULTS, "collateral_sweep.json")) as f:
        sweep = json.load(f)
    models = [("tidy", "separated model", CLEAN, "o", "-"),
              ("messy", "entangled model", COLLAT, "s", (0, (4, 2)))]
    fig = plt.figure(figsize=(PAPER_WIDE, 2.3), layout="constrained")
    axA, axB = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.45]})

    # ---- (a) the radius each edit type leaves for skill B ----------------
    kinds = ["ablation", "weight edit", "targeted steering",
             "diff-of-means steering"]
    shown = ["ablation", "weight edit", "targeted steering", "diff-of-means"]
    y = np.arange(len(kinds))[::-1]
    off = {"tidy": 0.14, "messy": -0.14}
    uned = {}
    for key, name, col, mk, ls in models:
        rows = sweep[key]
        uned[key] = next(r["preservation_radius"] for r in rows
                         if r["dose"] == 0)
        by = _p2_by_type(rob[key]["rows"])
        first = min((r for r in rows if r["removes_A"]
                     and r["preservation_radius"] is not None),
                    key=lambda r: r["dose"])
        vals = [by[k][0] for k in kinds[:3]] + [first["preservation_radius"]]
        axA.plot([uned[key], uned[key]], [-0.5, len(kinds) - 0.62], color=col,
                 lw=0.7, ls=(0, (1, 1.5)), zorder=1)   # stops below the label
        axA.scatter(vals, y + off[key], marker=mk, s=24, color=col,
                    edgecolors=SURFACE, linewidths=0.4, zorder=3, label=name)
        for v, yy in zip(vals, y + off[key]):
            axA.text(v, yy - 0.11, f"{v:.4f}", ha="center", va="top",
                     color=col, fontsize=PAPER_NOTE_PT, zorder=4,
                     bbox=dict(fc=SURFACE, ec="none", pad=0.15))
        axA.text(vals[-1] - 0.0009, y[-1] + off[key], f"dose {first['dose']:g}",
                 ha="right", va="center", color=col)
    axA.set_yticks(y, shown)
    axA.set_ylim(-0.6, len(kinds) - 0.2)
    axA.set_xlim(0.070, 0.1035)
    axA.set_xticks([0.07, 0.08, 0.09, 0.10])
    axA.set_xlabel(r"skill $B$ certified radius")
    axA.text(sum(uned.values()) / 2, len(kinds) - 0.45, "unedited",
             ha="center", va="bottom", color=INK2)
    axA.legend(loc="upper left", frameon=False, handletextpad=0.2,
               borderaxespad=0.1, labelspacing=0.3)
    for side in ("top", "right", "left"):
        axA.spines[side].set_visible(False)
    axA.tick_params(axis="y", length=0)
    axA.grid(axis="x", color=GRIDL, lw=0.5, zorder=0)
    axA.set_axisbelow(True)
    axA.set_title("(a)", loc="left")

    # ---- (b) the dose response of diff-of-means steering ----------------
    fits = {}
    for key, name, col, mk, ls in models:
        rows = sweep[key]
        cert = [r for r in rows if r["preservation_radius"] is not None]
        xs = np.array([r["dose"] for r in cert], float)
        ys = np.array([r["preservation_radius"] for r in cert], float)
        a, b = np.polyfit(xs, ys, 1)
        fits[key] = 1 - ((ys - (a * xs + b)) ** 2).sum() / \
            ((ys - ys.mean()) ** 2).sum()
        xf = np.array([0.0, xs.max()])
        axB.plot(xf, a * xf + b, color=col, lw=0.8, ls=ls, zorder=2)
        for r in cert:
            axB.scatter([r["dose"]], [r["preservation_radius"]], marker=mk,
                        s=18, facecolors=col if r["removes_A"] else SURFACE,
                        edgecolors=col, linewidths=0.7, zorder=3)
        broken = [r["dose"] for r in rows if r["preservation_radius"] is None]
        if broken:
            # tie the refutation to its model: connector in the model's style
            axB.plot([xs.max(), min(broken)], [ys[xs.argmax()], 0.0],
                     color=col, lw=0.7, ls=(0, (1, 1.5)), zorder=2)
            axB.scatter([min(broken)], [0.0], marker="X", s=38, color=CRIT,
                        edgecolors=SURFACE, linewidths=0.4, zorder=4,
                        clip_on=False)
    print("  fig:edits linear fits, R^2 over certified points: " +
          ", ".join(f"{k} {v:.4f}" for k, v in fits.items()))
    axB.set_xlim(0, 50)
    axB.set_ylim(0, 0.105)
    axB.set_xlabel("diff-of-means steering dose")
    axB.set_ylabel(r"skill $B$ certified radius")
    for side in ("top", "right"):
        axB.spines[side].set_visible(False)
    axB.grid(color=GRIDL, lw=0.5, zorder=0)
    axB.set_axisbelow(True)
    state = [Line2D([0], [0], lw=0, marker="o", ms=4, color=MUTED,
                    label=r"removes $A$"),
             Line2D([0], [0], lw=0, marker="o", ms=4, mfc=SURFACE, mec=MUTED,
                    mew=0.7, label=r"too weak to remove $A$"),
             Line2D([0], [0], lw=0, marker="X", ms=5.5, color=CRIT,
                    mec=SURFACE, mew=0.4, label=r"preservation of $B$ refuted")]
    axB.legend(handles=state, loc="upper right", frameon=False,
               handletextpad=0.2, borderaxespad=0.1, labelspacing=0.3)
    axB.set_title("(b)", loc="left")
    _save_paper(fig, "fig_edits", wide=True)


@paper_figure
def paper_circuit():
    """fig:p5exact -- the exact transformer's distributed circuit (panel (a) of
    the on-screen p5; its radii now live in Table II). One decoder block drawn
    as the forward pass actually runs: each block reads the residual stream and
    adds its output back (x1 = x + attn, x2 = x1 + MLP), then a linear readout.
    Skill A's circuit, from run_transformer.find_circuit (a greedy search over
    heads and MLP neurons that keeps skill B above the test gate), is filled
    dark with white text -- a cue that survives greyscale -- and named below.
    Neither half alone removes skill A (rung2_circuit_search.log, seed 5: both
    heads off leaves A's worst logit at +1.80, the whole MLP off at +17.20)."""
    from matplotlib.patches import FancyBboxPatch
    from run_transformer import SUBJECT_CONFIG
    with open(os.path.join(RESULTS, "transformer_report.json")) as f:
        circuit = [tuple(c) for c in json.load(f)["circuit"]]
    c_heads = sorted(i for k, i in circuit if k == "head")
    c_mlp = sorted(i for k, i in circuit if k == "mlp")
    H, M = SUBJECT_CONFIG["n_heads"], SUBJECT_CONFIG["d_mlp"]
    ON, OFF = BAD, "#dde6f2"

    fig = plt.figure(figsize=(PAPER_COLUMN, 1.62), layout="constrained")
    ax = fig.add_subplot()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.7)
    ax.axis("off")
    ys = 1.45                                     # the residual stream

    def box(x, y, w, h, fc, ec=BASELINE, lw=0.7, z=2, r=0.10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, lw=lw,
                     zorder=z, boxstyle=f"round,pad=0.02,rounding_size={r}"))

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7,
                                    shrinkA=0, shrinkB=0, mutation_scale=6))

    def plus(x):
        ax.add_patch(plt.Circle((x, ys), 0.17, fc=SURFACE, ec=MUTED, lw=0.7,
                                zorder=3))
        ax.text(x, ys, "+", ha="center", va="center", color=INK2,
                fontsize=PAPER_NOTE_PT, zorder=4)

    # input and readout sit on the stream
    box(0.05, ys - 0.62, 1.30, 1.24, "#eef2f7")
    ax.text(0.70, ys, "tokens\n" + r"$+\,\varepsilon$", ha="center",
            va="center", color=INK2, linespacing=1.2)
    box(8.75, ys - 0.62, 1.20, 1.24, "#eef2f7")
    ax.text(9.35, ys, "readout", ha="center", va="center", color=INK2)
    # the stream, with the two residual additions
    xa, xm = 4.05, 7.75          # each write-back sits inside its block's span
    arrow(1.35, ys, 8.75, ys)
    plus(xa)
    plus(xm)
    ax.text(5.0, ys - 0.30, "residual stream", ha="center", va="top",
            color=MUTED, fontsize=PAPER_NOTE_PT)

    # attention block: reads the stream, writes back at the first +
    ax_, aw, ay, ah = 1.85, 2.45, 2.30, 1.75
    box(ax_, ay, aw, ah, "#f4f6f9")
    ax.text(ax_ + aw / 2, ay + ah + 0.12, "attention", ha="center",
            va="bottom", color=INK2)
    hw = (aw - 0.30 - 0.15 * (H - 1)) / H
    for h in range(H):
        on = h in c_heads
        box(ax_ + 0.15 + h * (hw + 0.15), ay + 0.40, hw, ah - 0.80,
            ON if on else OFF, ec=SURFACE, z=3, r=0.06)
        ax.text(ax_ + 0.15 + h * (hw + 0.15) + hw / 2, ay + ah / 2,
                f"head {h}", ha="center", va="center", zorder=4,
                color=SURFACE if on else INK2)
    arrow(ax_ + 0.30, ys, ax_ + 0.30, ay)
    arrow(xa, ay, xa, ys + 0.17)

    # MLP block: 2 rows of neurons, reads after the first +, writes at second
    mx, mw, my, mh = 4.70, 3.35, 2.30, 1.75
    box(mx, my, mw, mh, "#f4f6f9")
    ax.text(mx + mw / 2, my + mh + 0.12, f"MLP ({M} neurons)", ha="center",
            va="bottom", color=INK2)
    per_row = (M + 1) // 2
    nw, nh, gap = 0.62, 0.55, 0.15
    x0 = mx + (mw - (per_row * nw + (per_row - 1) * gap)) / 2
    for j in range(M):
        r, c = divmod(j, per_row)
        on = j in c_mlp
        nx, ny = x0 + c * (nw + gap), my + mh - 0.30 - (r + 1) * nh - r * 0.15
        box(nx, ny, nw, nh, ON if on else OFF, ec=SURFACE, z=3, r=0.05)
        ax.text(nx + nw / 2, ny + nh / 2, str(j), ha="center", va="center",
                fontsize=PAPER_NOTE_PT, zorder=4,
                color=SURFACE if on else INK2)
    arrow(mx + 0.30, ys, mx + 0.30, my)
    arrow(xm, my, xm, ys + 0.17)

    # name the circuit in words, so identity never rests on colour alone
    names = [f"head {', '.join(map(str, c_heads))}",
             f"MLP neurons {', '.join(map(str, c_mlp))}"]
    ax.text(5.0, 0.05, "skill A circuit: " + " + ".join(names), ha="center",
            va="bottom", color=ON)
    _save_paper(fig, "fig_circuit")


@paper_figure
def paper_influence():
    """fig:influence -- how the non-interference certificate is proved (the
    on-screen u5b, re-authored at column width; the influence NUMBERS stay in
    the section V-C text). Drawn from verify.prove_independence: the edited
    network g is encoded twice, over inputs x and y constrained to the same
    box and pinned equal on every coordinate except the free (forbidden) one;
    the solver is asked for a pair whose head-A outputs differ by more than
    kappa. unsat proves the head moves by at most kappa when only x0 changes;
    sat returns the pair. certified_influence bisects kappa between the two.
    Free cells are hatched (not just tinted), so the distinction survives
    greyscale."""
    from matplotlib.patches import FancyBboxPatch
    FREE, TIED = "#fbe7e7", "#eef2f7"

    fig = plt.figure(figsize=(PAPER_COLUMN, 1.80), layout="constrained")
    ax = fig.add_subplot()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, fc, ec=BASELINE, lw=0.7, z=2, r=0.10, hatch=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, lw=lw,
                     zorder=z, hatch=hatch,
                     boxstyle=f"round,pad=0.02,rounding_size={r}"))

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7,
                                    shrinkA=0, shrinkB=0, mutation_scale=6))

    # ---- the input pair: free forbidden coordinate, every other one tied --
    plt.rcParams["hatch.color"] = CRIT
    plt.rcParams["hatch.linewidth"] = 0.4
    rows = {"x": 4.15, "y": 2.95}                 # row centres
    ch = 0.78                                     # cell height
    fx, fw = 0.05, 1.25                           # free column
    tx, tw = 1.50, 1.75                           # tied column
    ax.text(fx + fw / 2, 4.75, "free", ha="center", va="bottom", color=CRIT)
    ax.text(tx + tw / 2, 4.75, "tied", ha="center", va="bottom", color=INK2)
    for v, yc in rows.items():
        box(fx, yc - ch / 2, fw, ch, FREE, ec=CRIT, hatch="////")
        ax.text(fx + fw / 2, yc, f"${v}_0$", ha="center", va="center",
                color=INK, zorder=4,
                bbox=dict(fc=FREE, ec="none", pad=0.6))
        box(tx, yc - ch / 2, tw, ch, TIED)
        ax.text(tx + tw / 2, yc, f"${v}_1,\\ldots,{v}_{{d-1}}$",
                ha="center", va="center", color=INK, zorder=4)
    ax.text(tx + tw / 2, (rows["x"] + rows["y"]) / 2, "$=$", ha="center",
            va="center", color=INK2)

    # ---- the same edited network, encoded once per copy ----
    gx, gw = 3.85, 1.15
    ax.text(gx + gw / 2, 4.75, "edited $g$", ha="center", va="bottom",
            color=INK2)
    for v, yc in rows.items():
        arrow(tx + tw, yc, gx, yc)
        box(gx, yc - ch / 2, gw, ch, TIED)
        ax.text(gx + gw / 2, yc, f"$g_A({v})$", ha="center", va="center",
                color=INK, zorder=4)

    # ---- the comparator ----
    cx, cw, cy, chh = 5.75, 4.20, 3.55 - 0.45, 0.90
    box(cx, cy, cw, chh, SURFACE, ec=INK2)
    ax.text(cx + cw / 2, cy + chh / 2, r"$|g_A(x)-g_A(y)|>\kappa$ ?",
            ha="center", va="center", color=INK, zorder=4)
    arrow(gx + gw, rows["x"], cx, cy + chh * 0.72)
    arrow(gx + gw, rows["y"], cx, cy + chh * 0.28)

    # ---- the two answers, and the bisection that closes the gap ----
    oy, oh, ow = 0.55, 1.20, 4.10
    lx, rx = 1.45, 5.85
    box(lx, oy, ow, oh, INK2, ec=INK2)
    ax.text(lx + ow / 2, oy + oh / 2,
            "unsat: proved\n" + r"influence $\leq\kappa$", ha="center",
            va="center", color=SURFACE, zorder=4, linespacing=1.25)
    box(rx, oy, ow, oh, SURFACE, ec=INK2)
    ax.text(rx + ow / 2, oy + oh / 2,
            "sat: a pair $(x, y)$\n" + r"refutes $\kappa$", ha="center",
            va="center", color=INK, zorder=4, linespacing=1.25)
    arrow(cx + 0.30, cy, lx + ow * 0.85, oy + oh + 0.05)
    arrow(rx + ow * 0.55, cy, rx + ow * 0.55, oy + oh + 0.05)
    ax.text((lx + rx + ow) / 2, 0.05,
            r"bisection on $\kappa$ gives the certified influence",
            ha="center", va="bottom", color=INK2, fontsize=PAPER_NOTE_PT)
    _save_paper(fig, "fig_influence")


@paper_figure
def paper_radius():
    """fig:radius -- A3, the certified radius drawn in input space (fills
    section V-D). Subject: the entangled toy model after the ablation edit, and
    the claim "skill B stays HIGH" over R = [0,1] x [0.6,1] -- the Table II entry
    0.0961. (a) the whole claim: R, and R inflated by the certified radius.
    (b) the strip where the inflated edge meets the model: the edited model's
    own skill-B boundary {g_B = 0} (traced from the float forward pass on a
    fine grid; piecewise linear, with a kink where a hidden unit switches),
    the spec's x1 = 0.5 rule, the proved edge at 0.6 - eps*, and the solver's
    counterexample one bisection step further out. The edge stops where the
    edited model's own boundary peaks (x0 = 0), 0.0039 above the rule.
    The subject is retrained exactly as run_robustness.py does (seed 4, the
    first entangled seed with a test-passing ablation; ~15 s) and checked against
    the committed report: same circuit, counterexample on the wrong side."""
    from matplotlib.patches import Rectangle, ConnectionPatch
    from tiny_model import TinyMLP, train
    from edits import apply_ablation
    with open(os.path.join(RESULTS, "robustness_report.json")) as f:
        rep = json.load(f)["messy"]
    row = next(r for r in rep["rows"] if r["edit"] == "ablation")
    ph = row["preservation_high"]
    eps, eps_fail = ph["radius"], ph["first_failure"]
    cex = ph["counterexample_at_failure"]
    lo, rule = 0.6, 0.5                          # R_B^high = [0,1] x [0.6, 1]
    edge = lo - eps

    base = train(TinyMLP(H=16, seed=4), l1=0.0, seed=5, verbose=False)
    g = apply_ablation(base, rep["circuit"])
    assert g.forward(np.array([cex]))[0, 1] <= 0, "counterexample not refuting"
    x0 = np.linspace(0, 1, 401)
    x1 = np.linspace(0.49, 0.51, 20001)
    zero = []
    for a in x0:                                  # top of {g_B <= 0} per x0
        b = g.forward(np.stack([np.full_like(x1, a), x1], 1))[:, 1]
        zero.append(x1[np.nonzero(b <= 0)[0].max()])
    zero = np.array(zero)
    assert zero.max() <= edge + 1e-3 and zero.max() > rule

    EDGE, FILL = CLEAN, "#c9d7ec"   # tint must survive print; #dde6f2 read as white
    plt.rcParams["hatch.color"] = "#6f93c8"
    plt.rcParams["hatch.linewidth"] = 0.4
    fig = plt.figure(figsize=(PAPER_COLUMN, 1.75), layout="constrained")
    axA, axB = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.0, 1.35]})

    # ---- (a) the claim region and its certified inflation ----
    # the certified region R_eps* is shaded; the band inflation adds to R
    # (between the certified edge and R's edge at 0.6) is also hatched, so it
    # stays distinguishable in greyscale
    axA.add_patch(Rectangle((0, edge), 1, 1 - edge, fc=FILL, ec="none",
                            zorder=0))
    axA.add_patch(Rectangle((0, edge), 1, lo - edge, fc="none", ec="none",
                            hatch="////", lw=0, zorder=1))
    axA.add_patch(Rectangle((0, lo), 1, 1 - lo, fill=False, ec=INK2, lw=0.8,
                            ls=(0, (3, 1.5)), zorder=3))
    axA.text(0.5, 0.80, "$R$", ha="center", va="center", color=INK)
    axA.plot([0, 1], [edge, edge], color=EDGE, lw=1.1, zorder=3)
    axA.axhline(rule, color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=2)
    axA.annotate("", xy=(0.80, edge), xytext=(0.80, lo), zorder=4,
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=0.6,
                                 shrinkA=0, shrinkB=0, mutation_scale=5))
    axA.text(0.76, (lo + edge) / 2, r"$\varepsilon^{*}$", ha="right",
             va="center", color=INK, zorder=4,
             bbox=dict(fc=FILL, ec="none", pad=0.4))
    axA.set_xlim(0, 1)
    axA.set_ylim(0.4, 1.0)
    axA.set_xticks([0, 1])
    axA.set_yticks([0.4, 0.5, 0.6, 1.0])
    axA.set_xlabel("$x_0$", labelpad=-4)
    axA.set_ylabel("$x_1$", labelpad=1)
    axA.set_title("(a)", loc="left")

    # ---- (b) where the certified edge stops ----
    axB.fill_between([0, 1], edge, 0.508, color=FILL, lw=0, zorder=0)
    axB.fill_between([0, 1], edge, 0.508, facecolor="none", hatch="////",
                     lw=0, zorder=1)       # all of (b)'s shading is added band
    axB.plot([0, 1], [edge, edge], color=EDGE, lw=1.1, zorder=3)
    axB.text(0.98, edge + 0.0003, r"certified edge $0.6-\varepsilon^{*}$",
             ha="right", va="bottom", color=EDGE, zorder=4,
             bbox=dict(fc=FILL, ec="none", pad=0.4))
    axB.plot(x0, zero, color=INK, lw=0.9, zorder=4)
    axB.text(0.33, 0.5024, "$g_B=0$", ha="left", va="bottom", color=INK)
    axB.axhline(rule, color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=2)
    axB.text(0.93, rule + 0.0002, "rule $x_1=0.5$", ha="right", va="bottom",
             color=INK2)
    axB.plot([cex[0]], [cex[1]], marker="*", ms=6.5, color=CRIT, mec=SURFACE,
             mew=0.4, zorder=6, clip_on=False)
    axB.annotate("counterexample", xy=(cex[0] + 0.02, cex[1]),
                 xytext=(0.10, 0.4987), color=CRIT, fontsize=PAPER_NOTE_PT,
                 va="center", zorder=6,
                 arrowprops=dict(arrowstyle="-", color=CRIT, lw=0.5,
                                 shrinkA=1, shrinkB=1))
    gap = edge - rule
    axB.annotate("", xy=(0.965, rule), xytext=(0.965, edge), zorder=5,
                 arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.5,
                                 shrinkA=0, shrinkB=0, mutation_scale=4))
    axB.text(0.93, rule + 0.0027, f"{gap:.4f}", ha="right", va="center",
             color=INK2,
             fontsize=PAPER_NOTE_PT)
    axB.set_xlim(0, 1)
    axB.set_ylim(0.4955, 0.508)
    axB.set_xticks([0, 1])
    axB.set_yticks([0.496, 0.500, 0.504, 0.508])
    axB.set_xlabel("$x_0$", labelpad=-4)
    axB.set_title("(b)", loc="left")
    for ax in (axA, axB):
        ax.tick_params(length=2, pad=1.5)
    _save_paper(fig, "fig_radius")


FIGURES = {"p1": fig_p1, "u1": fig_u1, "u1b": fig_u1b,
           "p2": fig_p2, "p2b": fig_p2b, "p3": fig_p3,
           "p4": fig_p4, "p5": fig_p5, "p6": fig_p6,
           "u2": fig_u2, "u3": fig_u3, "u4": fig_u4, "u5": fig_u5,
           "u5b": fig_u5b,
           "p7": fig_p7, "p8": fig_p8, "p9": fig_p9, "p10": fig_p10,
           "p3b": fig_p3b}


PAPER_FIGURES = {"bp": paper_bp, "illusion": paper_illusion,
                 "edits": paper_edits, "circuit": paper_circuit,
                 "influence": paper_influence, "radius": paper_radius}


def main():
    args = sys.argv[1:]
    if args and args[0] == "--paper":
        registry, which, kind = PAPER_FIGURES, args[1:], "paper figure"
    else:
        registry, which, kind = FIGURES, args, "figure"
    for key in (which or list(registry)):
        key = key.lower()
        if key not in registry:
            print(f"  unknown {kind} '{key}'; known: {', '.join(registry)}")
            continue
        print(f"building {kind} {key} ...")
        registry[key]()


if __name__ == "__main__":
    main()
