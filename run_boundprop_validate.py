"""
run_boundprop_validate.py — M0: does the bound-propagation pipeline agree with
   the EXACT Z3 truth? (paper §IV-B, check M0)
=============================================================================

Run it with (from the isolated bound-prop env — see requirements-boundprop.txt):
    .venv-boundprop/bin/python run_boundprop_validate.py

WHY THIS EXISTS
---------------
The bound-propagation extension moves past the exact-Z3 frontier (auto_LiRPA),
which is SOUND but INCOMPLETE — a failed certificate can be prover looseness, not
a real property. Before trusting it on a transformer Z3 can't touch, we must show
the pipeline is wired correctly and never over-certifies. The one subject where
BOTH tools apply is the toy ReLU MLP: Z3 gives the *exact* certified radius, and
auto_LiRPA handles ReLU + box perturbation natively and tightly. (There is no
softmax transformer to validate against, by construction — softmax is why the
exact threshold-gate transformer uses gate attention.)

THE GATE (what M0 must show)
----------------------------
For the same edited model, same region, same claim:

    auto_LiRPA certified radius  <=  Z3 exact certified radius        (+ a tiny tol)

i.e. the bound method must never certify a *larger* wiggle room than the exact
truth (that would be an unsound bug). Ideally the two are close (CROWN is near-tight
on small ReLU nets), which also says the method is useful, not just sound.

Report: results/boundprop_validate_report.{md,json}.
"""

from __future__ import annotations
import json
import os

import numpy as np
import torch
import torch.nn as nn

from tiny_model import TinyMLP, train, find_skill_circuit, accuracy, \
    true_label_A, true_label_B
from verify import certified_radius as z3_certified_radius

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

REGION_A_HIGH = (0.6, 1.0, 0.0, 1.0)
REGION_B_HIGH = (0.0, 1.0, 0.6, 1.0)
REGION_B_LOW = (0.0, 1.0, 0.0, 0.4)
EPS_MAX = 0.5
TOL = 1e-3


class EditedMLP(nn.Module):
    """The toy model's forward pass as a torch module, with skill A's circuit
    ABLATED (the ablated neurons contribute 0 — implemented by zeroing their
    read-out columns W2[:, j], exactly matching verify.build_logits(ablate=...)).
    Weights are copied from the trained NumPy TinyMLP; this net is only ever
    evaluated, never trained."""

    def __init__(self, model: TinyMLP, ablate: list[int]):
        super().__init__()
        H, d = model.W1.shape
        self.lin1 = nn.Linear(d, H)
        self.lin2 = nn.Linear(H, 2)
        self.double()                                  # params -> float64 BEFORE copy
        W2 = model.W2.copy()
        W2[:, list(ablate)] = 0.0                      # the edit, as a weight change
        with torch.no_grad():
            self.lin1.weight.copy_(torch.tensor(model.W1, dtype=torch.float64))
            self.lin1.bias.copy_(torch.tensor(model.b1, dtype=torch.float64))
            self.lin2.weight.copy_(torch.tensor(W2, dtype=torch.float64))
            self.lin2.bias.copy_(torch.tensor(model.b2, dtype=torch.float64))

    def forward(self, x):
        return self.lin2(torch.relu(self.lin1(x)))


def _inflate(box, eps):
    lo0, hi0, lo1, hi1 = box
    return (max(0.0, lo0 - eps), min(1.0, hi0 + eps),
            max(0.0, lo1 - eps), min(1.0, hi1 + eps))


def alirpa_certifies(bounded, box, eps, out_idx, want) -> bool:
    """Does auto_LiRPA prove the claim over the (inflated) box? want='nonpositive'
    → UB(output[out_idx]) <= 0 everywhere; want='positive' → LB > 0 everywhere."""
    lo0, hi0, lo1, hi1 = _inflate(box, eps)
    x_L = torch.tensor([[lo0, lo1]], dtype=torch.float64)
    x_U = torch.tensor([[hi0, hi1]], dtype=torch.float64)
    center = (x_L + x_U) / 2.0
    ptb = PerturbationLpNorm(norm=np.inf, x_L=x_L, x_U=x_U)
    bt = BoundedTensor(center, ptb)
    lb, ub = bounded.compute_bounds(x=(bt,), method="CROWN")
    lb, ub = lb.detach().numpy()[0], ub.detach().numpy()[0]
    if want == "nonpositive":
        return bool(ub[out_idx] <= 0.0)
    return bool(lb[out_idx] > 0.0)


def alirpa_radius(net, box, out_idx, want) -> float | None:
    """Largest eps for which auto_LiRPA still certifies, by bisection — the
    bound-prop analogue of verify.certified_radius. Returns None if even eps=0
    fails; EPS_MAX if it saturates."""
    bounded = BoundedModule(net, torch.zeros(1, 2, dtype=torch.float64))
    if not alirpa_certifies(bounded, box, 0.0, out_idx, want):
        return None
    if alirpa_certifies(bounded, box, EPS_MAX, out_idx, want):
        return EPS_MAX
    lo, hi = 0.0, EPS_MAX
    while hi - lo > TOL:
        mid = (lo + hi) / 2.0
        if alirpa_certifies(bounded, box, mid, out_idx, want):
            lo = mid
        else:
            hi = mid
    return lo


def _fmt(r):
    if r is None:
        return "refuted at eps=0"
    return f">= {EPS_MAX} (cap)" if r >= EPS_MAX - TOL else f"{r:.3f}"


def main():
    torch.set_grad_enabled(False)
    print("=" * 70)
    print("M0 — bound propagation (auto_LiRPA) vs exact Z3, on the toy ReLU MLP")
    print("=" * 70)

    print("\n[1] Train the two-skill MLP and find + ablate skill A's circuit...")
    model = train(TinyMLP(H=16, seed=0), verbose=False)
    Xt = np.random.default_rng(0).uniform(0, 1, size=(4000, 2))
    Yt = np.stack([true_label_A(Xt), true_label_B(Xt)], axis=1)
    acc = accuracy(model, Xt, Yt)
    circuit_A = find_skill_circuit(model, "A")
    print(f"    accuracy A {acc[0]:.4f} B {acc[1]:.4f}; skill-A circuit "
          f"{circuit_A} (ablated)")
    net = EditedMLP(model, circuit_A)

    # sanity: torch edited forward must match the numpy edited forward
    probe = np.random.default_rng(1).uniform(0, 1, size=(500, 2))
    np_logits = model.forward(probe, ablate=circuit_A)
    tt_logits = net(torch.tensor(probe, dtype=torch.float64)).numpy()
    fwd_gap = float(np.abs(np_logits - tt_logits).max())
    print(f"    torch-vs-numpy edited forward gap: {fwd_gap:.2e}")
    assert fwd_gap < 1e-9, "torch net does not match the numpy edited model"

    claims = [
        ("removal (A nonpositive over A-region)", REGION_A_HIGH, 0, "nonpositive",
         "A", "nonpositive"),
        ("preservation (B positive over B-high)", REGION_B_HIGH, 1, "positive",
         "B", "positive"),
        ("preservation (B nonpositive over B-low)", REGION_B_LOW, 1, "nonpositive",
         "B", "nonpositive"),
    ]

    print("\n[2] Certified radii — auto_LiRPA vs Z3, per claim...")
    rows = []
    all_sound = True
    for name, box, idx, want, z3sel, z3want in claims:
        a_r = alirpa_radius(net, box, idx, want)
        z_r = z3_certified_radius(model, box, z3sel, z3want, ablate=circuit_A,
                                  eps_max=EPS_MAX, tol=TOL)["radius"]
        # soundness: auto_LiRPA must NOT certify a larger radius than exact Z3
        a_val = -1.0 if a_r is None else a_r
        z_val = -1.0 if z_r is None else z_r
        sound = a_val <= z_val + 5 * TOL
        all_sound = all_sound and sound
        rows.append({"claim": name, "alirpa": a_r, "z3": z_r, "sound": sound})
        print(f"    {name}: auto_LiRPA {_fmt(a_r)} | Z3 {_fmt(z_r)} | "
              f"{'SOUND' if sound else 'UNSOUND(!)'}")

    verdict = ("PASS — auto_LiRPA under-approximates the exact Z3 radius on every "
               "claim; the pipeline is sound and ready for M1."
               if all_sound else
               "FAIL — auto_LiRPA certified a LARGER radius than exact Z3 on some "
               "claim; the wiring is unsound, do NOT proceed to M1.")
    print(f"\n[3] Gate: {verdict}")

    _write_report(acc, circuit_A, fwd_gap, rows, all_sound, verdict)
    print("\nDone.")


def _write_report(acc, circuit_A, fwd_gap, rows, all_sound, verdict):
    md = os.path.join(RESULTS, "boundprop_validate_report.md")
    js = os.path.join(RESULTS, "boundprop_validate_report.json")
    lines = [
        "# M0: bound propagation (auto_LiRPA) validated against exact Z3",
        "",
        "Output of `run_boundprop_validate.py` (check M0). Before "
        "trusting the sound-but-incomplete bound-propagation path on a transformer "
        "Z3 cannot reach, we check it on the one subject where BOTH tools apply — "
        "the toy ReLU MLP. The gate: auto_LiRPA's certified radius must "
        "**under-approximate** the exact Z3 radius on every claim (never certify a "
        "larger wiggle room — that would be unsound).",
        "",
        f"Toy model accuracy: skill A {acc[0]:.4f}, skill B {acc[1]:.4f}. Skill-A "
        f"circuit (ablated): {circuit_A}. Torch-vs-numpy edited-forward gap: "
        f"{fwd_gap:.2e} (the two agree, so both tools bound the same function).",
        "",
        "| claim | auto_LiRPA radius | Z3 exact radius | sound? |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['claim']} | {_fmt(r['alirpa'])} | {_fmt(r['z3'])} | "
                     f"{'yes' if r['sound'] else '**NO**'} |")
    lines += [
        "",
        f"**Gate verdict:** {verdict}",
        "",
        "Reading it: auto_LiRPA (CROWN, backward mode) computes a sound linear "
        "relaxation of the edited network over each box; its certified radius is a "
        "lower bound on the true one, so it should sit at or below Z3's exact "
        "radius. Equality/closeness also means the relaxation is tight enough to "
        "be *useful*, not merely sound. With M0 passing, M1 runs the same pipeline "
        "on a softmax+LayerNorm transformer past the exact-Z3 frontier, where Z3 "
        "cannot follow and a PGD attack brackets the certified radius from above.",
    ]
    with open(md, "w") as f:
        f.write("\n".join(lines))
    with open(js, "w") as f:
        json.dump({"accuracy": {"A": acc[0], "B": acc[1]},
                   "circuit_A": circuit_A, "forward_gap": fwd_gap,
                   "rows": rows, "all_sound": all_sound}, f, indent=2,
                  default=float)
    print(f"\nReports written: {md} and .json")


if __name__ == "__main__":
    main()
