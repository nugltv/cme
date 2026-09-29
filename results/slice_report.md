# Certified mechanistic edits on the base toy model

This is the output of `run_slice.py`. It shows, on the smallest
possible model, that we can *prove* (not just test) what an edit does.

## The model

- A tiny network with two skills. Trained accuracy: **A = 0.9995**, **B = 0.9999**.
- Skill A's circuit (found by knock-out search): **neuron(s) [3]**.
- Skill B's circuit (for contrast): neuron(s) [1].
- After switching off skill A's circuit, numeric accuracy became A = 0.5023 (skill A gone), B = 0.9998 (skill B kept).

## The certificates (each is a proof over an entire continuous region)

| Certificate | Proved? | Solver result | Grid cross-check |
|---|---|---|---|
| Skill A now says LOW everywhere it should say HIGH (x0>=0.6) | ✅ | proved (no counterexample exists) | 0 violations / 40401 pts |
| Skill B still says HIGH everywhere it should (x1>=0.6) | ✅ | proved (no counterexample exists) | 0 violations / 40401 pts |
| Skill B still says LOW everywhere it should (x1<=0.4) | ✅ | proved (no counterexample exists) | 0 violations / 40401 pts |
| ORIGINAL (unedited) model says HIGH everywhere it should (x0>=0.6) | ✅ | proved (no counterexample exists) | 0 violations / 40401 pts |

## What this means, in one sentence

We **proved**, for every input in the relevant regions (not just tested samples), that switching off skill A's circuit destroys skill A everywhere it should have worked, while leaving skill B fully intact — a *certified* mechanistic edit.
