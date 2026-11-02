# Group 4 — MGT 4110 A5 Synergy Board

Shared repo for A5, A6 and A7. Members: Amoah, Gopal, Smith.

**Selected asset:** Victor Amoah's daily productivity data (Y = deep-work minutes, interaction = Focus Blocks x Interruptions, n = 50 real days).

## Files
| File | What it does |
|---|---|
| `synergy_distances.py` | Section 2: Jaccard (tooling traits) and Gower (rows, A2 Δ, domain) distance matrices |
| `synthesize_productivity.py` | Section 4: generates the 1,000-row synthetic dataset from Victor's real parameters, then refits the A3 model to check β₃ comes back |
| `data/` | Put Victor's real 50-row file here as `victor_productivity_REAL_n50.csv`. Synthetic output goes here too |

Run with Python 3 + numpy + pandas: `python synergy_distances.py`, `python synthesize_productivity.py`.

## Disclosure
Any file with `SYNTHETIC` in its name is **synthetic data**, generated from an explicit data-generating process grounded in Victor Amoah's real A3 parameters (n = 50). Every synthetic row has `is_synthetic = 1`. It is not real tracked data and must be labeled as synthetic wherever it's used in A6 and A7.
