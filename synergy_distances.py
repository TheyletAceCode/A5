"""A5 Section 2 - team distance matrices (Jaccard + Gower).
Fill in the TRAITS and PROFILES dicts with the team's real values, then run:
    python synergy_distances.py
Needs numpy and pandas.
"""
import numpy as np
import pandas as pd

# ---- Layer 1: binary tooling traits (1 = yes, 0 = no) ----
# TODO: replace None with each member's real answers
# Source: Chayse, 2026-11-02 ("I have python experience and Victor does. She has stat and python experience.")
# External context = the tracker logged conditions outside the person (class schedule, transit) rather than
# only their own behavior. Gopal = 1 because class day and MARTA use are outside-schedule flags. CONFIRM.
TRAIT_NAMES = ["Python", "SQL", "Prior stats course", "Tracked external context"]
TRAITS = {
    "Smith":  [1, 0, 0, 0],   # spending: own purchases, hours worked, payday
    "Amoah":  [1, 0, 0, 0],   # productivity: own focus blocks, interruptions, sleep, caffeine, emails
    "Gopal":  [1, 0, 1, 1],   # walking: steps, movement hour, class day, MARTA use
}

# ---- Layer 2: mixed profiles ----
# rows and domain are from the Selection Matrix.
# a2_delta = |linear CV R^2 - random forest CV R^2| from each member's A2 (pitch slides/videos):
#   Smith 0.072 vs 0.090, Amoah 0.881 vs 0.894, Gopal 0.7776 vs 0.7390
PROFILES = pd.DataFrame({
    "member":   ["Smith", "Amoah", "Gopal"],
    "rows":     [40, 50, 30],
    "a2_delta": [abs(0.072 - 0.090), abs(0.881 - 0.894), abs(0.7776 - 0.7390)],
    "domain":   ["Personal finance", "Productivity", "Health/wellness"],
}).set_index("member")


def jaccard_matrix(traits):
    names = list(traits)
    X = np.array([traits[n] for n in names], dtype=bool)
    n = len(names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            union = np.sum(X[i] | X[j])
            shared = np.sum(X[i] & X[j])   # (0,0) matches are ignored
            D[i, j] = 1 - shared / union if union else 0.0
    return pd.DataFrame(D, index=names, columns=names)


def gower_matrix(df, numeric, nominal):
    names = df.index.tolist()
    n = len(names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            parts = []
            for c in numeric:
                rng = df[c].max() - df[c].min()
                parts.append(abs(df[c].iloc[i] - df[c].iloc[j]) / rng if rng else 0.0)
            for c in nominal:
                parts.append(0.0 if df[c].iloc[i] == df[c].iloc[j] else 1.0)
            D[i, j] = np.mean(parts)
    return pd.DataFrame(D, index=names, columns=names)


if __name__ == "__main__":
    pd.set_option("display.precision", 4)
    if any(v is None for row in TRAITS.values() for v in row):
        print("Fill in TRAITS first.")
    else:
        J = jaccard_matrix(TRAITS)
        print("Jaccard distance matrix\n", J.round(4), "\n")
    if PROFILES["a2_delta"].isna().any():
        print("Fill in a2_delta first.")
    else:
        G = gower_matrix(PROFILES, ["rows", "a2_delta"], ["domain"])
        print("Gower distance matrix\n", G.round(4))
        pairs = [(a, b, G.loc[a, b]) for i, a in enumerate(G.index) for b in G.index[i + 1:]]
        a, b, d = min(pairs, key=lambda p: p[2])
        print(f"\nMost similar pair (Gower): {a} & {b} = {d:.4f}")
