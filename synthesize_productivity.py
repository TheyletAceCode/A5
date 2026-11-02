"""A5 Section 4 - DISCLOSED SYNTHETIC scale-up of Victor Amoah's real productivity data.

Group 4, MGT 4110. Real source: Victor's 50 tracked days (target = daily deep-work minutes;
predictors = focus blocks, interruptions, sleep, caffeine, emails).

Same technique as our A4 work: an explicit data-generating process (DGP) with random noise,
written out in code so anyone can audit it. The difference from A4 is that every parameter
here comes from Victor's REAL data and A3 model instead of assumed values.

Run:  python synthesize_productivity.py
Output: group4_productivity_SYNTHETIC_n1000.csv  (every row has is_synthetic = 1)
"""
import numpy as np
import pandas as pd

N = 1000
SEED = 4110
PREDICTORS = ["focus_blocks", "interruptions", "sleep_hours", "caffeine", "emails"]

# ---- Real parameters (fill from Victor's 50-row file / A3 regression output) ----
REAL = {
    "means": None,        # TODO: dict {predictor: mean}
    "sds": None,          # TODO: dict {predictor: sd}
    "corr": None,         # TODO: 5x5 correlation matrix in PREDICTORS order
    "coef": {             # A3 interaction model: Y = b0 + sum(b*X) + b3*(focus_blocks*interruptions)
        "intercept": None,
        "focus_blocks": 36.17,   # DERIVED from Victor's A3 slide, confirm with his output
        "interruptions": None,
        "sleep_hours": None,
        "caffeine": None,
        "emails": None,
        "fb_x_int": -1.61,       # beta3, DERIVED from Victor's A3 slide, confirm with his output
    },
    "resid_sd": None,     # TODO: residual SD of the A3 model
}
# How the two DERIVED values were found: Victor's A3 slide gives the payoff of one more focus
# block at 4, 8 and 13 interruptions (29.7, 23.3, 15.2 minutes). That payoff equals
# b_focus + beta3 * interruptions, so a straight line through those points gives
# beta3 = -1.61 and b_focus = 36.17.


def generate(n, real, seed=SEED):
    rng = np.random.default_rng(seed)
    mu = np.array([real["means"][p] for p in PREDICTORS])
    sd = np.array([real["sds"][p] for p in PREDICTORS])
    cov = np.array(real["corr"]) * np.outer(sd, sd)
    X = pd.DataFrame(rng.multivariate_normal(mu, cov, size=n), columns=PREDICTORS)
    for c in ["focus_blocks", "interruptions", "caffeine", "emails"]:   # counts
        X[c] = X[c].round().clip(lower=0)
    X["sleep_hours"] = X["sleep_hours"].clip(lower=0).round(1)
    b = real["coef"]
    y = b["intercept"] + sum(b[p] * X[p] for p in PREDICTORS) \
        + b["fb_x_int"] * X["focus_blocks"] * X["interruptions"] \
        + rng.normal(0, real["resid_sd"], n)
    X["deep_work_minutes"] = y.clip(lower=0).round(1)
    X["is_synthetic"] = 1
    return X


def check(df, real):
    """Refit the A3 model on the synthetic rows and compare to the real values."""
    cols = PREDICTORS + ["fb_x_int"]
    d = df.assign(fb_x_int=df.focus_blocks * df.interruptions)
    A = np.column_stack([np.ones(len(d))] + [d[c] for c in cols])
    est = np.linalg.lstsq(A, d.deep_work_minutes, rcond=None)[0]
    print(f"{'term':<15}{'real':>10}{'synthetic':>12}")
    for name, e in zip(["intercept"] + cols, est):
        print(f"{name:<15}{real['coef'][name]:>10.3f}{e:>12.3f}")
    print("\nMeans / SDs on synthetic rows:\n", df[PREDICTORS + ['deep_work_minutes']].agg(['mean', 'std']).round(2))


if __name__ == "__main__":
    missing = [k for k, v in REAL.items() if v is None] + \
              [k for k, v in REAL["coef"].items() if v is None]
    if missing:
        raise SystemExit(f"Fill these from Victor's real data first: {missing}")
    df = generate(N, REAL)
    check(df, REAL)
    out = f"group4_productivity_SYNTHETIC_n{N}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}. SYNTHETIC data grounded in Victor Amoah's real A3 parameters (n = 50).")
