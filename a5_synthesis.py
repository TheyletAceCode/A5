"""Group 4 A5: auditable synthetic data grounded in Victor's supplied sample.

Run: python a5_synthesis.py /path/to/productivity.xlsx --output-dir generated
The original workbook stays local. Output rows are SYNTHETIC, not new observations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import t

FEATURES = ['focus_blocks', 'interruptions', 'sleep_hours', 'caffeine_mg', 'emails_received']
TARGET = 'deep_work_minutes'


def design(x):
    return np.column_stack([np.ones(len(x)), x, x[:, 0] * x[:, 1]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('generated'))
    parser.add_argument('--rows', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 500 <= args.rows <= 1000:
        parser.error('Use 500 to 1000 synthetic rows.')
    source = pd.read_excel(args.workbook)
    data = source[FEATURES + [TARGET]].apply(pd.to_numeric, errors='raise')
    if data.isna().any().any():
        raise ValueError('Missing values must be reviewed before fitting.')
    x, y = data[FEATURES].to_numpy(float), data[TARGET].to_numpy(float)
    a = design(x)
    if np.linalg.matrix_rank(a) != a.shape[1]:
        raise ValueError('The interaction model is rank deficient.')
    beta = np.linalg.lstsq(a, y, rcond=None)[0]
    residual = y - a @ beta
    degrees = len(y) - a.shape[1]
    sigma = np.sqrt(residual @ residual / degrees)
    se = np.sqrt(np.diag(sigma**2 * np.linalg.inv(a.T @ a)))
    pvalues = 2 * t.sf(np.abs(beta / se), degrees)
    adj_r2 = 1 - (residual @ residual / np.sum((y-y.mean())**2)) * (len(y)-1) / degrees

    # Balanced resampling retains representation of all supplied profiles.
    # Small correlated perturbations create nearby profiles without inventing
    # a new population. Jitter SD = 20% of each observed feature's SD.
    rng = np.random.default_rng(args.seed)
    indices = np.tile(np.arange(len(x)), int(np.ceil(args.rows / len(x))))
    rng.shuffle(indices)
    indices = indices[:args.rows]
    # For N divisible by 50, each of the 50 profiles has equal representation.
    # np.repeat fixes the exact deterministic ordering for the submitted run.
    if args.rows % len(x) == 0:
        rng = np.random.default_rng(args.seed)
        indices = np.repeat(np.arange(len(x)), args.rows // len(x))
        rng.shuffle(indices)
    simulated_x = x[indices] + rng.multivariate_normal(
        np.zeros(len(FEATURES)), np.cov(x, rowvar=False) * 0.2**2, size=args.rows
    )
    for j, feature in enumerate(FEATURES):
        simulated_x[:, j] = np.clip(simulated_x[:, j], x[:, j].min(), x[:, j].max())
        simulated_x[:, j] = np.round(simulated_x[:, j], 1 if feature == 'sleep_hours' else 0)
    raw_y = design(simulated_x) @ beta + rng.normal(0, sigma, args.rows)
    simulated_y = np.maximum(0, raw_y)
    synthetic = pd.DataFrame(simulated_x, columns=FEATURES)
    for feature in FEATURES:
        if feature != 'sleep_hours':
            synthetic[feature] = synthetic[feature].astype(int)
    synthetic[TARGET] = simulated_y
    synthetic.insert(0, 'synthetic_id', [f'SYN-{i+1:04d}' for i in range(args.rows)])
    synthetic['data_status'] = 'SYNTHETIC'
    synthetic['source_real_n'] = len(data)
    synthetic['dgp_seed'] = args.seed
    synthetic['source_domain'] = 'Victors_productivity_tracking'

    args.output_dir.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(args.output_dir / 'victor_productivity_SYNTHETIC_1000.csv', index=False)
    comparison = pd.DataFrame({
        'source_mean': data.mean(), 'source_sd': data.std(ddof=1),
        'synthetic_mean': synthetic[FEATURES + [TARGET]].mean(),
        'synthetic_sd': synthetic[FEATURES + [TARGET]].std(ddof=1),
    })
    comparison.to_csv(args.output_dir / 'moment_comparison.csv', index_label='variable')
    source_corr = data.corr()
    synthetic_corr = synthetic[FEATURES + [TARGET]].corr()
    source_corr.to_csv(args.output_dir / 'source_correlations.csv')
    synthetic_corr.to_csv(args.output_dir / 'synthetic_correlations.csv')
    synthetic_beta = np.linalg.lstsq(design(simulated_x), simulated_y, rcond=None)[0]
    audit = {
        'disclosure': 'SYNTHETIC rows grounded in a supplied 50-row sample; not independent real evidence.',
        'source_rows': len(data), 'synthetic_rows': len(synthetic), 'seed': args.seed,
        'source_beta_interaction': float(beta[-1]),
        'synthetic_refit_beta_interaction': float(synthetic_beta[-1]),
        'source_interaction_p_value': float(pvalues[-1]),
        'source_adjusted_r_squared': float(adj_r2),
        'source_residual_sd': float(sigma),
        'target_values_floored_at_zero': int(np.sum(raw_y < 0)),
        'unique_synthetic_predictor_profiles': int(len(synthetic[FEATURES].drop_duplicates())),
        'max_absolute_correlation_difference': float(np.max(np.abs(source_corr.values-synthetic_corr.values))),
        'target_mean_relative_difference': float(abs(simulated_y.mean()-y.mean()) / y.mean()),
        'target_sd_relative_difference': float(abs(simulated_y.std(ddof=1)-y.std(ddof=1)) / y.std(ddof=1)),
        'interaction_relative_difference': float(abs((synthetic_beta[-1]-beta[-1])/beta[-1])),
        'input_jitter_fraction_of_observed_sd': 0.2,
        'predictor_bounds': 'Observed minimum and maximum; counts rounded to integers; sleep to 0.1 hours.',
        'validation_limits': {'mean_relative': 0.10, 'sd_relative': 0.15, 'correlation_absolute': 0.15, 'interaction_relative': 0.20},
    }
    audit['checks_passed'] = bool(
        audit['target_mean_relative_difference'] <= .10 and audit['target_sd_relative_difference'] <= .15
        and audit['max_absolute_correlation_difference'] <= .15 and audit['interaction_relative_difference'] <= .20
        and not synthetic.isna().any().any() and (simulated_y >= 0).all()
    )
    params = {
        'coefficient_order': ['intercept'] + FEATURES + ['focus_blocks_x_interruptions'],
        'coefficients': beta.tolist(), 'means': data.mean().to_dict(),
        'sample_standard_deviations': data.std(ddof=1).to_dict(),
        'correlations': source_corr.to_dict(), 'residual_sd': float(sigma),
        'source_file': args.workbook.name,
    }
    (args.output_dir / 'source_parameters.json').write_text(json.dumps(params, indent=2))
    (args.output_dir / 'validation.json').write_text(json.dumps(audit, indent=2))

    members = ['Chayse', 'Victor', 'Savitha']
    tooling = np.array([[1,0,1,0], [1,0,1,0], [1,0,1,1]], dtype=bool)
    jaccard = pd.DataFrame(squareform(pdist(tooling, 'jaccard')), index=members, columns=members)
    jaccard.to_csv(args.output_dir / 'jaccard_distances.csv')
    # Available-data Gower distance: equal weights for observed volume and
    # nominal project domain. A2 model divergence is missing, not zero.
    # Supply all three A2 deltas before describing this as the full requested
    # three-feature Gower calculation.
    rows = np.array([40.,50.,30.])
    domain = np.array(['spending','productivity','walking'])
    gower = (np.abs(rows[:,None]-rows[None,:])/np.ptp(rows) + (domain[:,None]!=domain[None,:]))/2
    pd.DataFrame(gower, index=members, columns=members).to_csv(args.output_dir / 'gower_available_features.csv')
    print(json.dumps(audit, indent=2))
    if not audit['checks_passed']:
        raise RuntimeError('Review validation results before using this synthetic dataset.')


if __name__ == '__main__':
    main()
