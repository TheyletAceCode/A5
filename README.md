# A5 Group 4

Selected asset: Victor's daily productivity project. Target: daily deep-work
minutes. Interaction: focus blocks x interruptions. Target synthetic N: 1,000.

## Reproduce the analysis

Use Python 3 with numpy, pandas, scipy, and openpyxl:

```sh
python -m pip install -r requirements.txt
python a5_synthesis.py /path/to/productivity_focus_rows_3-27_74-98.xlsx --output-dir generated
```

Keep the original workbook local. It is not included in this public repository.
The script checks the inputs, refits the interaction model, generates a clearly
labeled synthetic CSV, computes the team distances, and exports validation.
Use the same workbook and seed 42 to reproduce the submitted outputs.

## Grounded synthesis

The source has 50 rows and five predictors. The fitted model uses all five
predictors plus focus_blocks x interruptions. Interaction coefficient:
-1.877218; adjusted R-squared: 0.926212; interaction p-value: 0.000237844.

The DGP uses balanced resampling of observed predictor profiles and small
correlated perturbations (20% of the observed feature SDs), restricted to
observed ranges. Counts are rounded to integers and sleep to 0.1 hours.
Targets follow the fitted model plus Gaussian noise with SD 19.83424 minutes.
Negative target draws are set to zero and this adjustment is reported.

Every output row is labeled SYNTHETIC. These are simulated observations,
not 1,000 new days or independent confirmation of a productivity effect.
Synthetic record counts do not justify smaller real-world confidence intervals.
New external-context fields are not invented or treated as measured.

## Files

- `a5_synthesis.py`: full executable data-generating process and distance math.
- `generated/victor_productivity_SYNTHETIC_1000.csv`: 1,000 labeled synthetic rows.
- `generated/source_parameters.json`: source means, SDs, correlations, coefficients.
- `generated/moment_comparison.csv`: source and synthetic means and sample SDs.
- `generated/source_correlations.csv` and `synthetic_correlations.csv`: correlation audit.
- `generated/validation.json`: checks, tolerances, seed, clipping, and fitted interaction.
- `generated/jaccard_distances.csv`: complete binary tooling distance matrix.
- `generated/gower_available_features.csv`: provisional Gower matrix using row volume and domain.

## Remaining team inputs

The full requested Gower calculation needs each member's A2 model divergence
(RF CV R-squared minus linear CV R-squared). These scores were not included in
the supplied group draft. The current available-data Gower calculation uses
equal weights for row volume and nominal domain. It does not fill missing
model scores with zero. Add the verified deltas and recompute before describing
the matrix as the full three-feature Gower analysis.

Each member must confirm their actual contribution. Add the shareable AI
conversation link to the final assignment; this repository is not a substitute
for that transcript link. Section 3 is outside this revision's scope.
