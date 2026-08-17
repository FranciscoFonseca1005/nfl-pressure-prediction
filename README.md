# NFL Pass-Rush Pressure Prediction at the Snap

[English](README.md) | [Español](README.es.md)

A leakage-aware Machine Learning project that estimates, at the moment of the snap, the probability that an NFL pass rusher will subsequently generate pressure on the quarterback.

The project was developed as a professional Data Scientist technical assessment. It emphasizes temporal validation, reproducibility, honest model selection, interpretable results, and strict separation between model development and final testing.

For the complete methodology and evidence, read the [English technical report](reports/technical_report.md).

## Executive summary

The analytical dataset contains 36,259 pass-rusher observations from 8,531 plays in NFL weeks 1–8. A pressure occurred in 4,214 observations, producing an overall prevalence of 11.621942%.

Each row represents one pass rusher in one play, uniquely identified by:

```text
gameId + playId + nflId
```

The binary target is:

```text
pressure = pff_hurry OR pff_hit OR pff_sack
```

Only information available at the snap was permitted as input. Post-snap movement, play outcomes, blocking results, and PFF outcome fields were excluded from the predictors.

A regularized, unweighted `LogisticRegression` pipeline was selected. Although Extra Trees achieved a slightly higher validation Average precision, its improvement was only 0.001790 and its 95% bootstrap interval, `[-0.011327, 0.014929]`, included zero. The logistic model was retained because its performance was statistically equivalent while offering a simpler and more interpretable solution.

On the untouched week-8 test set, the final model achieved:

- Average precision: 0.15775008696279033
- ROC-AUC: 0.5968898557731045
- F1: 0.23833167825223436
- Recall: 0.47619047619047616
- Precision: 0.15894039735099338

The model provides useful ranking and prioritization, but its absolute discrimination is modest and it produces many false positives. It should not be presented as an autonomous, high-confidence pressure detector.

## Football concepts

- **Quarterback:** the offensive player who normally receives the ball and attempts the pass.
- **Pass rusher:** a defender whose assignment is to reach or disrupt the quarterback.
- **Snap:** the instant when the play begins and the ball is delivered to the quarterback.
- **Hurry:** pressure that forces the quarterback to act earlier or less comfortably.
- **Hit:** contact with the quarterback associated with the pass rush.
- **Sack:** the quarterback is tackled behind the line of scrimmage before completing a pass.
- **Pressure:** for this project, any PFF hurry, hit, or sack.

The prediction unit is a pass rusher within a play. A play is uniquely counted with `gameId + playId`; `playId` alone is not globally unique.

## Data

The project uses public NFL player-tracking, PFF scouting, and team-information data from [Kaggle](https://www.kaggle.com/datasets/dmay01/usefuldata).

Key validated quantities:

| Item | Value |
|---|---:|
| Raw tracking rows | 8,314,178 |
| Raw tracking columns | 16 |
| PFF player-play records | 188,254 |
| Source plays | 8,557 |
| Plays with a valid manual snap | 8,532 |
| Final analytical rows | 36,259 |
| Final analytical plays | 8,531 |
| Positive pressures | 4,214 |
| Overall pressure rate | 11.621942% |
| Approved predictors | 51 |
| Missing or non-finite predictor values | 0 |

The physical tracking files did not correspond one-to-one with actual NFL weeks. Therefore, `actual_week` was derived from each game's date rather than inferred from the Parquet filename.

## Analytical design

The model uses 51 approved predictors:

- 50 numeric predictors.
- One categorical predictor: `rusher_position_lined_up`.

The features describe the pre-play geometry and movement available at the snap, including rusher-to-quarterback distance, relative gaps, quarterback depth, nearest-blocker relationships, normalized field coordinates, speeds, and directional components.

Numeric features are standardized. The categorical position variable is processed with one-hot encoding and `handle_unknown="ignore"`.

Identifiers, target components, post-snap information, outcome variables, blocking-result fields, audit-only fields, and variables that could leak future information are excluded.

## Temporal evaluation design

A temporal split was used instead of a random split so that evaluation better represents deployment on future games.

| Purpose | Weeks | Rows | Plays | Positives | Pressure rate |
|---|---:|---:|---:|---:|---:|
| Initial training | 1–6 | 27,950 | 6,587 | 3,283 | 11.745975% |
| Validation and threshold selection | 7 | 3,865 | 913 | 427 | 11.047865% |
| Final untouched test | 8 | 4,444 | 1,031 | 504 | 11.341134% |
| Complete dataset | 1–8 | 36,259 | 8,531 | 4,214 | 11.621942% |

Week 8 remained sealed during data preparation, feature selection, model comparison, hyperparameter evaluation, and threshold selection. It was opened only once for final inference after the complete policy had been locked.

## Methodology

1. Validate raw file inventory, schemas, keys, snap events, and weekly coverage.
2. Construct the pressure target from PFF hurry, hit, and sack labels.
3. Build one analytical row for every eligible pass rusher at the snap.
4. remove identifiers, future information, outcomes, and leakage-prone variables.
5. Perform exploratory analysis without accessing week 8.
6. Compare baseline logistic models on validation week 7.
7. Evaluate Extra Trees and HistGradientBoosting with temporal cross-validation.
8. Select the model using performance, uncertainty, simplicity, and interpretability.
9. Select the operational threshold exclusively on validation week 7.
10. Lock the model, feature order, and threshold with SHA-256 evidence.
11. Fit the selected pipeline once on weeks 1–7.
12. Evaluate once on untouched week 8 without any post-test adjustment.

## Model selection

### Baselines

| Model | Validation Average precision | ROC-AUC | Log loss |
|---|---:|---:|---:|
| Dummy prior | 0.110479 | 0.500000 | 0.347754 |
| `LogisticRegression_unweighted` | 0.151899 | 0.600006 | 0.342382 |
| `LogisticRegression_balanced` | 0.151749 | 0.601697 | 0.675077 |

### Advanced candidates

| Model | Validation Average precision | ROC-AUC | Log loss | Temporal-CV mean AP |
|---|---:|---:|---:|---:|
| Extra Trees | 0.153689 | 0.602314 | 0.341647 | 0.155191 |
| Logistic regression | 0.151899 | 0.600006 | 0.342382 | 0.154996 |
| HistGradientBoosting | 0.145960 | 0.595577 | 0.343728 | 0.150105 |

The selected model is `LogisticRegression_unweighted` with:

```text
C=1.0
solver=lbfgs
max_iter=2000
random_state=42
class_weight=None
```

The small Extra Trees advantage was not stable under bootstrap resampling. The logistic pipeline therefore offered the best balance of performance, transparency, and operational simplicity.

![Validation Average precision by model](reports/report_model_average_precision.png)

## Locked decision threshold

The probability threshold was selected by maximizing F1 on validation week 7. Ties would be resolved by choosing the higher threshold.

```text
Locked threshold = 0.12485514806532311
```

Validation performance at this threshold:

| Metric | Value |
|---|---:|
| TN | 2,259 |
| FP | 1,179 |
| FN | 211 |
| TP | 216 |
| Precision | 0.154839 |
| Recall | 0.505855 |
| F1 | 0.237102 |
| Specificity | 0.657068 |
| Predicted-positive rate | 0.360931 |

A conventional threshold of 0.5 produced no positive predictions because the maximum validation probability was approximately 0.4499. The selected threshold reflects the model's probability distribution and the chosen F1 objective; it is not a universal football rule.

## Final test results

After the model and threshold were locked, the pipeline was fitted exactly once on weeks 1–7, using 31,815 rows and 3,710 positive observations. No week-8 row was used during fitting.

| Metric | Untouched week-8 result |
|---|---:|
| Average precision | 0.15775008696279033 |
| ROC-AUC | 0.5968898557731045 |
| Log loss | 0.34781248418273736 |
| Precision | 0.15894039735099338 |
| Recall | 0.47619047619047616 |
| F1 | 0.23833167825223436 |
| Specificity | 0.6776649746192893 |
| Predicted-positive rate | 0.3397839783978398 |
| Actual-positive rate | 0.11341134113411341 |

Final confusion matrix:

| Actual / predicted | No pressure | Pressure |
|---|---:|---:|
| No pressure | TN = 2,670 | FP = 1,270 |
| Pressure | FN = 264 | TP = 240 |

![Final week-8 confusion matrix](reports/report_test_confusion_matrix.png)

The final F1 was slightly higher than validation F1, while Average precision and ROC-AUC remained in a similar range. This supports stable temporal generalization, although the overall discrimination remains modest.

## Ranking and calibration

Additional week-8 evidence:

| Diagnostic | Value |
|---|---:|
| Mean predicted probability | 0.115319 |
| Actual pressure rate | 0.11341134113411341 |
| Brier score | 0.099395 |
| Top-decile pressure rate | 19.101% |
| Top-decile lift | 1.684234 |
| AP lift over prevalence | 1.391× |

The highest-scored 10% of pass-rusher observations contained pressure at a substantially higher rate than the overall population. This is the strongest practical use of the model: ranking observations for analyst review, film study, or prioritization.

![Pressure rate by prediction decile](reports/stage9_decile_stability.png)

## Exploratory stability

Pressure prevalence was relatively stable across the eight weeks, supporting the temporal comparison while still preserving a realistic future-period test.

![Weekly pressure rate](reports/report_weekly_pressure_rate.png)

## Interpretation

The final model captures a real but limited signal from snap-time geometry and movement.

Appropriate uses include:

- Ranking pass-rusher observations by estimated pressure likelihood.
- Prioritizing plays for analyst or coaching review.
- Supporting exploratory football analysis.
- Providing an interpretable baseline for future research.

Inappropriate uses include:

- Treating every positive classification as a confirmed future pressure.
- Replacing film review or domain expertise.
- Claiming causal effects from model coefficients.
- Presenting the system as an autonomous high-confidence detector.

The threshold favors recall over precision. It identifies 240 of 504 true pressures but also flags 1,270 non-pressure observations. That tradeoff must be considered in any operational use.

## Repository structure

```text
nfl-pressure-prediction/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_analytical_table.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_modeling_baselines.ipynb
│   ├── 05_advanced_modeling.ipynb
│   ├── 06_threshold_selection.ipynb
│   └── 07_final_evaluation.ipynb
├── reports/
├── src/
│   └── nfl_pressure/
├── README.md
├── README.es.md
└── requirements.txt
```

Large raw and generated data files are intentionally excluded from Git where appropriate.

## Installation

The validated environment uses Python 3.14.0.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name nfl-pressure --display-name "Python (nfl-pressure)"
```

Download the source dataset from Kaggle and place the original files under `data/raw/` without changing their names.

## Reproduction

Run the notebooks in numerical order:

| Notebook | Purpose |
|---|---|
| [01_data_validation.ipynb](notebooks/01_data_validation.ipynb) | Raw-data validation and target definition |
| [02_analytical_table.ipynb](notebooks/02_analytical_table.ipynb) | Snap-level analytical table construction |
| [03_exploratory_analysis.ipynb](notebooks/03_exploratory_analysis.ipynb) | Leakage-safe exploratory analysis |
| [04_modeling_baselines.ipynb](notebooks/04_modeling_baselines.ipynb) | Baseline models and initial selection |
| [05_advanced_modeling.ipynb](notebooks/05_advanced_modeling.ipynb) | Advanced candidates and temporal CV |
| [06_threshold_selection.ipynb](notebooks/06_threshold_selection.ipynb) | Validation-only threshold selection |
| [07_final_evaluation.ipynb](notebooks/07_final_evaluation.ipynb) | Locked final evaluation on week 8 |

The notebooks include executed outputs, quality controls, and artifact checks. The final evaluation notebook completed with 93/93 quality controls, 14/14 PASS markers, and 12/12 artifact-hash checks.

## Documentation

- [English technical report](reports/technical_report.md)
- [Reporte técnico en español](reports/technical_report_es.md)
- [Versión en español de este README](README.es.md)

## Limitations

- The data cover only weeks 1–8 of one NFL season.
- PFF pressure labels are expert annotations and may contain judgment variability.
- Snap-time features cannot represent events that develop later in the play.
- Multiple pass rushers from the same play are not fully independent observations.
- The selected threshold was optimized for F1 and creates many false positives.
- Performance may change for other seasons, teams, schemes, or tracking systems.
- The model is predictive, not causal.
- External temporal validation would be required before production deployment.

## Reproducibility and test integrity

The project preserves a complete evidence trail through saved tables, model artifacts, predictions, manifests, quality-control reports, and SHA-256 hashes.

Week 8 was not used for model selection, feature selection, hyperparameter tuning, calibration, or threshold selection. After the final test was opened, no model, predictor, hyperparameter, threshold, or conclusion was adjusted.

This strict no-post-test-adjustment policy is essential to interpreting the final results honestly.
