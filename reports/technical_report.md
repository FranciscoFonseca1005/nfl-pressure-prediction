# NFL Pass-Rush Pressure Prediction at the Snap

## Technical Report

**Author:** Francisco Miguel Fonseca Martínez  
**Evaluation date:** August 17, 2026  
**Repository:** [nfl-pressure-prediction](https://github.com/FranciscoFonseca1005/nfl-pressure-prediction)  
**Final evaluation commit:** `57b088e41425b18c844e18cd34ba6cb73f8dae7e`

> **Project status:** The final out-of-time evaluation was completed on NFL week 8 without any post-test changes to the model, features, hyperparameters, calibration, or classification threshold.

---

## Executive Summary

This project develops and validates a machine learning solution that estimates, at the moment of the snap, the probability that an individual pass rusher will produce quarterback pressure later in the play.

In football terms, a **pass rusher** is a defensive player attempting to reach or disrupt the quarterback. A **pressure** is defined from the supplied PFF scouting labels as at least one recorded hurry, quarterback hit, or sack. The model makes one prediction per pass-rusher/play combination using only information available when the ball is snapped.

The analysis used NFL tracking and PFF scouting data covering weeks 1 through 8. After data validation and snap-quality exclusions, the final analytical table contained:

- 36,259 pass-rusher observations.
- 8,531 unique plays, identified by `gameId + playId`.
- 4,214 positive pressure outcomes.
- An overall pressure prevalence of 11.622%.
- 51 eligible predictors: 50 numeric tracking features and one categorical alignment feature.

A strict temporal design was used:

| Purpose | NFL weeks | Rows | Unique plays | Positives | Pressure rate |
|---|---:|---:|---:|---:|---:|
| Initial training | 1-6 | 27,950 | 6,587 | 3,283 | 11.746% |
| Validation and threshold selection | 7 | 3,865 | 913 | 427 | 11.048% |
| Final untouched test | 8 | 4,444 | 1,031 | 504 | 11.341% |

An unweighted logistic regression pipeline was retained over more complex Extra Trees and histogram gradient boosting candidates. Extra Trees produced only a marginal validation average-precision gain of 0.001790, and the bootstrap 95% confidence interval for that gain included zero. The additional complexity was therefore not supported by stable evidence.

The classification threshold was selected exclusively on week 7 by maximizing F1 and was permanently locked at `0.12485514806532311`. The selected model was then cloned and fitted once on weeks 1-7 before week 8 was accessed.

Final week-8 performance was:

| Metric | Week-8 result |
|---|---:|
| Average precision | 0.157750 |
| ROC-AUC | 0.596890 |
| Log loss | 0.347812 |
| Precision | 0.158940 |
| Recall | 0.476190 |
| F1 | 0.238332 |
| Specificity | 0.677665 |
| Actual pressure rate | 0.113411 |
| Predicted-positive rate | 0.339784 |

The model's final average precision was 1.391 times the week-8 prevalence. Its highest-scored decile contained a 19.101% observed pressure rate, representing 1.684 times the test prevalence.

The results support **modest but temporally stable ranking ability**. The model can help prioritize pass rushers for review or downstream analysis, but its low precision and large number of false positives make it unsuitable as a high-confidence autonomous pressure detector.

---

## 1. Problem Definition

### 1.1 Analytical question

The project addresses the following question:

> Using only player and ball tracking information available at the snap, can a model rank individual pass rushers by their probability of generating quarterback pressure?

This is a binary probabilistic classification problem:

\[
P(\text{pressure}=1 \mid X_{\text{snap}})
\]

where:

- `pressure = 1` indicates a PFF-recorded hurry, hit, or sack.
- `pressure = 0` indicates that none of those events was recorded.
- \(X_{\text{snap}}\) contains only features available at the snap.

### 1.2 Unit of analysis

The analytical grain is one pass rusher in one play. Its unique key is:

```text
gameId + playId + nflId
```

This distinction matters because a single play can contain multiple pass rushers. Each rusher receives an individual probability and classification. A play itself must be identified by `gameId + playId`; `playId` alone may repeat across games.

### 1.3 Intended and excluded uses

The probability can be used to rank pass rushers, prioritize observations for film or analytical review, support descriptive analysis of snap geometry, or serve as one input to a broader decision-support system.

The model is not intended to predict sacks specifically, judge overall player quality, replace PFF scouting, use post-snap trajectories, make high-confidence autonomous decisions, or establish causal relationships between alignment and pressure.

---

## 2. Data Sources and Scope

The source dataset was obtained from the supplied [Kaggle dataset](https://www.kaggle.com/datasets/dmay01/usefuldata).

| Component | Content | Role in the project |
|---|---|---|
| Player tracking Parquet files | Player and football position, speed, direction, orientation, event, and frame data | Creation of snap-level predictors |
| PFF scouting data | Player role, alignment, and pass-rush outcomes | Definition of eligible rushers and the pressure target |
| Team information | Team identifiers and metadata | Validation and reference support |

The raw tracking data contained 8,314,178 rows and 16 columns across eight Parquet files. The PFF scouting source contained 188,254 player-play records.

The physical tracking filenames did not correspond perfectly to actual NFL weeks. Therefore, `actual_week` was derived from the game date rather than inferred from the Parquet filename. This prevented an incorrect temporal split.

The available scope covers 122 games from weeks 1 through 8. Conclusions should not automatically be generalized to an entire season, another season, postseason games, or a different tracking system.

---

## 3. Data Validation and Target Construction

### 3.1 Source validation

The initial quality process verified that the expected source files were present and non-empty, all tracking files had compatible schemas, every frame contained exactly 23 entities, frame identifiers had no unexpected gaps, PFF player-play keys were complete and unique, and tracking and PFF records matched one-to-one.

PFF indicators were checked for valid binary domains. Structural missingness was preserved rather than globally imputed, and historical team aliases were retained as reference metadata. The validation stage produced documented controls and notes instead of silently discarding unusual observations.

### 3.2 Snap selection

Features were constructed at the manually marked `ball_snap` event, the moment when the center begins the play by delivering the ball to the quarterback.

Of 8,557 source plays, 8,532 had a valid manual snap marker and 25 were excluded. Pass Rush rows declined from 36,362 to 36,259, meaning only 103 rows, approximately 0.283%, were removed. Using one explicit event prevents accidental mixing of pre-snap and post-snap information.

### 3.3 Pressure target

The binary target was constructed as:

```text
pressure = pff_hurry OR pff_hit OR pff_sack
```

| Target property | Result |
|---|---:|
| Total rows | 36,259 |
| Pressure positives | 4,214 |
| Pressure negatives | 32,045 |
| Pressure prevalence | 11.622% |
| Missing target values | 0 |
| Observed classes | 0 and 1 |

The target components were excluded from the model predictors to prevent leakage.

---

## 4. Analytical Table and Feature Engineering

The processed analytical table is stored at `data/processed/pass_rush_analytical_table.parquet`. It contains 36,259 rows and 70 documented columns. Every column was assigned to exactly one role: identifier, temporal metadata, target, predictor, or audit-only field.

### 4.1 Predictor contract

The approved model contract contains 50 numeric predictors and one categorical predictor, `rusher_position_lined_up`, for a total of 51 predictors in a permanently stored order. There were no missing or non-finite predictor values and no constant predictors.

### 4.2 Feature families

| Feature family | Examples | Interpretation |
|---|---|---|
| Rusher-quarterback geometry | Distance, longitudinal gap, lateral offset | Where the rusher is located relative to the quarterback |
| Rusher-ball geometry | Position relative to the ball | Where the defender begins relative to the snap point |
| Quarterback depth | Position behind the ball | Initial quarterback positioning |
| Nearest-blocker geometry | Distance and directional alignment | Relationship between the rusher and closest likely blocker |
| Motion | Speed, velocity components, closing speed | Direction and intensity of movement at the snap |
| Direction encoding | Sine and cosine transformations | Continuous representation of circular angles |
| PFF alignment | `rusher_position_lined_up` | Dataset-specific defensive alignment category |

Numeric variables were standardized inside the model pipeline. The categorical variable was one-hot encoded. Identifiers, audit fields, target components, week labels, and outcomes were excluded.

---

## 5. Exploratory Analysis

### 5.1 Class imbalance and temporal stability

Only 11.622% of observations produced pressure. A classifier could therefore obtain high accuracy by almost always predicting no pressure. Average precision was emphasized because it evaluates how effectively high scores concentrate the rare positive class.

![Weekly pressure prevalence](report_weekly_pressure_rate.png)

| Week | Rows | Unique plays | Positives | Pressure rate |
|---:|---:|---:|---:|---:|
| 1 | 4,988 | 1,172 | 598 | 11.989% |
| 2 | 4,498 | 1,062 | 521 | 11.583% |
| 3 | 4,809 | 1,139 | 606 | 12.601% |
| 4 | 4,733 | 1,108 | 557 | 11.768% |
| 5 | 4,681 | 1,105 | 528 | 11.280% |
| 6 | 4,241 | 1,001 | 473 | 11.153% |
| 7 | 3,865 | 913 | 427 | 11.048% |
| 8 | 4,444 | 1,031 | 504 | 11.341% |

Observed prevalence remained within a narrow range, reducing concern about severe target-rate drift between development and testing.

### 5.2 Snap-level feature differences

| Feature | No-pressure mean | Pressure mean | Difference |
|---|---:|---:|---:|
| Rusher-QB distance | 6.504 | 6.936 | +0.433 |
| Rusher-QB longitudinal gap | 5.076 | 5.278 | +0.203 |
| Nearest-blocker distance | 2.600 | 2.878 | +0.278 |
| Rusher velocity toward QB | 0.229 | 0.299 | +0.070 |
| Rusher-QB closing speed | 0.199 | 0.270 | +0.071 |

Pressure outcomes showed somewhat higher movement toward the quarterback and closing speed at the snap. These are marginal descriptive comparisons, not causal effects.

---

## 6. Experimental Design

Random row splitting would not represent prediction on future weeks. The project instead used weeks 1-6 for initial training, week 7 for model validation and threshold selection, and week 8 for one-time final testing. No `gameId + playId` key appeared in more than one partition.

Probability quality was evaluated with average precision, ROC-AUC, and log loss. Brier score was used descriptively for calibration. After threshold selection, decisions were evaluated with precision, recall, F1, specificity, predicted-positive rate, and confusion-matrix counts.

---

## 7. Baseline and Advanced Modeling

The baseline comparison included a prevalence-based dummy classifier, unweighted logistic regression, and class-balanced logistic regression. Numeric predictors were standardized and the alignment category was one-hot encoded inside each logistic pipeline.

| Model | Average precision | ROC-AUC | Log loss | Selected |
|---|---:|---:|---:|---|
| Dummy prior | 0.110479 | 0.500000 | 0.347754 | No |
| Logistic regression, unweighted | 0.151899 | 0.600006 | 0.342382 | Yes |
| Logistic regression, balanced | 0.151749 | 0.601697 | 0.675077 | No |

The selected logistic model used `C=1.0`, `solver="lbfgs"`, `max_iter=2000`, `random_state=42`, and no class weights. Compared with the dummy model, it improved average precision by 0.041420, or 37.49%.

Extra Trees and histogram gradient boosting were then assessed with locked candidate grids and sequential temporal validation folds on weeks 4-6.

![Model average-precision comparison](report_model_average_precision.png)

| Model | Week-7 AP | ROC-AUC | Log loss | Temporal-CV mean AP |
|---|---:|---:|---:|---:|
| Extra Trees | 0.153689 | 0.602314 | 0.341647 | 0.155191 |
| Logistic regression | 0.151899 | 0.600006 | 0.342382 | 0.154996 |
| Histogram gradient boosting | 0.145960 | 0.595577 | 0.343728 | 0.150105 |

Extra Trees achieved the highest point estimate, but its AP gain over logistic regression was only 0.001790. Its bootstrap 95% interval ranged from -0.011327 to 0.014929 and included zero. Intervals for ROC-AUC gain and log-loss reduction also included zero.

The unweighted logistic regression was retained because it offered nearly identical predictive performance, greater interpretability, simpler deployment, and no statistically supported disadvantage.

---

## 8. Threshold Selection

The validation probabilities ranged from approximately 0.0283 to 0.4499, so a threshold of 0.5 predicted zero positive cases. Every probability-changing threshold on week 7 was evaluated, producing 3,865 candidates.

The policy maximized validation F1 and selected the highest threshold in the event of a tie. The threshold was permanently locked before week 8:

```text
0.12485514806532311
```

| Threshold multiplier | Threshold | Precision | Recall | F1 | Specificity | Predicted-positive rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.099884 | 13.264% | 74.707% | 0.225282 | 39.325% | 62.225% |
| 0.90 | 0.112370 | 13.908% | 61.593% | 0.226920 | 52.647% | 48.926% |
| 0.95 | 0.118612 | 14.686% | 56.440% | 0.233075 | 59.279% | 42.458% |
| **1.00, selected** | **0.124855** | **15.484%** | **50.585%** | **0.237102** | **65.707%** | **36.093%** |
| 1.05 | 0.131098 | 15.509% | 42.857% | 0.227754 | 71.001% | 30.530% |
| 1.10 | 0.137341 | 15.314% | 34.895% | 0.212857 | 76.033% | 25.175% |
| 1.20 | 0.149826 | 14.556% | 23.419% | 0.179533 | 82.926% | 17.775% |

Lower thresholds detect more pressures but create more false alerts. Higher thresholds reduce alerts but miss more real pressures. At the selected threshold, week 7 produced 2,259 true negatives, 1,179 false positives, 211 false negatives, and 216 true positives.

---

## 9. Final Out-of-Time Evaluation

Before week 8 was accessed, the selected model and policy hashes, feature order, hyperparameters, threshold, and training recipe were validated. A pre-test manifest was exported, and any later model, feature, calibration, or threshold adjustment was forbidden.

The final pipeline was created by cloning the selected Stage-7 model and was fitted exactly once on weeks 1-7: 31,815 rows, 3,710 positives, and 51 predictors. No week-8 row was used during fitting.

| Metric | Validation week 7 | Test week 8 | Test minus validation |
|---|---:|---:|---:|
| Average precision | 0.151899 | 0.157750 | +0.005851 |
| ROC-AUC | 0.600006 | 0.596890 | -0.003117 |
| Log loss | 0.342382 | 0.347812 | +0.005430 |
| Precision | 0.154839 | 0.158940 | +0.004102 |
| Recall | 0.505855 | 0.476190 | -0.029664 |
| F1 | 0.237102 | 0.238332 | +0.001230 |
| Specificity | 0.657068 | 0.677665 | +0.020597 |
| Predicted-positive rate | 0.360931 | 0.339784 | -0.021147 |
| Actual-positive rate | 0.110479 | 0.113411 | +0.002933 |

The probability-ranking metrics changed only modestly, supporting temporal stability while also confirming limited absolute discrimination.

### 9.1 Final confusion matrix

![Final test confusion matrix](report_test_confusion_matrix.png)

| Outcome | Count |
|---|---:|
| True negatives | 2,670 |
| False positives | 1,270 |
| False negatives | 264 |
| True positives | 240 |

The model detected 240 of 504 real pressures, corresponding to 47.619% recall. Only 15.894% of positive predictions were correct. It predicted pressure for 33.978% of rusher observations, approximately three times the actual 11.341% rate.

### 9.2 Ranking and calibration

![Validation and test decile stability](stage9_decile_stability.png)

| Dataset | Actual rate | Mean predicted probability | Brier score | Top-decile rate | Top-decile lift |
|---|---:|---:|---:|---:|---:|
| Validation week 7 | 0.110479 | 0.117648 | 0.097418 | 0.170543 | 1.543670 |
| Test week 8 | 0.113411 | 0.115319 | 0.099395 | 0.191011 | 1.684234 |

On week 8, average precision was 1.391 times prevalence and the highest-scored decile had a 19.101% pressure rate. Mean predicted probability, 11.532%, was close to the actual 11.341% rate. The probability ranking is therefore more useful than the binary classification alone.

---

## 10. Key Findings

1. **The signal exists but is modest.** Snap-level tracking features improved ranking over the prevalence baseline, but final ROC-AUC remained below 0.60 and precision below 16%.
2. **The simpler model was justified.** Extra Trees did not show a stable improvement over logistic regression.
3. **Temporal generalization was stable.** Validation and test metrics remained close, and there was no severe week-8 collapse.
4. **The threshold favors sensitivity.** It recovers nearly half of pressures but creates many false alerts.
5. **Ranking is the strongest use case.** High-score groups concentrate more pressure outcomes than the full population.

---

## 11. Limitations and Risks

### 11.1 Limited scope

The data covers only weeks 1-8. Performance may change later in the season, in postseason games, or in another season.

### 11.2 Label dependence

The target inherits any annotation subjectivity, coverage rules, or error in the supplied PFF scouting labels.

### 11.3 Snap-only information

Pressure is affected by events after the snap, including blocking execution, quarterback movement, route development, and player interactions. These were deliberately excluded to preserve the prediction moment.

### 11.4 Correlated rusher observations

Multiple rushers can belong to the same play. The temporal split prevents a play from crossing partitions, but rows within a play are not statistically independent. Future uncertainty estimation could use play-level clustered resampling.

### 11.5 Threshold objective

F1 does not represent a specific scouting or business cost. A production threshold should reflect explicit operational costs and must be selected on new development data rather than changed after reviewing week 8.

### 11.6 Limited precision and external validity

Most positive classifications were false positives. The feature definitions, tracking format, and PFF alignment codes are also specific to the supplied data. Deployment elsewhere would require new schema, label, unit, calibration, and temporal validation.

### 11.7 No causal interpretation

Coefficients, feature differences, and alignment rates represent associations. They do not show that changing a player's position or alignment would cause pressure.

---

## 12. Reproducibility and Quality Controls

The final environment used Python 3.14.0, NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, and joblib 1.5.3. `random_state=42` was used where applicable.

| Notebook | Purpose |
|---|---|
| [`01_data_validation.ipynb`](../notebooks/01_data_validation.ipynb) | Raw-data inventory, schema validation, target validation, and interim exports |
| [`02_analytical_table.ipynb`](../notebooks/02_analytical_table.ipynb) | Snap-level feature engineering and analytical-table validation |
| [`03_exploratory_analysis.ipynb`](../notebooks/03_exploratory_analysis.ipynb) | Exploratory analysis and temporal split |
| [`04_modeling_baselines.ipynb`](../notebooks/04_modeling_baselines.ipynb) | Dummy and logistic baselines |
| [`05_advanced_modeling.ipynb`](../notebooks/05_advanced_modeling.ipynb) | Advanced candidates, temporal CV, and bootstrap comparison |
| [`06_threshold_selection.ipynb`](../notebooks/06_threshold_selection.ipynb) | Validation-only threshold selection and policy lock |
| [`07_final_evaluation.ipynb`](../notebooks/07_final_evaluation.ipynb) | Controlled final training and one-time week-8 evaluation |

The final clean execution confirmed 14 of 14 required PASS markers, 93 of 93 quality controls, 12 of 12 artifact hashes, sequential execution counts, zero error outputs, and no post-test adjustments. The final manifest status was `final_test_evaluation_completed_without_post_test_adjustment`.

Principal reproducibility artifacts include:

| Artifact | Purpose |
|---|---|
| [`stage7_selected_model.joblib`](../models/stage7_selected_model.joblib) | Selected model before threshold tuning |
| [`stage8_selected_policy.json`](stage8_selected_policy.json) | Locked model, features, threshold, and training protocol |
| [`stage9_final_model.joblib`](../models/stage9_final_model.joblib) | Model fitted once on weeks 1-7 |
| [`stage9_pretest_manifest.json`](stage9_pretest_manifest.json) | Evidence recorded before final test access |
| [`stage9_test_predictions.parquet`](stage9_test_predictions.parquet) | Final row-level probabilities and decisions |
| [`stage9_test_metrics.csv`](stage9_test_metrics.csv) | Final metrics |
| [`stage9_final_quality_report.csv`](stage9_final_quality_report.csv) | Final 93 quality controls |
| [`stage9_final_manifest.json`](stage9_final_manifest.json) | Final artifact inventory and hashes |
| [`report_visualizations.py`](../src/nfl_pressure/report_visualizations.py) | Reproducible report figures |

---

## 13. Conclusion

This project produced a validated, temporally evaluated pipeline for estimating pass-rush pressure probability at the snap.

The evidence shows that snap-level tracking features contain a real but limited ranking signal; transparent logistic regression performs essentially as well as the tested nonlinear alternatives; performance remained stable from validation week 7 to final test week 8; and the locked threshold recovers nearly half of real pressures while producing many false positives.

The model is therefore best positioned as an analytical prioritization tool. It can rank pass-rusher observations for review, enrich football analysis, or provide one input to a broader system containing post-snap context and domain expertise.

Most importantly, the final result was obtained under a pre-declared and auditable protocol. No model, feature, hyperparameter, threshold, or calibration change was made after the test data was accessed.

---

## Appendix A. Metric Definitions

| Metric | Practical interpretation |
|---|---|
| Average precision | How effectively higher probabilities concentrate rare pressure outcomes |
| ROC-AUC | Probability that a random pressure receives a higher score than a random non-pressure |
| Log loss | Penalty for inaccurate and overconfident probabilities; lower is better |
| Precision | Fraction of predicted pressures that were actual pressures |
| Recall | Fraction of actual pressures detected by the threshold |
| F1 | Harmonic mean of precision and recall |
| Specificity | Fraction of actual non-pressures correctly classified |
| Brier score | Mean squared difference between probability and outcome; lower is better |
| Lift | Group pressure rate divided by overall pressure prevalence |

## Appendix B. Final Test Summary

| Item | Locked final value |
|---|---|
| Selected model | `LogisticRegression_unweighted` |
| Final training period | Weeks 1-7 |
| Final test period | Week 8 |
| Predictors | 51 |
| Classification threshold | `0.12485514806532311` |
| Week-8 rows | 4,444 |
| Week-8 plays | 1,031 |
| Week-8 positives | 504 |
| Week-8 average precision | 0.157750 |
| Week-8 ROC-AUC | 0.596890 |
| Week-8 log loss | 0.347812 |
| Week-8 F1 | 0.238332 |
| Week-8 top-decile lift | 1.684234 |
| Post-test adjustments | None |
