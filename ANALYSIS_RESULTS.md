# Analysis Results

## Executive Summary

The analysis used 2,500 synthetic care-management workload records with 73 original variables.

### Key Findings

- **High-risk patients:** 236 patients (9.4% of the population).
- **Average open care gaps:** 1.55 per patient.
- **Synthetic 30-day unplanned acute-event rate:** 4.5%.
- High-risk patients had an **11.4%** acute-event rate compared with **3.0%** among low-risk patients.
- High-risk patients averaged **2.78 open care gaps**, versus **1.13** among low-risk patients.
- The analytically defined **Critical** priority band contained 57 patients and had an acute-event rate of approximately **19.3%**.
- Patients in approximately the top 10% of analytic scores captured **23.9% of all acute events**, with a **10.3% event rate**.
- The existing legacy P1 bucket contained 287 patients (11.5% of the population) but captured only about **15.9% of all acute events**.

This indicates that the transparent analytic priority score concentrated higher-risk patients more effectively than the intentionally imperfect legacy priority field.

## Logistic Regression Results

Test-set performance:

| Metric | Result |
|---|---:|
| Accuracy | 0.651 |
| Precision | 0.072 |
| Recall | 0.571 |
| F1 | 0.128 |
| ROC-AUC | 0.627 |

The outcome is intentionally imbalanced: only about 4.5% of patients experience the synthetic acute event. The model used `class_weight="balanced"` to increase sensitivity to the minority class.

Recall of **57.1%** means that the model identified a little over half of the true acute-event cases in the held-out test set. Precision is low because acute events are rare and the model deliberately prioritizes sensitivity.

The ROC-AUC of **0.627** shows modest discrimination. This should be treated as a baseline model rather than a production-ready prediction model.

## Important Interpretation

The business-rule priority score and the logistic-regression model solve related but different problems:

- The **priority score** is transparent and operational: it ranks the worklist based on clinically and operationally interpretable signals.
- The **logistic regression** estimates statistical association with the synthetic 30-day acute-event target.

For a care-management product, the transparent priority score may be easier for care managers and clinical stakeholders to understand initially, while the predictive model can be evaluated as a later enhancement.

## Portfolio Story

The project demonstrates an end-to-end analytics workflow:

1. Consolidate fragmented care-management signals.
2. Profile data quality and missingness.
3. Analyze care gaps and risk tiers.
4. Create a transparent workload-prioritization algorithm.
5. Rank the daily work queue.
6. Benchmark the new ranking against an existing legacy priority.
7. Build an interpretable classification model.
8. Evaluate the model using metrics appropriate for an imbalanced healthcare outcome.
9. Translate analytical findings into operational care-manager actions.

**Important:** All data and outcomes are synthetic and this project is not clinical decision support.
