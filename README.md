# Care Manager Care-Gap Prioritization

## Use Case Scenario

Every morning, a care manager opens a daily workload dashboard containing patients who require outreach, care-gap closure, post-discharge follow-up, medication review, care coordination, provider collaboration, documentation, or education.

The challenge is that a traditional work queue can show **what is due** without adequately showing **who should be addressed first**.

A patient with one routine preventive gap should not necessarily receive the same attention as a patient who was recently discharged, has heart failure, has missed several outreach attempts, has medication-adherence problems, has transportation barriers, and also has multiple open care gaps.

This project models a **care-gap-focused daily prioritization problem**. The dataset combines clinical risk indicators, utilization history, transition-of-care signals, care gaps, medication adherence, social determinants of health, communication history, care coordination needs, provider collaboration items, administrative workload, and education needs.

The goal is to use Python analytics to help a care manager identify patients who may need more urgent attention and to organize the daily workload around **risk + care-gap burden + time sensitivity**, rather than simply processing tasks in first-in/first-out order.

## Business Problem

Care managers often receive work items from multiple systems such as claims, EHR feeds, pharmacy data, care-management platforms, and health-information exchange sources. These work items may be individually valid but operationally fragmented.

Without a consolidated prioritization view, the care manager may:

- spend time on lower-risk tasks while a higher-risk patient waits;
- miss time-sensitive post-discharge follow-up;
- fail to see that several care gaps are accumulating for the same patient;
- overlook medication adherence or refill concerns;
- miss social barriers that make a care plan difficult to complete;
- repeat unsuccessful outreach without changing strategy;
- leave provider collaboration or community-resource referrals unresolved;
- exceed internal service-level expectations for overdue work.

### Business Objective

Develop an analytical approach that ranks the daily patient workload so the care manager can focus first on patients with the greatest combination of:

1. **Clinical risk and multimorbidity**
2. **Recent acute-care utilization or discharge**
3. **Open care gaps**
4. **Medication adherence concerns**
5. **Social determinants of health barriers**
6. **Communication and engagement difficulty**
7. **Pending care-coordination/provider actions**
8. **Overdue or aging work items**

A later Python analysis can compare the existing `legacy_priority` against a newly derived priority score and can also model the synthetic outcome `unplanned_acute_event_30d`.

## Example Daily Care-Manager Activities

- **Patient Communication:** scheduled calls/check-ins to review symptoms, medications, and treatment progress.
- **Care Coordination:** connect patients with transportation, food support, home health, or other community resources.
- **Provider Collaboration:** coordinate with PCPs, specialists, caregivers, and family members.
- **Administrative Work:** document outreach, update care plans, and complete activity/billing codes.
- **Patient Education:** explain diagnoses, medications, discharge instructions, and next steps in plain language.
- **Care-Gap Closure:** identify and act on overdue monitoring, screenings, transition-of-care tasks, medication reconciliation, or other eligible gaps.

## Dataset

- **Rows:** 2,500 synthetic patient workload records
- **Columns:** 73
- **Snapshot date:** 2026-10-01
- **High-risk tier:** 9.4%
- **Average open care gaps:** 1.55
- **SLA-overdue work items:** 39.4%
- **Synthetic 30-day acute-event outcome rate:** 4.5%

The data is **fully synthetic and contains no PHI**.

### Files

- `data/care_manager_daily_workload.csv` — main analysis dataset
- `data/data_dictionary.csv` — column definitions and analytics roles
- `src/validate_dataset.py` — lightweight dataset validation script

## Suggested Python Analysis Roadmap

1. Load and profile the dataset.
2. Identify missing/unknown values and data-quality issues.
3. Explore the distribution of risk tiers and open care gaps.
4. Compare care-gap burden by chronic condition and recent utilization.
5. Analyze outreach failures, SDOH barriers, medication adherence, and overdue work.
6. Build a care-manager priority score.
7. Rank the morning work queue from highest to lowest priority.
8. Compare the new score with `legacy_priority`.
9. Build a logistic-regression model for `unplanned_acute_event_30d`.
10. Evaluate precision, recall, ROC-AUC, and operational usefulness.
11. Explain which variables drive risk and convert findings into care-manager actions.
12. Create dashboard-ready summary tables or visualizations.

## Important Note

This repository is a portfolio/learning project. The data and risk logic are synthetic and should not be used for real clinical decision-making.

## Executed Jupyter Notebook

For a recruiter-friendly walkthrough with Python code, actual outputs, charts, and a healthcare business interpretation after each section, open:

`notebooks/care_manager_care_gap_prioritization_analysis.ipynb`

The notebook is committed with executed outputs so GitHub can render the full analysis directly.
