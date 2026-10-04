## Care-Gap Prioritization - Care Manager Dashboard

## Use Case Scenario

Every morning, a care manager opens a daily workload dashboard containing patients who require outreach, care-gap closure, post-discharge follow-up, medication review, care coordination, provider collaboration, documentation, or education.

The challenge is that a traditional work queue can show **what is due** without adequately showing **who should be addressed first**.

A patient with one routine preventive gap should not necessarily receive the same attention as a patient who was recently discharged, has heart failure, has missed several outreach attempts, has medication-adherence problems, has transportation barriers, and also has multiple open care gaps.

This project models a **care-gap-focused daily prioritization problem**. The dataset combines clinical risk indicators, utilization history, transition-of-care signals, care gaps, medication adherence, social determinants of health, communication history, care coordination needs, provider collaboration items, administrative workload, and education needs.

## Business Problem to Solve
The goal is to use Python analytics to help a care manager identify patients who may need more urgent attention and to organize the daily workload around **risk + care-gap burden + time sensitivity**, rather than simply processing tasks in first-in/first-out order.

A later Python analysis can compare the existing `legacy_priority` against a newly derived priority score and can also model the synthetic outcome `unplanned_acute_event_30d`.

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

## Technology Stack
Python, Pandas, NumPy, Matplotlib, scikit-learn, Logistic Regression, Jupyter

- For the complete executable analysis, see the Jupyter Notebook.
