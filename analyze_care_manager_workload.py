from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "care_manager_daily_workload.csv"
OUTPUT_DIR = PROJECT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("CARE MANAGER WORKLOAD ANALYSIS")
print("=" * 70)
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")
print(f"Duplicate patient IDs: {df['patient_id'].duplicated().sum():,}")

# ---------------------------------------------------------
# 2. Data-quality analysis
# ---------------------------------------------------------
missing = (
    df.isna().sum()
      .sort_values(ascending=False)
      .to_frame("missing_count")
)
missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
missing = missing[missing["missing_count"] > 0]
missing.to_csv(OUTPUT_DIR / "missing_values_summary.csv")

print("\nColumns with missing values:")
print(missing if not missing.empty else "None")

# ---------------------------------------------------------
# 3. Risk-tier exploratory analysis
# ---------------------------------------------------------
risk_summary = (
    df.groupby("risk_tier", observed=True)
      .agg(
          patients=("patient_id", "count"),
          avg_open_care_gaps=("open_care_gap_count", "mean"),
          avg_ed_visits_90d=("ed_visits_90d", "mean"),
          recent_discharge_rate=("recent_discharge_30d_flag", "mean"),
          acute_event_rate=("unplanned_acute_event_30d", "mean"),
          sla_overdue_rate=("sla_overdue_flag", "mean")
      )
      .reset_index()
)

risk_order = pd.CategoricalDtype(["Low", "Medium", "High"], ordered=True)
risk_summary["risk_tier"] = risk_summary["risk_tier"].astype(risk_order)
risk_summary = risk_summary.sort_values("risk_tier")

for col in ["recent_discharge_rate", "acute_event_rate", "sla_overdue_rate"]:
    risk_summary[col] = (risk_summary[col] * 100).round(1)

risk_summary["avg_open_care_gaps"] = risk_summary["avg_open_care_gaps"].round(2)
risk_summary["avg_ed_visits_90d"] = risk_summary["avg_ed_visits_90d"].round(2)
risk_summary.to_csv(OUTPUT_DIR / "risk_tier_summary.csv", index=False)

print("\nRisk-tier summary:")
print(risk_summary.to_string(index=False))

# ---------------------------------------------------------
# 4. Care-gap analysis
# ---------------------------------------------------------
gap_cols = [
    "a1c_overdue_gap",
    "retinal_exam_gap",
    "nephropathy_screen_gap",
    "bp_control_gap",
    "med_reconciliation_gap",
    "post_discharge_followup_gap",
    "preventive_screening_gap",
    "vaccination_gap",
    "behavioral_health_followup_gap",
    "statin_therapy_gap",
    "medication_adherence_gap"
]

gap_prevalence = (
    df[gap_cols].mean()
      .sort_values(ascending=False)
      .mul(100)
      .round(1)
      .rename("prevalence_pct")
      .reset_index()
      .rename(columns={"index": "care_gap"})
)

gap_prevalence.to_csv(OUTPUT_DIR / "care_gap_prevalence.csv", index=False)

print("\nMost common care gaps:")
print(gap_prevalence.head(10).to_string(index=False))

# ---------------------------------------------------------
# 5. Create analytic priority score
# ---------------------------------------------------------
# This is an interpretable business-rule score, not a clinical score.
# Weights are intentionally transparent so they can be reviewed by
# clinical/operational stakeholders.

risk_points = (
    df["risk_tier"]
      .map({"Low": 0, "Medium": 12, "High": 24})
      .astype(float)
)

df["analytic_priority_score"] = (
    risk_points
    + 16 * df["recent_discharge_30d_flag"]
    + 4 * df["ed_visits_90d"].clip(upper=3)
    + 3 * df["inpatient_admits_12m"].clip(upper=3)
    + 4 * df["open_care_gap_count"].clip(upper=5)
    + 8 * df["medication_adherence_gap"]
    + 3 * df["sdoh_positive_count"].clip(upper=4)
    + 10 * df["symptom_change_reported_flag"]
    + 2 * df["unanswered_outreach_30d"].clip(upper=3)
    + 8 * df["sla_overdue_flag"]
    + 6 * df["frailty_flag"]
    + 4 * df["provider_message_pending_flag"]
    + 4 * df["home_health_need_flag"]
    + 4 * df["care_plan_status"].isin(["Overdue", "Not Started"]).astype(int)
).clip(upper=100).round().astype(int)

df["analytic_priority_band"] = pd.cut(
    df["analytic_priority_score"],
    bins=[-1, 29, 49, 69, 100],
    labels=["Routine", "Elevated", "High", "Critical"]
)

df["analytic_rank"] = (
    df["analytic_priority_score"]
      .rank(method="first", ascending=False)
      .astype(int)
)

# ---------------------------------------------------------
# 6. Produce a prioritized daily worklist
# ---------------------------------------------------------
worklist_cols = [
    "analytic_rank",
    "patient_id",
    "care_manager_id",
    "analytic_priority_score",
    "analytic_priority_band",
    "legacy_priority",
    "risk_tier",
    "task_type",
    "open_care_gap_count",
    "recent_discharge_30d_flag",
    "ed_visits_90d",
    "inpatient_admits_12m",
    "medication_adherence_gap",
    "sdoh_positive_count",
    "symptom_change_reported_flag",
    "unanswered_outreach_30d",
    "sla_overdue_flag",
    "provider_message_pending_flag"
]

top_worklist = (
    df.sort_values(
        ["analytic_priority_score", "open_care_gap_count", "ed_visits_90d"],
        ascending=False
    )[worklist_cols]
    .head(20)
)

top_worklist.to_csv(
    OUTPUT_DIR / "top_20_priority_patients.csv",
    index=False
)

print("\nTop 20 prioritized patients:")
print(top_worklist.to_string(index=False))

# ---------------------------------------------------------
# 7. Compare existing legacy priority with analytic signals
# ---------------------------------------------------------
legacy_summary = (
    df.groupby("legacy_priority", observed=True)
      .agg(
          patients=("patient_id", "count"),
          avg_analytic_score=("analytic_priority_score", "mean"),
          high_risk_rate=("risk_tier", lambda s: (s == "High").mean()),
          acute_event_rate=("unplanned_acute_event_30d", "mean"),
          avg_open_gaps=("open_care_gap_count", "mean")
      )
      .reindex(["P1", "P2", "P3", "P4"])
      .reset_index()
)

legacy_summary["avg_analytic_score"] = legacy_summary["avg_analytic_score"].round(1)
legacy_summary["avg_open_gaps"] = legacy_summary["avg_open_gaps"].round(2)
legacy_summary["high_risk_rate"] = (legacy_summary["high_risk_rate"] * 100).round(1)
legacy_summary["acute_event_rate"] = (legacy_summary["acute_event_rate"] * 100).round(1)

legacy_summary.to_csv(
    OUTPUT_DIR / "legacy_priority_comparison.csv",
    index=False
)

print("\nLegacy priority comparison:")
print(legacy_summary.to_string(index=False))

# Event concentration in high-priority patients
total_events = df["unplanned_acute_event_30d"].sum()
top_10_cutoff = df["analytic_priority_score"].quantile(0.90)
top_10 = df[df["analytic_priority_score"] >= top_10_cutoff]

capture_rate = top_10["unplanned_acute_event_30d"].sum() / total_events
top10_event_rate = top_10["unplanned_acute_event_30d"].mean()

print(
    f"\nAnalytic top-decile event capture: {capture_rate:.1%}"
    f"\nAnalytic top-decile event rate: {top10_event_rate:.1%}"
)

# ---------------------------------------------------------
# 8. Logistic regression
# ---------------------------------------------------------
target = "unplanned_acute_event_30d"

numeric_features = [
    "age",
    "diabetes_flag",
    "chf_flag",
    "copd_flag",
    "ckd_flag",
    "frailty_flag",
    "chronic_condition_count",
    "ed_visits_90d",
    "observation_stays_90d",
    "inpatient_admits_12m",
    "recent_discharge_30d_flag",
    "systolic_bp",
    "open_care_gap_count",
    "medication_count",
    "medication_pdc",
    "medication_adherence_gap",
    "sdoh_positive_count",
    "days_since_last_contact",
    "unanswered_outreach_30d",
    "symptom_change_reported_flag",
    "community_resource_referral_open_flag",
    "home_health_need_flag",
    "provider_message_pending_flag",
    "specialist_count",
    "task_age_days",
    "sla_overdue_flag"
]

categorical_features = [
    "insurance_product",
    "care_plan_status",
    "task_type",
    "health_literacy",
    "preferred_contact_channel"
]

X = df[numeric_features + categorical_features].copy()
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

metrics = pd.DataFrame({
    "metric": ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
    "value": [
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, zero_division=0),
        recall_score(y_test, pred, zero_division=0),
        f1_score(y_test, pred, zero_division=0),
        roc_auc_score(y_test, prob)
    ]
})

metrics["value"] = metrics["value"].round(3)
metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

print("\nLogistic regression test metrics:")
print(metrics.to_string(index=False))

cm = confusion_matrix(y_test, pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)
cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

print("\nConfusion matrix:")
print(cm_df)

# ---------------------------------------------------------
# 9. Model drivers
# ---------------------------------------------------------
feature_names = model.named_steps["preprocess"].get_feature_names_out()
coefficients = model.named_steps["model"].coef_[0]

drivers = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "odds_ratio": np.exp(coefficients)
})

drivers["abs_coefficient"] = drivers["coefficient"].abs()

top_drivers = (
    drivers.sort_values("abs_coefficient", ascending=False)
           .drop(columns="abs_coefficient")
           .head(15)
)

top_drivers["coefficient"] = top_drivers["coefficient"].round(3)
top_drivers["odds_ratio"] = top_drivers["odds_ratio"].round(3)

top_drivers.to_csv(
    OUTPUT_DIR / "top_model_drivers.csv",
    index=False
)

print("\nTop model drivers:")
print(top_drivers.to_string(index=False))

# ---------------------------------------------------------
# 10. Save scored workload
# ---------------------------------------------------------
df.sort_values("analytic_rank").to_csv(
    OUTPUT_DIR / "care_manager_scored_worklist.csv",
    index=False
)

# ---------------------------------------------------------
# 11. Visualizations
# ---------------------------------------------------------
plt.figure(figsize=(7, 4.5))
(
    df["risk_tier"]
      .value_counts()
      .reindex(["Low", "Medium", "High"])
      .plot(kind="bar")
)
plt.title("Patient Count by Risk Tier")
plt.xlabel("Risk Tier")
plt.ylabel("Patients")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "risk_tier_distribution.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 5))
gap_plot = gap_prevalence.head(8).sort_values("prevalence_pct")
plt.barh(gap_plot["care_gap"], gap_plot["prevalence_pct"])
plt.title("Most Common Care Gaps")
plt.xlabel("Prevalence (%)")
plt.ylabel("Care Gap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "care_gap_prevalence.png", dpi=180)
plt.close()

plt.figure(figsize=(7, 4.5))
event_by_band = (
    df.groupby("analytic_priority_band", observed=True)
      ["unplanned_acute_event_30d"]
      .mean()
      .mul(100)
)
event_by_band.plot(kind="bar")
plt.title("30-Day Acute Event Rate by Analytic Priority Band")
plt.xlabel("Analytic Priority Band")
plt.ylabel("Event Rate (%)")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "event_rate_by_priority_band.png",
    dpi=180
)
plt.close()

print("\nAnalysis complete.")
print(f"Outputs written to: {OUTPUT_DIR}")
