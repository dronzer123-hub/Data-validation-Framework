"""
full_pipeline_example.py
Demonstrates automatic validation inside an sklearn ML pipeline.
Run: python examples/full_pipeline_example.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_validator.schema_validator import SchemaConfig, ColumnRule
from data_validator.statistical_validator import StatConfig
from data_validator.pipeline_validator import PipelineValidator, PipelineConfig


# ─── 1. Create synthetic dataset ─────────────────────────────────────────────
np.random.seed(42)
N = 500

df = pd.DataFrame({
    "age":           np.random.normal(40, 12, N).clip(18, 80),
    "salary":        np.random.normal(60000, 20000, N).clip(20000, 200000),
    "credit_score":  np.random.normal(680, 80, N).clip(300, 850),
    "years_exp":     np.random.normal(10, 5, N).clip(0, 40),
    "loan_amount":   np.random.normal(15000, 8000, N).clip(1000, 50000),
    "target":        np.random.randint(0, 2, N),
})

# Introduce a few issues
df.loc[5, "age"] = -5          # Invalid age
df.loc[10, "credit_score"] = 200  # Below range
df.loc[20:25, "salary"] = np.nan  # Some nulls

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ─── 2. Define validation rules ──────────────────────────────────────────────
schema_config = SchemaConfig(
    columns={
        "age":          ColumnRule(dtype="float64", nullable=False, min_value=18, max_value=100),
        "salary":       ColumnRule(dtype="float64", nullable=True, min_value=0),
        "credit_score": ColumnRule(dtype="float64", nullable=True, min_value=300, max_value=850),
        "years_exp":    ColumnRule(dtype="float64", nullable=False, min_value=0, max_value=50),
        "loan_amount":  ColumnRule(dtype="float64", nullable=False, min_value=0),
    },
    min_rows=10,
)

stat_config = StatConfig(
    z_score_threshold=3.0,
    drift_threshold=0.15,
    missing_rate_threshold=0.05,
)

pipeline_config = PipelineConfig(
    schema_config=schema_config,
    stat_config=stat_config,
    strict_mode=False,   # Set True to raise exception on errors
    save_reports=True,
    report_dir="reports",
)


# ─── 3. Build ML Pipeline with validation step ───────────────────────────────
validator = PipelineValidator(pipeline_config)

ml_pipeline = Pipeline([
    ("validate", validator),       # <-- Automatic validation
    ("scale",    StandardScaler()),
    ("model",    RandomForestClassifier(n_estimators=100, random_state=42)),
])


# ─── 4. Train (validation runs automatically during fit) ─────────────────────
print("=" * 60)
print("TRAINING PHASE — validation runs on X_train")
print("=" * 60)
ml_pipeline.fit(X_train, y_train)


# ─── 5. Predict (validation runs automatically during predict) ───────────────
print("\n" + "=" * 60)
print("INFERENCE PHASE — validation runs on X_test")
print("=" * 60)

# Introduce drift in test set to see drift detection
X_test_drifted = X_test.copy()
X_test_drifted["salary"] = X_test_drifted["salary"] * 3.0  # Big drift

y_pred = ml_pipeline.predict(X_test_drifted)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ─── 6. Access validation report ─────────────────────────────────────────────
report = validator.get_last_report()
print("\nValidation Summary:")
for k, v in report["summary"].items():
    print(f"  {k}: {v}")

print(f"\nReports saved in: {pipeline_config.report_dir}/")
print("Open the .html file in your browser for a detailed report.")
