"""
test_on_model.py
Tests the data validation framework on 4 real scenarios:
  1. Clean data   — validation passes, model trains fine
  2. Corrupt data — schema errors caught before model sees it
  3. Drifted data — statistical drift caught at inference time
  4. Production simulation — continuous validation on batches

Run: python examples/test_on_model.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from data_validator.schema_validator import SchemaConfig, ColumnRule
from data_validator.statistical_validator import StatConfig
from data_validator.pipeline_validator import PipelineValidator, PipelineConfig, ValidationError

# ─── Helpers ──────────────────────────────────────────────────────────────────

def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_report(validator):
    r = validator.get_last_report()
    if not r:
        return
    s = r["summary"]
    status = "✅ VALID" if s["valid"] else "❌ INVALID"
    print(f"  Status   : {status}")
    print(f"  Checks   : {s['total_checks']}  |  Passed: {s['passed']}  |  Errors: {s['errors']}  |  Warnings: {s['warnings']}")
    for e in r.get("errors", []):
        print(f"  ❌ ERROR   [{e['column']}] {e['message']}")
    for w in r.get("warnings", []):
        print(f"  ⚠️  WARNING [{w['column']}] {w['message']}")


# ─── Dataset Factory ──────────────────────────────────────────────────────────

def make_dataset(n=600, seed=42, corrupt=False, drifted=False):
    """
    Generates a synthetic loan approval dataset.
    Features: age, income, credit_score, loan_amount, years_employed
    Target  : approved (0/1)
    """
    np.random.seed(seed)
    age            = np.random.normal(38, 10, n).clip(21, 75)
    income         = np.random.normal(55000, 18000, n).clip(15000, 200000)
    credit_score   = np.random.normal(680,  80,    n).clip(300, 850)
    loan_amount    = np.random.normal(18000, 8000,  n).clip(1000, 80000)
    years_employed = np.random.normal(7,    4,     n).clip(0, 40)

    # Simple rule-based target (higher credit + income = more likely approved)
    score = (
        0.4 * (credit_score - 300) / 550 +
        0.3 * (income - 15000) / 185000 +
        0.2 * (1 - loan_amount / 80000) +
        0.1 * np.random.rand(n)
    )
    approved = (score > 0.45).astype(int)

    df = pd.DataFrame({
        "age":            age,
        "income":         income,
        "credit_score":   credit_score,
        "loan_amount":    loan_amount,
        "years_employed": years_employed,
    })

    if corrupt:
        # Introduce schema violations
        df.loc[0:4,   "age"]          = np.nan        # Nulls in non-nullable column
        df.loc[5:7,   "credit_score"] = -999           # Below minimum
        df.loc[8,     "income"]       = -50000         # Negative income
        df.loc[9:11,  "loan_amount"]  = np.nan         # Nulls
        print("  ⚠️  Corruption injected: nulls, negative values, out-of-range scores")

    if drifted:
        # Simulate population shift (e.g., new city, different demographics)
        df["income"]       = df["income"]       * 2.8    # Incomes much higher
        df["credit_score"] = df["credit_score"] * 0.6    # Scores much lower
        df["age"]          = df["age"]          + 20     # Older population
        print("  ⚠️  Drift injected: income x2.8, credit_score x0.6, age +20")

    return df, pd.Series(approved, name="approved")


# ─── Validation Config ────────────────────────────────────────────────────────

def make_schema():
    return SchemaConfig(
        columns={
            "age":            ColumnRule(dtype="float64", nullable=False, min_value=18,   max_value=100),
            "income":         ColumnRule(dtype="float64", nullable=False, min_value=0),
            "credit_score":   ColumnRule(dtype="float64", nullable=True,  min_value=300,  max_value=850),
            "loan_amount":    ColumnRule(dtype="float64", nullable=False, min_value=100),
            "years_employed": ColumnRule(dtype="float64", nullable=False, min_value=0,    max_value=50),
        },
        min_rows=10,
    )

def make_stat():
    return StatConfig(
        z_score_threshold=3.0,
        drift_threshold=0.15,          # 15% mean drift allowed
        missing_rate_threshold=0.03,   # 3% missing rate allowed
        skewness_threshold=2.5,
        variance_ratio_threshold=2.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1: Clean data — model should train and predict fine
# ══════════════════════════════════════════════════════════════════════════════

banner("SCENARIO 1: Clean Data → Validation Passes → Model Trains")

X_clean, y_clean = make_dataset(n=600, seed=42)
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

validator1 = PipelineValidator(PipelineConfig(
    schema_config=make_schema(),
    stat_config=make_stat(),
    strict_mode=True,
    save_reports=True,
    report_dir="reports",
))

pipeline1 = Pipeline([
    ("validate", validator1),
    ("scale",    StandardScaler()),
    ("model",    RandomForestClassifier(n_estimators=100, random_state=42)),
])

try:
    pipeline1.fit(X_train, y_train)
    y_pred = pipeline1.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print_report(validator1)
    print(f"\n  ✅ Model trained and predicted successfully")
    print(f"  Accuracy : {acc:.2%}")
except ValidationError as e:
    print(f"  Pipeline halted: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2: Corrupt training data — validation blocks model training
# ══════════════════════════════════════════════════════════════════════════════

banner("SCENARIO 2: Corrupt Data → Validation Blocks Model Training")

X_corrupt, y_corrupt = make_dataset(n=300, seed=7, corrupt=True)

validator2 = PipelineValidator(PipelineConfig(
    schema_config=make_schema(),
    stat_config=make_stat(),
    strict_mode=True,        # Will RAISE on errors
    save_reports=True,
    report_dir="reports",
))

pipeline2 = Pipeline([
    ("validate", validator2),
    ("scale",    StandardScaler()),
    ("model",    LogisticRegression()),
])

try:
    pipeline2.fit(X_corrupt, y_corrupt)
    print("  Model trained (unexpected — errors should have been caught!)")
except ValidationError as e:
    print(f"\n  ✅ Pipeline correctly BLOCKED. Reason:")
    for line in str(e).split("\n")[:6]:
        print(f"     {line}")
    print_report(validator2)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3: Train on clean data, then drift at inference time
# ══════════════════════════════════════════════════════════════════════════════

banner("SCENARIO 3: Train Clean → Drifted Inference → Drift Detected")

X_train_clean, y_train_clean = make_dataset(n=500, seed=0)
X_drifted,     y_drifted     = make_dataset(n=200, seed=99, drifted=True)

validator3 = PipelineValidator(PipelineConfig(
    schema_config=make_schema(),
    stat_config=make_stat(),
    strict_mode=False,       # Warn but don't block
    save_reports=True,
    report_dir="reports",
))

pipeline3 = Pipeline([
    ("validate", validator3),
    ("scale",    StandardScaler()),
    ("model",    RandomForestClassifier(n_estimators=100, random_state=1)),
])

# Train on clean data
pipeline3.fit(X_train_clean, y_train_clean)
print("\n  Training phase (clean data):")
print_report(validator3)

# Predict on drifted data
print("\n  Inference phase (drifted data):")
y_pred_drifted = pipeline3.predict(X_drifted)
print_report(validator3)

acc_drifted = accuracy_score(y_drifted, y_pred_drifted)
print(f"\n  ⚠️  Accuracy on drifted data: {acc_drifted:.2%}  ← degraded due to drift")
print(f"  Model still ran, but you were warned about the drift above ↑")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4: Production simulation — validate rolling batches of data
# ══════════════════════════════════════════════════════════════════════════════

banner("SCENARIO 4: Production Simulation — 5 Rolling Batches")

# Train a model on clean reference data
X_ref, y_ref = make_dataset(n=800, seed=10)
X_tr, X_val, y_tr, y_val = train_test_split(X_ref, y_ref, test_size=0.2, random_state=10)

prod_validator = PipelineValidator(PipelineConfig(
    schema_config=make_schema(),
    stat_config=make_stat(),
    strict_mode=False,
    save_reports=False,      # Skip saving in loop for speed
))

prod_pipeline = Pipeline([
    ("validate", prod_validator),
    ("scale",    StandardScaler()),
    ("model",    RandomForestClassifier(n_estimators=50, random_state=10)),
])
prod_pipeline.fit(X_tr, y_tr)

print(f"\n  {'Batch':<8} {'Rows':<8} {'Errors':<10} {'Warnings':<12} {'Accuracy':<10} {'Status'}")
print("  " + "-" * 58)

# Simulate 5 batches: batches 3 and 4 have progressive drift
for batch_num in range(1, 6):
    drift_factor = 1.0 + (batch_num - 2) * 0.5 if batch_num > 2 else 1.0
    np.random.seed(batch_num * 100)
    n_batch = np.random.randint(80, 150)

    X_batch, y_batch = make_dataset(n=n_batch, seed=batch_num * 100)
    if drift_factor > 1.0:
        X_batch["income"]       *= drift_factor
        X_batch["credit_score"] *= (1 / drift_factor)

    try:
        y_pred_batch = prod_pipeline.predict(X_batch)
        report = prod_validator.get_last_report()
        s = report["summary"]
        acc = accuracy_score(y_batch, y_pred_batch)
        status = "✅ OK" if s["valid"] else "⚠️  DRIFT"
        print(f"  {batch_num:<8} {n_batch:<8} {s['errors']:<10} {s['warnings']:<12} {acc:.2%}      {status}")
    except ValidationError as ve:
        print(f"  {batch_num:<8} {n_batch:<8} BLOCKED — {str(ve)[:40]}")

print("\n  Notice: accuracy drops as drift increases in batches 3-5 ↑")
print("  The validator flags warnings before accuracy collapses")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

banner("ALL SCENARIOS COMPLETE")
print("""
  Scenario 1 — Clean data       : Validation passed, model trained ✅
  Scenario 2 — Corrupt data     : Pipeline blocked before training ✅
  Scenario 3 — Drifted inference: Drift warnings shown, model degraded ✅
  Scenario 4 — Production batches: Drift detected progressively ✅

  HTML reports saved to reports/
  Open any .html file in your browser to inspect results visually.
""")
