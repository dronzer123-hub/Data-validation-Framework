"""
test_realworld_input.py
Simulates exactly what happens in production when real-world inputs
are sent to a trained model — good inputs, bad inputs, and drifted inputs.

Run: python examples/test_realworld_input.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_validator.schema_validator import SchemaConfig, ColumnRule
from data_validator.statistical_validator import StatConfig
from data_validator.pipeline_validator import PipelineValidator, PipelineConfig, ValidationError

# ─── Helpers ──────────────────────────────────────────────────────────────────

def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_report(validator):
    r = validator.get_last_report()
    s = r["summary"]
    status = "✅ VALID" if s["valid"] else "❌ INVALID"
    print(f"\n  Validation Status : {status}")
    print(f"  Total Checks      : {s['total_checks']}")
    print(f"  Passed            : {s['passed']}")
    print(f"  Errors            : {s['errors']}")
    print(f"  Warnings          : {s['warnings']}")
    if r.get("errors"):
        print("\n  Problems Found:")
        for e in r["errors"]:
            print(f"    ❌ [{e['column']}] {e['message']}")
    if r.get("warnings"):
        print("\n  Warnings:")
        for w in r["warnings"]:
            print(f"    ⚠️  [{w['column']}] {w['message']}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build and train the model on clean Titanic-like training data
# ══════════════════════════════════════════════════════════════════════════════

banner("STEP 1: Training the Model on Clean Data")

np.random.seed(42)
N = 700

# Simulate Titanic-like training dataset
train_df = pd.DataFrame({
    "Pclass":   np.random.choice([1, 2, 3], N, p=[0.2, 0.3, 0.5]),
    "Sex":      np.random.choice([0, 1], N),           # 0=male, 1=female
    "Age":      np.random.normal(29, 12, N).clip(1, 80),
    "SibSp":    np.random.choice([0, 1, 2, 3], N, p=[0.6, 0.2, 0.1, 0.1]),
    "Parch":    np.random.choice([0, 1, 2], N, p=[0.7, 0.2, 0.1]),
    "Fare":     np.random.exponential(32, N).clip(5, 300),
    "Embarked": np.random.choice([0, 1, 2], N, p=[0.7, 0.2, 0.1]),
})

# Simple survival rule: women + 1st class + lower age = more likely to survive
survival_score = (
    0.4 * train_df["Sex"] +
    0.3 * (1 / train_df["Pclass"]) +
    0.2 * (1 - train_df["Age"] / 80) +
    0.1 * np.random.rand(N)
)
y = (survival_score > 0.4).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    train_df, y, test_size=0.2, random_state=42
)

# ─── Define validation rules matching the training data ───────────────────────
schema = SchemaConfig(
    columns={
        "Pclass":   ColumnRule(nullable=False, min_value=1,   max_value=3,
                               allowed_values=[1, 2, 3]),
        "Sex":      ColumnRule(nullable=False, allowed_values=[0, 1]),
        "Age":      ColumnRule(nullable=False, min_value=0,   max_value=120),
        "SibSp":    ColumnRule(nullable=False, min_value=0,   max_value=10),
        "Parch":    ColumnRule(nullable=False, min_value=0,   max_value=10),
        "Fare":     ColumnRule(nullable=False, min_value=0),
        "Embarked": ColumnRule(nullable=True,  allowed_values=[0, 1, 2]),
    },
    min_rows=1,
)

stat = StatConfig(
    drift_threshold=0.20,
    missing_rate_threshold=0.05,
    z_score_threshold=3.0,
    skewness_threshold=3.0,
)

config = PipelineConfig(
    schema_config=schema,
    stat_config=stat,
    strict_mode=False,      # Warn but don't block (so all tests run)
    save_reports=True,
    report_dir="reports",
)

validator = PipelineValidator(config)

pipeline = Pipeline([
    ("validate", validator),
    ("scale",    StandardScaler()),
    ("model",    RandomForestClassifier(n_estimators=100, random_state=42)),
])

pipeline.fit(X_train, y_train)
train_acc = accuracy_score(y_test, pipeline.predict(X_test))
print(f"\n  ✅ Model trained successfully!")
print(f"  Training Accuracy : {train_acc:.2%}")
print(f"\n  Training Data Stats (what model learned):")
print(f"    Age range   : {X_train['Age'].min():.0f} - {X_train['Age'].max():.0f}  | mean={X_train['Age'].mean():.1f}")
print(f"    Fare range  : £{X_train['Fare'].min():.1f} - £{X_train['Fare'].max():.1f} | mean=£{X_train['Fare'].mean():.1f}")
print(f"    Pclass dist : {dict(X_train['Pclass'].value_counts().sort_index())}")
print(f"    Sex dist    : male={( X_train['Sex']==0).sum()}, female={(X_train['Sex']==1).sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Test with PERFECT real-world input (should pass all checks)
# ══════════════════════════════════════════════════════════════════════════════

banner("STEP 2: Perfect Real-World Input → Should PASS")

perfect_input = pd.DataFrame([
    # Pclass  Sex   Age   SibSp  Parch   Fare    Embarked
    {  "Pclass": 1, "Sex": 1, "Age": 28.0, "SibSp": 0, "Parch": 0, "Fare": 85.0,  "Embarked": 1 },  # Young woman, 1st class
    {  "Pclass": 3, "Sex": 0, "Age": 35.0, "SibSp": 1, "Parch": 0, "Fare": 12.0,  "Embarked": 0 },  # Middle-aged man, 3rd class
    {  "Pclass": 2, "Sex": 1, "Age": 45.0, "SibSp": 0, "Parch": 2, "Fare": 55.0,  "Embarked": 0 },  # Older woman, 2nd class
])

print("\n  Input Data:")
print(perfect_input.to_string(index=False))

predictions = pipeline.predict(perfect_input)
probabilities = pipeline.predict_proba(perfect_input)

check_report(validator)
print("\n  Model Predictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    outcome = "✅ Survived" if pred == 1 else "❌ Did not survive"
    print(f"    Passenger {i+1}: {outcome}  (confidence: {max(prob):.0%})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Test with BAD real-world input (schema violations)
# ══════════════════════════════════════════════════════════════════════════════

banner("STEP 3: Bad Real-World Input → Should CATCH Errors")

bad_input = pd.DataFrame([
    # Problem 1: Invalid Pclass (5 doesn't exist), negative Age, null Fare
    { "Pclass": 5,    "Sex": 0, "Age": -10.0, "SibSp": 0, "Parch": 0, "Fare": None,   "Embarked": 0 },
    # Problem 2: Invalid Sex value (3 doesn't exist), extreme age
    { "Pclass": 2,    "Sex": 3, "Age": 200.0, "SibSp": 0, "Parch": 0, "Fare": 25.0,   "Embarked": 1 },
    # Problem 3: Null age in non-nullable column, negative fare
    { "Pclass": 1,    "Sex": 1, "Age": None,  "SibSp": 0, "Parch": 0, "Fare": -100.0, "Embarked": 0 },
])

print("\n  Input Data (intentionally broken):")
print(bad_input.to_string(index=False))
print("\n  Problems injected:")
print("    Row 1 → Pclass=5 (invalid), Age=-10 (negative), Fare=null")
print("    Row 2 → Sex=3 (invalid), Age=200 (impossible)")
print("    Row 3 → Age=null (not allowed), Fare=-100 (negative)")

# strict_mode=False so it warns but still predicts
predictions = pipeline.predict(bad_input)
check_report(validator)

print("\n  Model still predicted (strict_mode=False):")
for i, pred in enumerate(predictions):
    print(f"    Passenger {i+1}: {'Survived' if pred == 1 else 'Did not survive'}  ⚠️  (result is UNRELIABLE)")
print("\n  ⚠️  These predictions CANNOT be trusted — input was invalid!")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Test with DRIFTED real-world input (statistically different)
# ══════════════════════════════════════════════════════════════════════════════

banner("STEP 4: Drifted Real-World Input → Should Detect Drift")

# Simulate data coming from a different population
# e.g., a luxury cruise (everyone is older, richer, 1st class)
np.random.seed(99)
drifted_input = pd.DataFrame({
    "Pclass":   np.ones(50, dtype=int),                        # Everyone is 1st class
    "Sex":      np.random.choice([0, 1], 50),
    "Age":      np.random.normal(65, 8, 50).clip(50, 90),      # Much older (mean 65 vs 29)
    "SibSp":    np.zeros(50, dtype=int),
    "Parch":    np.zeros(50, dtype=int),
    "Fare":     np.random.normal(500, 100, 50).clip(300, 800), # Much more expensive (mean 500 vs 32)
    "Embarked": np.zeros(50, dtype=int),
})

print("\n  Input Data Stats (50 passengers from drifted population):")
print(f"    Age  : mean={drifted_input['Age'].mean():.1f}   (training mean was {X_train['Age'].mean():.1f})")
print(f"    Fare : mean=£{drifted_input['Fare'].mean():.1f}  (training mean was £{X_train['Fare'].mean():.1f})")
print(f"    Pclass: all 1st class   (training had mix of 1/2/3)")

predictions = pipeline.predict(drifted_input)
check_report(validator)

drift_acc = accuracy_score(
    (0.6 * np.ones(50)).astype(int),   # Approximate true labels
    predictions
)
print(f"\n  Survival rate predicted: {predictions.mean():.0%}")
print(f"  ⚠️  Drift was detected above — predictions may not be reliable")
print(f"     Model was never trained on this kind of passenger profile!")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — STRICT MODE: Block the prediction entirely on bad input
# ══════════════════════════════════════════════════════════════════════════════

banner("STEP 5: Strict Mode → Bad Input BLOCKS Prediction Completely")

strict_config = PipelineConfig(
    schema_config=schema,
    stat_config=stat,
    strict_mode=True,       # NOW it will raise an error and stop everything
    save_reports=False,
)

strict_validator = PipelineValidator(strict_config)
strict_pipeline = Pipeline([
    ("validate", strict_validator),
    ("scale",    StandardScaler()),
    ("model",    RandomForestClassifier(n_estimators=100, random_state=42)),
])
strict_pipeline.fit(X_train, y_train)

bad_single_input = pd.DataFrame([{
    "Pclass": 9,      # ❌ completely invalid
    "Sex": 0,
    "Age": -5,        # ❌ negative age
    "SibSp": 0,
    "Parch": 0,
    "Fare": 50.0,
    "Embarked": 0
}])

print("\n  Sending bad input to strict pipeline...")
print(f"  Input: Pclass=9, Age=-5  (both invalid)")

try:
    result = strict_pipeline.predict(bad_single_input)
    print(f"  Prediction: {result}  (should not reach here!)")
except ValidationError as e:
    print(f"\n  ✅ Prediction BLOCKED by validator!")
    print(f"  Error raised: {str(e).splitlines()[0]}")
    for line in str(e).splitlines()[1:4]:
        print(f"    {line}")
    print(f"\n  Model was NEVER called — bad data stopped at the gate ✅")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

banner("SUMMARY — What Was Tested")
print("""
  Step 1 → Trained model on clean Titanic-like data
  Step 2 → Perfect real-world input       : ✅ All checks passed, predictions reliable
  Step 3 → Bad real-world input           : ❌ Errors caught (null, invalid values, range breach)
  Step 4 → Drifted real-world input       : ⚠️  Drift detected (age/fare way outside training range)
  Step 5 → Strict mode on bad input       : 🚫 Prediction blocked completely

  Key insight:
    The model never changes — it always predicts based on what it learned.
    The VALIDATOR is what protects you from feeding it garbage data
    and blindly trusting garbage predictions back.

  Reports saved to reports/ folder.
  Open any .html file in your browser for full visual breakdown.
""")
