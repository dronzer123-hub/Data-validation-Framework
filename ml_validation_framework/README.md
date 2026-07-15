# ML Data Validation Framework

A production-ready data validation framework that **automatically runs inside your scikit-learn ML pipeline**. It catches schema violations, null values, range breaches, data drift, outliers, and distribution shifts — before bad data reaches your model.

---

## Project Structure

```
ml_validation_framework/
├── data_validator/
│   ├── __init__.py
│   ├── schema_validator.py      # Schema rules: types, nulls, ranges, regex
│   ├── statistical_validator.py # Drift, outliers, skewness detection
│   ├── pipeline_validator.py    # sklearn-compatible orchestrator
│   ├── report_generator.py      # HTML report builder
│   └── config_loader.py         # Load rules from YAML
├── configs/
│   └── validation_config.yaml   # Example rule config file
├── examples/
│   ├── full_pipeline_example.py # End-to-end sklearn pipeline demo
│   └── standalone_example.py    # Standalone usage (no sklearn)
├── tests/
│   └── test_validators.py       # 16 unit tests
├── reports/                     # Auto-generated HTML + JSON reports
├── requirements.txt
├── setup.py
└── README.md
```

---

## Step-by-Step: How to Run

### Step 1 — Clone / Download

```bash
# If using git:
git clone <your-repo-url>
cd ml_validation_framework

# Or just unzip the downloaded folder and cd into it
cd ml_validation_framework
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate it:
# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `scikit-learn`, `PyYAML`, `pytest`

### Step 4 — Run the tests (verify everything works)

```bash
python -m pytest tests/ -v
```

Expected output: **16 passed** in ~6 seconds.

### Step 5 — Run the standalone example

```bash
python examples/standalone_example.py
```

This loads synthetic data with intentional issues (nulls, invalid values, drift) and prints:
```
VALIDATION RESULTS
==================================================
  Valid     : False
  Checks    : 25
  Passed    : 21
  Errors    : 3
  Warnings  : 1

ERRORS:
  [age] Column 'age': 6 null values found (not allowed)
  [income] Column 'income': 1 values below min 0
  [income] 'income': mean drifted 298.39% (threshold 10.00%)
```

### Step 6 — Run the full sklearn pipeline example

```bash
python examples/full_pipeline_example.py
```

This creates a **complete ML pipeline** (validate → scale → train RandomForest), runs it through train and inference phases, and prints a classification report.

### Step 7 — Open the HTML report

After running either example, open the generated report in your browser:

```bash
# macOS
open reports/validation_*.html

# Linux
xdg-open reports/validation_*.html

# Windows
start reports/validation_*.html
```

The report shows: pass/fail counts, per-column stats, all errors/warnings, and a full check breakdown.

---

## How to Use in YOUR Pipeline

### Option A — sklearn Pipeline (auto-triggered on fit + predict)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from data_validator.schema_validator import SchemaConfig, ColumnRule
from data_validator.pipeline_validator import PipelineValidator, PipelineConfig

# 1. Define validation rules
schema = SchemaConfig(columns={
    "age":    ColumnRule(nullable=False, min_value=0, max_value=120),
    "salary": ColumnRule(min_value=0),
    "score":  ColumnRule(min_value=300, max_value=850),
})

config = PipelineConfig(
    schema_config=schema,
    strict_mode=True,      # Raises ValidationError if errors found
    save_reports=True,     # Saves HTML + JSON reports to reports/
)

# 2. Insert validator as first step
pipeline = Pipeline([
    ("validate", PipelineValidator(config)),   # <-- AUTO-TRIGGERED HERE
    ("scale",    StandardScaler()),
    ("model",    RandomForestClassifier()),
])

# 3. Validation runs automatically on every fit() and predict()
pipeline.fit(X_train, y_train)     # validates X_train
pipeline.predict(X_test)           # validates X_test
```

### Option B — Standalone (no sklearn)

```python
from data_validator.pipeline_validator import PipelineValidator, PipelineConfig

config = PipelineConfig(strict_mode=False, save_reports=True)
validator = PipelineValidator(config)

validator.fit(reference_df)          # Learn baseline statistics
report = validator.validate(new_df)  # Validate incoming data

print(report["summary"])
```

### Option C — Load rules from YAML

```python
from data_validator.config_loader import load_schema_config, load_stat_config
from data_validator.pipeline_validator import PipelineValidator, PipelineConfig
from data_validator.statistical_validator import StatConfig

schema  = load_schema_config("configs/validation_config.yaml")
stat    = load_stat_config("configs/validation_config.yaml")
config  = PipelineConfig(schema_config=schema, stat_config=stat)

validator = PipelineValidator(config)
```

---

## Configuration Reference

### SchemaConfig / ColumnRule

| Parameter        | Type           | Description                                      |
|-----------------|----------------|--------------------------------------------------|
| `dtype`          | `str`          | Expected pandas dtype (e.g. `"float64"`)         |
| `nullable`       | `bool`         | Whether null values are allowed (default `True`) |
| `min_value`      | `float`        | Minimum allowed value                            |
| `max_value`      | `float`        | Maximum allowed value                            |
| `allowed_values` | `list`         | Whitelist of valid values                        |
| `regex`          | `str`          | Regex pattern for string columns                 |
| `unique`         | `bool`         | All values must be unique                        |

### StatConfig

| Parameter                  | Default | Description                              |
|---------------------------|---------|------------------------------------------|
| `z_score_threshold`        | `3.0`   | Z-score above which a value is an outlier|
| `drift_threshold`          | `0.10`  | Max allowed fractional mean drift        |
| `missing_rate_threshold`   | `0.05`  | Max allowed fraction of missing values   |
| `skewness_threshold`       | `2.0`   | Max allowed absolute skewness            |
| `variance_ratio_threshold` | `2.0`   | Max ratio of new/reference variance      |

### PipelineConfig

| Parameter           | Default     | Description                                    |
|--------------------|-------------|------------------------------------------------|
| `strict_mode`       | `True`      | Raise `ValidationError` if errors found        |
| `run_schema_checks` | `True`      | Toggle schema validation on/off                |
| `run_stat_checks`   | `True`      | Toggle statistical checks on/off               |
| `save_reports`      | `True`      | Save HTML + JSON reports automatically         |
| `report_dir`        | `"reports"` | Directory where reports are saved              |

---

## What Gets Validated

### Schema Checks
- ✅ Required columns present
- ✅ Column data types match expected
- ✅ Null values within allowed policy
- ✅ Numeric values within min/max range
- ✅ Categorical values in allowed set
- ✅ String values match regex pattern
- ✅ Unique constraint enforcement
- ✅ Minimum and maximum row counts

### Statistical Checks
- ✅ Mean drift detection (vs. reference data)
- ✅ Variance ratio shift detection
- ✅ Outlier detection via z-score
- ✅ Skewness / distribution shape check
- ✅ Missing rate threshold enforcement
- ✅ Feature correlation structure shift

---

## Reports

Every validation run saves two files to `reports/`:

- `validation_YYYYMMDD_HHMMSS.html` — visual report, open in any browser
- `validation_YYYYMMDD_HHMMSS.json` — machine-readable, use in CI/CD

---

## Running in CI/CD

```bash
# In your CI script:
python -c "
import pandas as pd
from data_validator.pipeline_validator import PipelineValidator, PipelineConfig
config = PipelineConfig(strict_mode=True, save_reports=True)
v = PipelineValidator(config)
v.fit(pd.read_csv('data/train.csv'))
v.transform(pd.read_csv('data/incoming.csv'))
print('Validation passed ✓')
"
```

If validation fails, the script exits with a non-zero code and CI fails.
