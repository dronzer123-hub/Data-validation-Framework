"""
standalone_example.py
Use the validator standalone (no sklearn required).
Run: python examples/standalone_example.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from data_validator.schema_validator import SchemaConfig, ColumnRule
from data_validator.statistical_validator import StatisticalValidator, StatConfig
from data_validator.pipeline_validator import PipelineValidator, PipelineConfig

# Synthetic reference data
np.random.seed(0)
ref_df = pd.DataFrame({
    "age":    np.random.normal(35, 10, 300).clip(18, 80),
    "income": np.random.normal(50000, 15000, 300).clip(10000, 200000),
    "score":  np.random.normal(700, 50, 300).clip(300, 850),
})

# New data with issues
new_df = ref_df.copy()
new_df.loc[0:5, "age"] = np.nan          # Introduce nulls
new_df.loc[10, "income"] = -9999         # Invalid negative
new_df["income"] = new_df["income"] * 4  # Massive drift

schema = SchemaConfig(
    columns={
        "age":    ColumnRule(nullable=False, min_value=18, max_value=100),
        "income": ColumnRule(min_value=0),
        "score":  ColumnRule(min_value=300, max_value=850),
    },
    min_rows=10,
)

config = PipelineConfig(
    schema_config=schema,
    strict_mode=False,
    save_reports=True,
    report_dir="reports",
)

validator = PipelineValidator(config)
validator.fit(ref_df)                    # Learn reference stats
report = validator.validate(new_df)      # Validate incoming data

print("VALIDATION RESULTS")
print("=" * 50)
summary = report["summary"]
print(f"  Valid     : {summary['valid']}")
print(f"  Checks    : {summary['total_checks']}")
print(f"  Passed    : {summary['passed']}")
print(f"  Errors    : {summary['errors']}")
print(f"  Warnings  : {summary['warnings']}")

if report["errors"]:
    print("\nERRORS:")
    for e in report["errors"]:
        print(f"  [{e['column']}] {e['message']}")

if report["warnings"]:
    print("\nWARNINGS:")
    for w in report["warnings"]:
        print(f"  [{w['column']}] {w['message']}")

print("\nHTML report saved to reports/")
