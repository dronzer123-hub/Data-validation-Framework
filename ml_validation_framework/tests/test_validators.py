"""
test_validators.py
Run: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np

from data_validator.schema_validator import SchemaValidator, SchemaConfig, ColumnRule
from data_validator.statistical_validator import StatisticalValidator, StatConfig
from data_validator.pipeline_validator import PipelineValidator, PipelineConfig, ValidationError


# ─── Schema Validator Tests ───────────────────────────────────────────────────

class TestSchemaValidator:

    def make_df(self):
        return pd.DataFrame({
            "age":    [25, 30, 45, 22, 60],
            "salary": [50000.0, 75000.0, 90000.0, 40000.0, 100000.0],
            "level":  ["Junior", "Senior", "Senior", "Junior", "Lead"],
        })

    def test_valid_data_passes(self):
        config = SchemaConfig(columns={
            "age":    ColumnRule(dtype="int64", nullable=False, min_value=18, max_value=100),
            "salary": ColumnRule(dtype="float64", min_value=0),
            "level":  ColumnRule(allowed_values=["Junior", "Senior", "Lead"]),
        })
        v = SchemaValidator(config)
        results = v.validate(self.make_df())
        errors = [r for r in v.results if not r.passed and r.severity == "error"]
        assert len(errors) == 0

    def test_null_violation(self):
        df = self.make_df()
        df.loc[0, "age"] = None
        config = SchemaConfig(columns={"age": ColumnRule(nullable=False)})
        v = SchemaValidator(config)
        v.validate(df)
        errors = [r for r in v.results if not r.passed and r.severity == "error"]
        assert any("null" in r.rule for r in errors)

    def test_range_violation(self):
        df = self.make_df()
        df.loc[0, "age"] = 200  # Out of range
        config = SchemaConfig(columns={"age": ColumnRule(max_value=100)})
        v = SchemaValidator(config)
        v.validate(df)
        assert not v.is_valid()

    def test_allowed_values_violation(self):
        df = self.make_df()
        df.loc[0, "level"] = "Intern"  # Not allowed
        config = SchemaConfig(columns={"level": ColumnRule(
            allowed_values=["Junior", "Senior", "Lead"]
        )})
        v = SchemaValidator(config)
        v.validate(df)
        assert not v.is_valid()

    def test_missing_required_column(self):
        df = pd.DataFrame({"age": [1, 2, 3]})
        config = SchemaConfig(columns={
            "age": ColumnRule(), "salary": ColumnRule()
        })
        v = SchemaValidator(config)
        v.validate(df)
        assert not v.is_valid()

    def test_min_rows(self):
        df = pd.DataFrame({"age": [1, 2]})
        config = SchemaConfig(min_rows=10)
        v = SchemaValidator(config)
        v.validate(df)
        assert not v.is_valid()

    def test_unique_violation(self):
        df = pd.DataFrame({"id": [1, 1, 2, 3]})
        config = SchemaConfig(columns={"id": ColumnRule(unique=True)})
        v = SchemaValidator(config)
        v.validate(df)
        assert not v.is_valid()


# ─── Statistical Validator Tests ─────────────────────────────────────────────

class TestStatisticalValidator:

    def ref_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "x": np.random.normal(0, 1, 200),
            "y": np.random.normal(100, 10, 200),
        })

    def test_no_drift_passes(self):
        ref = self.ref_df()
        v = StatisticalValidator(StatConfig(drift_threshold=50.0, skewness_threshold=5.0, variance_ratio_threshold=5.0))
        v.fit(ref)
        # Same distribution
        np.random.seed(99)
        new = pd.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(100, 10, 100),
        })
        v.validate(new)
        errors = [r for r in v.results if not r.passed and r.severity == "error"]
        assert len(errors) == 0

    def test_drift_detected(self):
        ref = self.ref_df()
        v = StatisticalValidator(StatConfig(drift_threshold=0.05))
        v.fit(ref)
        drifted = pd.DataFrame({
            "x": np.random.normal(5, 1, 100),   # Mean shifted by 5
            "y": np.random.normal(100, 10, 100),
        })
        results = v.validate(drifted)
        drift_failures = [r for r in results if r.check == "mean_drift" and not r.passed]
        assert len(drift_failures) >= 1

    def test_outlier_warning(self):
        ref = self.ref_df()
        v = StatisticalValidator(StatConfig(z_score_threshold=2.0))
        v.fit(ref)
        outliers = ref.copy()
        outliers.loc[:5, "x"] = 100  # Extreme outliers
        results = v.validate(outliers)
        outlier_fails = [r for r in results if r.check == "outliers" and not r.passed]
        assert len(outlier_fails) >= 1

    def test_standalone_no_fit(self):
        """Should not crash if fit() was never called."""
        v = StatisticalValidator()
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        results = v.validate(df)
        assert isinstance(results, list)


# ─── Pipeline Validator Tests ─────────────────────────────────────────────────

class TestPipelineValidator:

    def test_sklearn_pipeline_fit_transform(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        df = pd.DataFrame({
            "a": np.random.normal(0, 1, 100),
            "b": np.random.normal(5, 2, 100),
        })
        config = PipelineConfig(strict_mode=False, save_reports=False)
        pipe = Pipeline([
            ("validate", PipelineValidator(config)),
            ("scale", StandardScaler()),
        ])
        transformed = pipe.fit_transform(df)
        assert transformed.shape == (100, 2)

    def test_strict_mode_raises(self):
        df = pd.DataFrame({"age": [-100, -200]})  # Invalid ages
        schema = SchemaConfig(columns={"age": ColumnRule(min_value=0)})
        config = PipelineConfig(schema_config=schema, strict_mode=True, save_reports=False)
        v = PipelineValidator(config)
        with pytest.raises(ValidationError):
            v.transform(df)

    def test_non_strict_returns_data(self):
        df = pd.DataFrame({"age": [-100, -200]})
        schema = SchemaConfig(columns={"age": ColumnRule(min_value=0)})
        config = PipelineConfig(schema_config=schema, strict_mode=False, save_reports=False)
        v = PipelineValidator(config)
        result = v.transform(df)
        assert len(result) == 2  # Data returned unchanged

    def test_report_generated(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        config = PipelineConfig(save_reports=False)
        v = PipelineValidator(config)
        v.fit(df)
        v.transform(df)
        assert v.last_report is not None
        assert "summary" in v.last_report
        assert "column_stats" in v.last_report

    def test_numpy_array_input(self):
        X = np.random.rand(50, 4)
        config = PipelineConfig(save_reports=False)
        v = PipelineValidator(config)
        v.fit(X)
        out = v.transform(X)
        assert out.shape == (50, 4)
