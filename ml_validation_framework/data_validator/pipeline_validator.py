"""
pipeline_validator.py
Orchestrates SchemaValidator + StatisticalValidator.
Acts as a scikit-learn compatible pipeline step (fit/transform/fit_transform).
Can be inserted directly into sklearn Pipeline.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging
import json
import os
from pathlib import Path
from datetime import datetime

from .schema_validator import SchemaValidator, SchemaConfig
from .statistical_validator import StatisticalValidator, StatConfig
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails with severity=error and strict_mode=True."""
    pass


@dataclass
class PipelineConfig:
    schema_config: Optional[SchemaConfig] = None
    stat_config: Optional[StatConfig] = None
    strict_mode: bool = True            # If True, raise on validation error
    run_stat_checks: bool = True        # Toggle statistical checks
    run_schema_checks: bool = True      # Toggle schema checks
    report_dir: str = "reports"         # Where to save HTML/JSON reports
    save_reports: bool = True
    log_level: str = "INFO"


class PipelineValidator:
    """
    Drop-in sklearn-compatible data validation step.

    Standalone usage:
        validator = PipelineValidator(config)
        validator.fit(X_train)                # Learn reference statistics
        X_validated = validator.transform(X_test)  # Validate & return data

    Sklearn Pipeline:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([
            ("validate", PipelineValidator(config)),
            ("scale",    StandardScaler()),
            ("model",    YourModel()),
        ])
        pipe.fit(X_train, y_train)
        pipe.predict(X_test)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self._schema_validator: Optional[SchemaValidator] = None
        self._stat_validator: Optional[StatisticalValidator] = None
        self._fitted = False
        self.last_report: Optional[Dict] = None

        if self.config.schema_config and self.config.run_schema_checks:
            self._schema_validator = SchemaValidator(self.config.schema_config)

        if self.config.run_stat_checks:
            self._stat_validator = StatisticalValidator(
                self.config.stat_config or StatConfig()
            )

        Path(self.config.report_dir).mkdir(parents=True, exist_ok=True)

    # ─── sklearn API ──────────────────────────────────────────────────────────

    def fit(self, X, y=None):
        """Fit reference statistics on training data."""
        df = self._to_df(X)
        logger.info(f"Fitting PipelineValidator on {df.shape[0]} rows × {df.shape[1]} cols")

        if self._stat_validator and self.config.run_stat_checks:
            self._stat_validator.fit(df)

        self._fitted = True
        logger.info("PipelineValidator fitted ✓")
        return self

    def transform(self, X, y=None):
        """Validate data; return it unchanged (raises if strict_mode and errors found)."""
        df = self._to_df(X)
        logger.info(f"Validating {df.shape[0]} rows × {df.shape[1]} cols")

        all_results = []

        if self._schema_validator and self.config.run_schema_checks:
            schema_results = self._schema_validator.validate(df)
            all_results.extend(schema_results)

        if self._stat_validator and self.config.run_stat_checks:
            if not self._fitted:
                logger.warning("Statistical validator not fitted — running standalone checks")
            stat_results = self._stat_validator.validate(df)
            all_results.extend(stat_results)

        # Build report
        report = self._build_report(all_results, df)
        self.last_report = report

        if self.config.save_reports:
            self._save_reports(report)

        # Log summary
        errors = [r for r in all_results if not r.passed and r.severity == "error"]
        warnings = [r for r in all_results if not r.passed and r.severity == "warning"]
        logger.info(
            f"Validation summary: {len(all_results)} checks | "
            f"{len(errors)} errors | {len(warnings)} warnings"
        )

        if errors and self.config.strict_mode:
            error_msgs = "\n  ".join(r.message for r in errors)
            raise ValidationError(
                f"Data validation failed with {len(errors)} error(s):\n  {error_msgs}"
            )

        return X  # Pass data through unchanged

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    # ─── Convenience ──────────────────────────────────────────────────────────

    def validate(self, X) -> Dict:
        """Alias: validate without fitting. Returns report dict."""
        self.transform(X)
        return self.last_report

    def get_last_report(self) -> Optional[Dict]:
        return self.last_report

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _to_df(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        raise TypeError(f"Expected DataFrame or ndarray, got {type(X)}")

    def _build_report(self, all_results, df: pd.DataFrame) -> Dict:
        errors = [r for r in all_results if not r.passed and r.severity == "error"]
        warnings = [r for r in all_results if not r.passed and r.severity == "warning"]
        passed = [r for r in all_results if r.passed]

        report = {
            "timestamp": datetime.now().isoformat(),
            "data_shape": {"rows": len(df), "columns": len(df.columns)},
            "summary": {
                "total_checks": len(all_results),
                "passed": len(passed),
                "errors": len(errors),
                "warnings": len(warnings),
                "valid": len(errors) == 0,
            },
            "errors": [
                {"check": r.check if hasattr(r, "check") else r.rule,
                 "column": r.column, "message": r.message,
                 "details": r.details}
                for r in errors
            ],
            "warnings": [
                {"check": r.check if hasattr(r, "check") else r.rule,
                 "column": r.column, "message": r.message,
                 "details": r.details}
                for r in warnings
            ],
            "all_results": [
                {"check": r.check if hasattr(r, "check") else r.rule,
                 "column": r.column, "passed": r.passed,
                 "message": r.message, "severity": r.severity}
                for r in all_results
            ],
            "column_stats": self._column_stats(df),
        }
        return report

    def _column_stats(self, df: pd.DataFrame) -> Dict:
        stats = {}
        for col in df.columns:
            s = df[col]
            entry = {"dtype": str(s.dtype), "null_count": int(s.isna().sum()),
                     "null_pct": round(float(s.isna().mean()) * 100, 2)}
            if pd.api.types.is_numeric_dtype(s):
                entry.update({
                    "mean": round(float(s.mean()), 4) if not s.isna().all() else None,
                    "std":  round(float(s.std()), 4) if not s.isna().all() else None,
                    "min":  round(float(s.min()), 4) if not s.isna().all() else None,
                    "max":  round(float(s.max()), 4) if not s.isna().all() else None,
                })
            stats[col] = entry
        return stats

    def _save_reports(self, report: Dict):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # JSON
        json_path = Path(self.config.report_dir) / f"validation_{ts}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"JSON report saved: {json_path}")
        # HTML
        html_path = Path(self.config.report_dir) / f"validation_{ts}.html"
        ReportGenerator.save_html(report, str(html_path))
        logger.info(f"HTML report saved: {html_path}")
