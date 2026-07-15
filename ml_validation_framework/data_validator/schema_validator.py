"""
schema_validator.py
Validates DataFrame schema: column types, nulls, ranges, allowed values, shape.
"""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ColumnRule:
    dtype: Optional[str] = None           # Expected dtype (e.g., "float64", "int64", "object")
    nullable: bool = True                  # Allow null values?
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    regex: Optional[str] = None            # For string columns
    unique: bool = False                   # Must all values be unique?


@dataclass
class SchemaConfig:
    columns: Dict[str, ColumnRule] = field(default_factory=dict)
    min_rows: Optional[int] = None
    max_rows: Optional[int] = None
    allow_extra_columns: bool = True
    require_all_columns: bool = True


@dataclass
class ValidationResult:
    passed: bool
    rule: str
    column: Optional[str]
    message: str
    severity: str = "error"    # "error" | "warning"
    details: Optional[Dict] = None


class SchemaValidator:
    """
    Validates a pandas DataFrame against a SchemaConfig.

    Usage:
        config = SchemaConfig(
            columns={
                "age": ColumnRule(dtype="int64", nullable=False, min_value=0, max_value=120),
                "salary": ColumnRule(dtype="float64", min_value=0),
            },
            min_rows=10,
        )
        validator = SchemaValidator(config)
        results = validator.validate(df)
    """

    def __init__(self, config: SchemaConfig):
        self.config = config
        self.results: List[ValidationResult] = []

    def validate(self, df: pd.DataFrame) -> List[ValidationResult]:
        self.results = []
        logger.info("Starting schema validation...")

        self._check_shape(df)
        self._check_required_columns(df)
        self._check_extra_columns(df)

        for col_name, rule in self.config.columns.items():
            if col_name not in df.columns:
                continue
            series = df[col_name]
            self._check_dtype(series, col_name, rule)
            self._check_nulls(series, col_name, rule)
            self._check_range(series, col_name, rule)
            self._check_allowed_values(series, col_name, rule)
            self._check_regex(series, col_name, rule)
            self._check_unique(series, col_name, rule)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed and r.severity == "error")
        logger.info(f"Schema validation complete: {passed} passed, {failed} failed")
        return self.results

    def is_valid(self) -> bool:
        return all(r.passed or r.severity != "error" for r in self.results)

    # ─── Internal Checks ──────────────────────────────────────────────────────

    def _check_shape(self, df: pd.DataFrame):
        rows = len(df)
        if self.config.min_rows and rows < self.config.min_rows:
            self.results.append(ValidationResult(
                passed=False, rule="min_rows", column=None,
                message=f"DataFrame has {rows} rows, expected at least {self.config.min_rows}",
                details={"actual": rows, "expected_min": self.config.min_rows}
            ))
        else:
            self.results.append(ValidationResult(passed=True, rule="min_rows", column=None,
                                                  message=f"Row count {rows} OK"))

        if self.config.max_rows and rows > self.config.max_rows:
            self.results.append(ValidationResult(
                passed=False, rule="max_rows", column=None,
                message=f"DataFrame has {rows} rows, expected at most {self.config.max_rows}",
                details={"actual": rows, "expected_max": self.config.max_rows}
            ))

    def _check_required_columns(self, df: pd.DataFrame):
        if not self.config.require_all_columns:
            return
        missing = [c for c in self.config.columns if c not in df.columns]
        if missing:
            self.results.append(ValidationResult(
                passed=False, rule="required_columns", column=None,
                message=f"Missing required columns: {missing}",
                details={"missing": missing}
            ))
        else:
            self.results.append(ValidationResult(passed=True, rule="required_columns",
                                                  column=None, message="All required columns present"))

    def _check_extra_columns(self, df: pd.DataFrame):
        if self.config.allow_extra_columns:
            return
        extra = [c for c in df.columns if c not in self.config.columns]
        if extra:
            self.results.append(ValidationResult(
                passed=False, rule="extra_columns", column=None,
                message=f"Unexpected columns found: {extra}", severity="warning",
                details={"extra": extra}
            ))

    def _check_dtype(self, series: pd.Series, col: str, rule: ColumnRule):
        if rule.dtype is None:
            return
        actual = str(series.dtype)
        # Allow compatible numeric types
        numeric_types = {"int64", "int32", "float64", "float32"}
        if rule.dtype in numeric_types and actual in numeric_types:
            self.results.append(ValidationResult(passed=True, rule="dtype", column=col,
                                                  message=f"dtype {actual} (compatible with {rule.dtype})"))
            return
        if actual != rule.dtype:
            self.results.append(ValidationResult(
                passed=False, rule="dtype", column=col,
                message=f"Column '{col}': expected dtype {rule.dtype}, got {actual}",
                details={"expected": rule.dtype, "actual": actual}
            ))
        else:
            self.results.append(ValidationResult(passed=True, rule="dtype", column=col,
                                                  message=f"dtype {actual} OK"))

    def _check_nulls(self, series: pd.Series, col: str, rule: ColumnRule):
        null_count = series.isna().sum()
        if not rule.nullable and null_count > 0:
            self.results.append(ValidationResult(
                passed=False, rule="nullable", column=col,
                message=f"Column '{col}': {null_count} null values found (not allowed)",
                details={"null_count": int(null_count), "null_pct": round(null_count / len(series) * 100, 2)}
            ))
        else:
            self.results.append(ValidationResult(passed=True, rule="nullable", column=col,
                                                  message=f"Null check OK ({null_count} nulls)"))

    def _check_range(self, series: pd.Series, col: str, rule: ColumnRule):
        if rule.min_value is None and rule.max_value is None:
            return
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if rule.min_value is not None:
            below = (numeric < rule.min_value).sum()
            if below > 0:
                self.results.append(ValidationResult(
                    passed=False, rule="min_value", column=col,
                    message=f"Column '{col}': {below} values below min {rule.min_value}",
                    details={"violations": int(below), "min_allowed": rule.min_value}
                ))
            else:
                self.results.append(ValidationResult(passed=True, rule="min_value", column=col,
                                                      message=f"min_value check OK"))
        if rule.max_value is not None:
            above = (numeric > rule.max_value).sum()
            if above > 0:
                self.results.append(ValidationResult(
                    passed=False, rule="max_value", column=col,
                    message=f"Column '{col}': {above} values above max {rule.max_value}",
                    details={"violations": int(above), "max_allowed": rule.max_value}
                ))
            else:
                self.results.append(ValidationResult(passed=True, rule="max_value", column=col,
                                                      message=f"max_value check OK"))

    def _check_allowed_values(self, series: pd.Series, col: str, rule: ColumnRule):
        if rule.allowed_values is None:
            return
        invalid = ~series.dropna().isin(rule.allowed_values)
        count = invalid.sum()
        if count > 0:
            sample = series.dropna()[invalid].unique()[:5].tolist()
            self.results.append(ValidationResult(
                passed=False, rule="allowed_values", column=col,
                message=f"Column '{col}': {count} values not in allowed set",
                details={"invalid_samples": sample, "allowed": rule.allowed_values}
            ))
        else:
            self.results.append(ValidationResult(passed=True, rule="allowed_values", column=col,
                                                  message="allowed_values check OK"))

    def _check_regex(self, series: pd.Series, col: str, rule: ColumnRule):
        if rule.regex is None:
            return
        str_series = series.dropna().astype(str)
        mismatched = ~str_series.str.match(rule.regex)
        count = mismatched.sum()
        if count > 0:
            self.results.append(ValidationResult(
                passed=False, rule="regex", column=col,
                message=f"Column '{col}': {count} values don't match pattern '{rule.regex}'",
                details={"violations": int(count)}
            ))
        else:
            self.results.append(ValidationResult(passed=True, rule="regex", column=col,
                                                  message="regex check OK"))

    def _check_unique(self, series: pd.Series, col: str, rule: ColumnRule):
        if not rule.unique:
            return
        dupes = series.duplicated().sum()
        if dupes > 0:
            self.results.append(ValidationResult(
                passed=False, rule="unique", column=col,
                message=f"Column '{col}': {dupes} duplicate values found",
                details={"duplicate_count": int(dupes)}
            ))
        else:
            self.results.append(ValidationResult(passed=True, rule="unique", column=col,
                                                  message="uniqueness check OK"))
