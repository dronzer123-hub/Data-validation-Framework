"""
statistical_validator.py
Detects data drift, outliers, skewness, and distribution shifts
between a reference (training) dataset and incoming data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class StatConfig:
    z_score_threshold: float = 3.0          # Outlier threshold (std deviations)
    drift_threshold: float = 0.1            # Max allowed mean drift (fraction)
    missing_rate_threshold: float = 0.05    # Max allowed missing rate
    skewness_threshold: float = 2.0         # Max allowed absolute skewness
    variance_ratio_threshold: float = 2.0   # Max ratio of new/ref variance
    correlation_threshold: float = 0.1      # Max correlation drop allowed


@dataclass
class StatResult:
    passed: bool
    check: str
    column: Optional[str]
    message: str
    severity: str = "error"
    value: Optional[float] = None
    threshold: Optional[float] = None
    details: Optional[Dict] = None


class StatisticalValidator:
    """
    Runs statistical checks on incoming data against a reference distribution.

    Usage:
        # Fit on training/reference data
        validator = StatisticalValidator(config)
        validator.fit(reference_df)

        # Validate new data
        results = validator.validate(new_df)
    """

    def __init__(self, config: Optional[StatConfig] = None):
        self.config = config or StatConfig()
        self.reference_stats: Dict = {}
        self.results: List[StatResult] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None):
        """Compute reference statistics from training/baseline data."""
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols:
            s = df[col].dropna()
            self.reference_stats[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "median": float(s.median()),
                "q1": float(s.quantile(0.25)),
                "q3": float(s.quantile(0.75)),
                "missing_rate": float(df[col].isna().mean()),
                "skewness": float(s.skew()),
                "n": len(s),
            }
        self._fitted = True
        logger.info(f"StatisticalValidator fitted on {len(cols)} columns, {len(df)} rows")

    def validate(self, df: pd.DataFrame) -> List[StatResult]:
        """Validate incoming data. Returns list of StatResult objects."""
        self.results = []

        if not self._fitted:
            logger.warning("No reference data — running standalone checks only")
            self._standalone_checks(df)
            return self.results

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in self.reference_stats:
                continue
            ref = self.reference_stats[col]
            series = df[col]

            self._check_missing_rate(series, col, ref)
            self._check_mean_drift(series, col, ref)
            self._check_variance_ratio(series, col, ref)
            self._check_outliers(series, col)
            self._check_skewness(series, col)

        self._check_feature_correlations(df)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        logger.info(f"Statistical validation: {passed} passed, {failed} failed")
        return self.results

    def get_summary(self) -> Dict:
        """Return a summary dict of all checks."""
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "errors": sum(1 for r in self.results if not r.passed and r.severity == "error"),
            "warnings": sum(1 for r in self.results if not r.passed and r.severity == "warning"),
        }

    def is_valid(self) -> bool:
        return all(r.passed or r.severity != "error" for r in self.results)

    # ─── Internal Checks ──────────────────────────────────────────────────────

    def _standalone_checks(self, df: pd.DataFrame):
        """Run checks that don't require reference data."""
        for col in df.select_dtypes(include=[np.number]).columns:
            self._check_outliers(df[col], col)
            self._check_skewness(df[col], col)
            self._check_missing_rate(df[col], col, ref=None)

    def _check_missing_rate(self, series: pd.Series, col: str, ref: Optional[Dict]):
        rate = series.isna().mean()
        threshold = self.config.missing_rate_threshold
        if rate > threshold:
            self.results.append(StatResult(
                passed=False, check="missing_rate", column=col,
                message=f"'{col}': missing rate {rate:.2%} exceeds threshold {threshold:.2%}",
                severity="warning", value=round(rate, 4), threshold=threshold,
                details={"ref_missing_rate": ref.get("missing_rate") if ref else None}
            ))
        else:
            self.results.append(StatResult(passed=True, check="missing_rate", column=col,
                                            message=f"'{col}': missing rate {rate:.2%} OK", value=round(rate, 4)))

    def _check_mean_drift(self, series: pd.Series, col: str, ref: Dict):
        new_mean = series.dropna().mean()
        ref_mean = ref["mean"]
        if abs(ref_mean) < 1e-10:
            drift = abs(new_mean - ref_mean)
        else:
            drift = abs(new_mean - ref_mean) / abs(ref_mean)

        threshold = self.config.drift_threshold
        if drift > threshold:
            self.results.append(StatResult(
                passed=False, check="mean_drift", column=col,
                message=f"'{col}': mean drifted {drift:.2%} (threshold {threshold:.2%})",
                severity="error", value=round(drift, 4), threshold=threshold,
                details={"ref_mean": round(ref_mean, 4), "new_mean": round(float(new_mean), 4)}
            ))
        else:
            self.results.append(StatResult(passed=True, check="mean_drift", column=col,
                                            message=f"'{col}': mean drift {drift:.2%} OK", value=round(drift, 4)))

    def _check_variance_ratio(self, series: pd.Series, col: str, ref: Dict):
        new_std = series.dropna().std()
        ref_std = ref["std"]
        if ref_std < 1e-10 or new_std < 1e-10:
            return
        ratio = max(new_std / ref_std, ref_std / new_std)
        threshold = self.config.variance_ratio_threshold
        if ratio > threshold:
            self.results.append(StatResult(
                passed=False, check="variance_ratio", column=col,
                message=f"'{col}': variance ratio {ratio:.2f} exceeds {threshold:.2f}",
                severity="warning", value=round(ratio, 4), threshold=threshold,
                details={"ref_std": round(ref_std, 4), "new_std": round(float(new_std), 4)}
            ))
        else:
            self.results.append(StatResult(passed=True, check="variance_ratio", column=col,
                                            message=f"'{col}': variance ratio {ratio:.2f} OK", value=round(ratio, 4)))

    def _check_outliers(self, series: pd.Series, col: str):
        s = series.dropna()
        if len(s) < 3:
            return
        z = np.abs((s - s.mean()) / (s.std() + 1e-10))
        outlier_count = int((z > self.config.z_score_threshold).sum())
        outlier_pct = outlier_count / len(s)
        if outlier_pct > 0.01:  # >1% outliers is a warning
            self.results.append(StatResult(
                passed=False, check="outliers", column=col,
                message=f"'{col}': {outlier_count} outliers ({outlier_pct:.2%}) via z-score > {self.config.z_score_threshold}",
                severity="warning", value=round(outlier_pct, 4), threshold=0.01,
                details={"outlier_count": outlier_count, "z_threshold": self.config.z_score_threshold}
            ))
        else:
            self.results.append(StatResult(passed=True, check="outliers", column=col,
                                            message=f"'{col}': outlier rate {outlier_pct:.2%} OK"))

    def _check_skewness(self, series: pd.Series, col: str):
        s = series.dropna()
        if len(s) < 3:
            return
        skew = float(s.skew())
        threshold = self.config.skewness_threshold
        if abs(skew) > threshold:
            self.results.append(StatResult(
                passed=False, check="skewness", column=col,
                message=f"'{col}': skewness {skew:.2f} exceeds threshold ±{threshold}",
                severity="warning", value=round(skew, 4), threshold=threshold
            ))
        else:
            self.results.append(StatResult(passed=True, check="skewness", column=col,
                                            message=f"'{col}': skewness {skew:.2f} OK"))

    def _check_feature_correlations(self, df: pd.DataFrame):
        """Detects if new data correlation structure has changed significantly."""
        if not self._fitted or "correlations" not in self.reference_stats:
            return
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return
        new_corr = numeric.corr()
        ref_corr = pd.DataFrame(self.reference_stats["correlations"])
        common = [c for c in new_corr.columns if c in ref_corr.columns]
        if len(common) < 2:
            return
        diff = (new_corr[common].loc[common] - ref_corr[common].loc[common]).abs()
        max_diff = float(diff.values[~np.eye(len(common), dtype=bool)].max())
        if max_diff > self.config.correlation_threshold:
            self.results.append(StatResult(
                passed=False, check="correlation_shift", column=None,
                message=f"Max correlation shift {max_diff:.3f} exceeds threshold {self.config.correlation_threshold}",
                severity="warning", value=round(max_diff, 4),
                threshold=self.config.correlation_threshold
            ))
