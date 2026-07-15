"""
ci_validation.py
CI/CD integration script for ML Data Validation Framework.
Produces: "Validation passed ✓" on clean data
          Exits with code 1 (non-zero) when validation fails in strict mode

Usage:
    python ci_validation.py --data clean       → Validation passed ✓  (exit code 0)
    python ci_validation.py --data corrupt     → Validation FAILED    (exit code 1)
    python ci_validation.py --data drifted     → Validation FAILED    (exit code 1)
    python ci_validation.py                    → runs all three modes
"""
import sys
import os
import argparse
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

from data_validator.schema_validator import SchemaConfig, ColumnRule
from data_validator.statistical_validator import StatConfig
from data_validator.pipeline_validator import PipelineValidator, PipelineConfig, ValidationError


# ─── ANSI colors (work in PowerShell 7+, Linux, Mac) ─────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def green(t):  return f"{GREEN}{t}{RESET}"
def red(t):    return f"{RED}{t}{RESET}"
def yellow(t): return f"{YELLOW}{t}{RESET}"
def cyan(t):   return f"{CYAN}{t}{RESET}"
def bold(t):   return f"{BOLD}{t}{RESET}"


# ─── Dataset factory ──────────────────────────────────────────────────────────

def make_clean_data(n=300, seed=42):
    np.random.seed(seed)
    return pd.DataFrame({
        "age":            np.random.normal(35, 10, n).clip(18, 75),
        "income":         np.random.normal(55000, 15000, n).clip(10000, 150000),
        "credit_score":   np.random.normal(680, 60, n).clip(300, 850),
        "loan_amount":    np.random.normal(15000, 5000, n).clip(1000, 50000),
        "years_employed": np.random.normal(7, 3, n).clip(0, 40),
    })

def make_corrupt_data(n=300, seed=42):
    df = make_clean_data(n, seed)
    df.loc[0:4,  "age"]          = np.nan       # Nulls in non-nullable column
    df.loc[5:7,  "credit_score"] = -999          # Below allowed minimum
    df.loc[8,    "income"]       = -50000        # Negative income
    df.loc[9:11, "loan_amount"]  = np.nan        # Null loan amounts
    return df

def make_drifted_data(n=300, seed=99):
    df = make_clean_data(n, seed)
    df["income"]       = df["income"]       * 3.5   # Massive income drift
    df["credit_score"] = df["credit_score"] * 0.5   # Credit scores collapsed
    df["age"]          = df["age"]          + 25    # Much older population
    return df

def make_reference_data(n=500, seed=0):
    return make_clean_data(n, seed)


# ─── Validation config ────────────────────────────────────────────────────────

def get_schema():
    return SchemaConfig(
        columns={
            "age":            ColumnRule(nullable=False, min_value=18,  max_value=100),
            "income":         ColumnRule(nullable=False, min_value=0),
            "credit_score":   ColumnRule(nullable=True,  min_value=300, max_value=850),
            "loan_amount":    ColumnRule(nullable=False, min_value=0),
            "years_employed": ColumnRule(nullable=False, min_value=0,   max_value=50),
        },
        min_rows=10,
    )

def get_stat():
    return StatConfig(
        drift_threshold=0.15,
        missing_rate_threshold=0.05,
        z_score_threshold=3.0,
        skewness_threshold=2.5,
        variance_ratio_threshold=2.0,
    )


# ─── Core CI validation function ─────────────────────────────────────────────

def run_ci_validation(data: pd.DataFrame,
                      reference: pd.DataFrame,
                      label: str,
                      strict: bool = True) -> int:
    """
    Runs validation. Returns exit code: 0 = pass, 1 = fail.
    This is what your CI/CD pipeline calls.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print()
    print(bold(f"┌{'─'*56}┐"))
    print(bold(f"│  ML Data Validation — CI/CD Check{' '*21}│"))
    print(bold(f"│  {timestamp}{' '*20}│"))
    print(bold(f"└{'─'*56}┘"))
    print()
    print(f"  {cyan('Dataset')}       : {label}")
    print(f"  {cyan('Rows')}          : {len(data)}")
    print(f"  {cyan('Columns')}       : {list(data.columns)}")
    print(f"  {cyan('Strict mode')}   : {str(strict)}")
    print(f"  {cyan('Reference rows')}: {len(reference)}")
    print()
    print(f"  Running checks", end="", flush=True)

    config = PipelineConfig(
        schema_config=get_schema(),
        stat_config=get_stat(),
        strict_mode=False,          # We handle exit code ourselves
        save_reports=True,
        report_dir="reports",
        log_level="WARNING",        # Suppress info logs in CI output
    )

    validator = PipelineValidator(config)
    validator.fit(reference)        # Fit on reference/training data

    # Animate dots while validating
    for _ in range(3):
        time.sleep(0.2)
        print(".", end="", flush=True)
    print()

    try:
        validator.transform(data)
    except ValidationError:
        pass  # We read the report ourselves

    report  = validator.get_last_report()
    summary = report["summary"]
    errors  = report.get("errors",   [])
    warnings= report.get("warnings", [])

    # ── Print detailed results ─────────────────────────────────────────────
    print()
    print(f"  {'─'*54}")
    print(f"  {'CHECK':<30} {'RESULT'}")
    print(f"  {'─'*54}")

    for r in report.get("all_results", []):
        icon   = green("✓") if r["passed"] else (red("✗") if r["severity"] == "error" else yellow("⚠"))
        col    = f"[{r['column']}]" if r["column"] else "[global]"
        status = green("PASS") if r["passed"] else (red("FAIL") if r["severity"] == "error" else yellow("WARN"))
        label_str = f"{r['check']} {col}"
        print(f"  {icon}  {label_str:<35} {status}")

    print(f"  {'─'*54}")
    print()

    # ── Summary block ─────────────────────────────────────────────────────
    print(f"  {bold('SUMMARY')}")
    print(f"  Total checks  : {summary['total_checks']}")
    print(f"  Passed        : {green(summary['passed'])}")
    print(f"  Errors        : {red(summary['errors']) if summary['errors'] else green('0')}")
    print(f"  Warnings      : {yellow(summary['warnings']) if summary['warnings'] else green('0')}")
    print()

    if errors:
        print(f"  {red(bold('ERRORS FOUND:'))}")
        for e in errors:
            print(f"    {red('✗')} [{e['column']}] {e['message']}")
        print()

    if warnings:
        print(f"  {yellow(bold('WARNINGS:'))}")
        for w in warnings:
            print(f"    {yellow('⚠')}  [{w['column']}] {w['message']}")
        print()

    # ── Final verdict — this is the key output line ────────────────────────
    print(f"  {'─'*54}")

    if summary["valid"]:
        print(f"  {bold(green('Validation passed ✓'))}")
        print(f"  {'─'*54}")
        print(f"  {green('Pipeline can proceed. Data is clean.')}")
        print(f"  Exit code : {green('0')}")
        exit_code = 0
    else:
        print(f"  {bold(red('Validation FAILED ✗'))}")
        print(f"  {'─'*54}")
        if strict:
            print(f"  {red('Strict mode ON  → CI pipeline will FAIL.')}")
            print(f"  {red('Deployment blocked. Fix data issues before retrying.')}")
        else:
            print(f"  {yellow('Strict mode OFF → Pipeline continues with warnings.')}")
        print(f"  Exit code : {red('1') if strict else yellow('0 (warnings only)')}")
        exit_code = 1 if strict else 0

    print()

    # Save JSON summary for CI artifact
    ci_summary = {
        "timestamp": timestamp,
        "dataset":   label,
        "valid":     summary["valid"],
        "exit_code": exit_code,
        "summary":   summary,
        "errors":    errors,
        "warnings":  warnings,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/ci_result.json", "w") as f:
        json.dump(ci_summary, f, indent=2)
    print(f"  {cyan('Report saved')} → reports/ci_result.json")
    print(f"  {cyan('HTML report')}  → reports/validation_*.html")
    print()

    return exit_code


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CI/CD Data Validation Script")
    parser.add_argument(
        "--data",
        choices=["clean", "corrupt", "drifted", "all"],
        default="all",
        help="Which dataset to validate (default: all)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Exit with code 1 on validation failure (default: True)"
    )
    args = parser.parse_args()

    reference = make_reference_data()

    if args.data == "clean" or args.data == "all":
        print(f"\n{bold(cyan('━'*60))}")
        print(bold(cyan("  TEST: Clean Data (should pass)")))
        print(bold(cyan('━'*60)))
        code = run_ci_validation(make_clean_data(), reference, "clean_dataset.csv", strict=True)
        if args.data == "clean":
            sys.exit(code)

    if args.data == "corrupt" or args.data == "all":
        print(f"\n{bold(cyan('━'*60))}")
        print(bold(cyan("  TEST: Corrupt Data (should fail with exit code 1)")))
        print(bold(cyan('━'*60)))
        code = run_ci_validation(make_corrupt_data(), reference, "corrupt_dataset.csv", strict=True)
        if args.data == "corrupt":
            sys.exit(code)

    if args.data == "drifted" or args.data == "all":
        print(f"\n{bold(cyan('━'*60))}")
        print(bold(cyan("  TEST: Drifted Data (should fail with exit code 1)")))
        print(bold(cyan('━'*60)))
        code = run_ci_validation(make_drifted_data(), reference, "drifted_dataset.csv", strict=True)
        if args.data == "drifted":
            sys.exit(code)

    if args.data == "all":
        print(bold(cyan('━'*60)))
        print(bold(cyan("  ALL CI TESTS COMPLETE")))
        print(bold(cyan('━'*60)))
        print(f"""
  What just happened:
    clean   → {green('exit code 0')}  → CI pipeline continues ✓
    corrupt → {red('exit code 1')}  → CI pipeline FAILS, deployment blocked ✗
    drifted → {red('exit code 1')}  → CI pipeline FAILS, deployment blocked ✗

  In a real GitHub Actions / Jenkins / GitLab CI pipeline,
  exit code 1 marks the job as FAILED and stops the deployment.
        """)


if __name__ == "__main__":
    main()
