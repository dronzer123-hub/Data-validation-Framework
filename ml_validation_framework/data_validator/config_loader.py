"""
config_loader.py
Load SchemaConfig from a YAML file so rules can be stored outside Python code.
"""
import yaml
from pathlib import Path
from data_validator.schema_validator import SchemaConfig, ColumnRule
from data_validator.statistical_validator import StatConfig
from data_validator.pipeline_validator import PipelineConfig


def load_schema_config(path: str) -> SchemaConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    columns = {}
    for col, rules in raw.get("columns", {}).items():
        columns[col] = ColumnRule(
            dtype=rules.get("dtype"),
            nullable=rules.get("nullable", True),
            min_value=rules.get("min_value"),
            max_value=rules.get("max_value"),
            allowed_values=rules.get("allowed_values"),
            regex=rules.get("regex"),
            unique=rules.get("unique", False),
        )

    return SchemaConfig(
        columns=columns,
        min_rows=raw.get("min_rows"),
        max_rows=raw.get("max_rows"),
        allow_extra_columns=raw.get("allow_extra_columns", True),
        require_all_columns=raw.get("require_all_columns", True),
    )


def load_stat_config(path: str) -> StatConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    stat = raw.get("statistical", {})
    return StatConfig(
        z_score_threshold=stat.get("z_score_threshold", 3.0),
        drift_threshold=stat.get("drift_threshold", 0.1),
        missing_rate_threshold=stat.get("missing_rate_threshold", 0.05),
        skewness_threshold=stat.get("skewness_threshold", 2.0),
        variance_ratio_threshold=stat.get("variance_ratio_threshold", 2.0),
    )


def load_pipeline_config(schema_yaml: str, stat_yaml: Optional[str] = None) -> PipelineConfig:
    from typing import Optional
    schema = load_schema_config(schema_yaml)
    stat = load_stat_config(stat_yaml) if stat_yaml else StatConfig()
    return PipelineConfig(schema_config=schema, stat_config=stat)
