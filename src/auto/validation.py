"""Validation helpers for automatic rawdata analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common.config_loader import load_analysis_plan, load_variable_map
from src.pipeline.runtime import PipelineRuntime


@dataclass(slots=True)
class AutoRunValidationItem:
    item: str
    passed: bool
    evidence: str


@dataclass(slots=True)
class AutoRunValidationReport:
    passed: bool
    items: list[AutoRunValidationItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _artifact_exists(runtime: PipelineRuntime, key: str) -> bool:
    try:
        runtime.get_artifact(key)
    except KeyError:
        return False
    return True


def _add_item(items: list[AutoRunValidationItem], item: str, passed: bool, evidence: str) -> None:
    items.append(AutoRunValidationItem(item=item, passed=passed, evidence=evidence))


def validate_auto_run_outputs(
    *,
    runtime: PipelineRuntime,
    output_files: list[str],
    require_model_outputs: bool = False,
) -> AutoRunValidationReport:
    items: list[AutoRunValidationItem] = []
    output_paths = [Path(path) for path in output_files]

    for key in [
        "auto_rawdata_load_result",
        "auto_variable_map",
        "auto_analysis_plan",
        "auto_regression_pipeline_build_result",
    ]:
        _add_item(
            items,
            f"artifact:{key}",
            _artifact_exists(runtime, key),
            "runtime artifact present" if _artifact_exists(runtime, key) else "runtime artifact missing",
        )

    required_files = {
        "auto_run_summary.xlsx",
        "auto_run_report.md",
        "analysis_base.parquet",
        "variable_role_inference.xlsx",
        "inferred_variable_map.xlsx",
        "analysis_plan_summary.xlsx",
        "auto_analysis_plan.yaml",
        "auto_variable_map.yaml",
    }
    if require_model_outputs:
        required_files.update(
            {
                "coefficients.xlsx",
                "fit_statistics.xlsx",
            }
        )

    for filename in sorted(required_files):
        matching_paths = [path for path in output_paths if path.name == filename]
        exists = bool(matching_paths) and all(path.exists() for path in matching_paths)
        _add_item(
            items,
            f"file:{filename}",
            exists,
            str(matching_paths[0]) if matching_paths else "output file not listed",
        )

    analysis_plan_paths = [path for path in output_paths if path.name == "auto_analysis_plan.yaml"]
    if analysis_plan_paths:
        try:
            plan = load_analysis_plan(analysis_plan_paths[0])
            _add_item(
                items,
                "yaml:auto_analysis_plan",
                bool(plan.variables.dependent) and plan.analyses.regression.enabled,
                "analysis plan loaded and regression is enabled",
            )
        except Exception as error:  # noqa: BLE001 - report validation problem without failing caller.
            _add_item(items, "yaml:auto_analysis_plan", False, str(error))

    variable_map_paths = [path for path in output_paths if path.name == "auto_variable_map.yaml"]
    if variable_map_paths:
        try:
            variable_map = load_variable_map(variable_map_paths[0])
            _add_item(
                items,
                "yaml:auto_variable_map",
                bool(variable_map.variables),
                f"{len(variable_map.variables)} variables loaded",
            )
        except Exception as error:  # noqa: BLE001 - report validation problem without failing caller.
            _add_item(items, "yaml:auto_variable_map", False, str(error))

    warnings = [f"{item.item}: {item.evidence}" for item in items if not item.passed]
    return AutoRunValidationReport(
        passed=not warnings,
        items=items,
        warnings=warnings,
    )
