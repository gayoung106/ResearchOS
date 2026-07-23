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
    suggestion: str = ""


@dataclass(slots=True)
class AutoRunValidationReport:
    passed: bool
    items: list[AutoRunValidationItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)




def _suggestion_for_item(item: str) -> str:
    if item.startswith("artifact:auto_rawdata_load_result"):
        return "\uc6d0\uc790\ub8cc \ud30c\uc77c\uc774 rawdata \ud3f4\ub354\uc5d0 \uc788\ub294\uc9c0, \ud30c\uc77c\uc774 \uc77d\uc744 \uc218 \uc788\ub294 CSV/Excel \ud615\uc2dd\uc778\uc9c0 \ud655\uc778\ud558\uc138\uc694."
    if item.startswith("artifact:auto_variable_map"):
        return "\ubcc0\uc218 \ucd94\ub860 \ub2e8\uacc4\uac00 \uc131\uacf5\ud588\ub294\uc9c0 \ud655\uc778\ud558\uace0, \ucf54\ub4dc\ubd81/\uc124\ubb38\uc9c0 \ub610\ub294 \uc218\ub3d9 \ubcc0\uc218 \ubcf4\uc815 \uc635\uc158\uc744 \ucd94\uac00\ud558\uc138\uc694."
    if item.startswith("artifact:auto_analysis_plan"):
        return "\uc885\uc18d\ubcc0\uc218\uc640 \ub3c5\ub9bd\ubcc0\uc218\uac00 \ud558\ub098 \uc774\uc0c1 \ucd94\ub860\ub418\ub3c4\ub85d \ubcc0\uc218\uba85, \ucf54\ub4dc\ubd81 role \ud78c\ud2b8, \uc218\ub3d9 \ubcf4\uc815 \uc635\uc158\uc744 \ud655\uc778\ud558\uc138\uc694."
    if item.startswith("artifact:auto_regression_pipeline_build_result"):
        return "\uc790\ub3d9 \ubd84\uc11d\uacc4\ud68d\uc774 \uc0dd\uc131\ub41c \ub4a4 Builder \ub4f1\ub85d \ub2e8\uacc4\uac00 \uc2e4\ud589\ub418\uc5c8\ub294\uc9c0 \ud655\uc778\ud558\uc138\uc694."
    if item.startswith("file:auto_run_report.md") or item.startswith("file:auto_final_report.md"):
        return "\uc790\ub3d9 \uc2e4\ud589\uc774 \uc911\uac04\uc5d0 \uc911\ub2e8\ub418\uc5c8\ub294\uc9c0 \ud655\uc778\ud558\uace0 result/00_auto_run \ud3f4\ub354 \uc4f0\uae30 \uad8c\ud55c\uc744 \ud655\uc778\ud558\uc138\uc694."
    if item.startswith("file:output_manifest.xlsx"):
        return "\uc0b0\ucd9c\ubb3c \ubaa9\ub85d\uc744 \uc800\uc7a5\ud558\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4. result/00_auto_run \ud3f4\ub354 \uc4f0\uae30 \uad8c\ud55c\uacfc Excel \uc800\uc7a5 \uc758\uc874\uc131\uc744 \ud655\uc778\ud558\uc138\uc694."
    if item.startswith("file:analysis_base.parquet"):
        return "\uc6d0\uc790\ub8cc \ub85c\ub529 \ub2e8\uacc4\uac00 \uc644\ub8cc\ub418\uc5c8\ub294\uc9c0 \ud655\uc778\ud558\uace0 pyarrow/parquet \uc800\uc7a5 \uc758\uc874\uc131\uc744 \ud655\uc778\ud558\uc138\uc694."
    if item.startswith("file:variable_role_inference") or item.startswith("file:inferred_variable_map"):
        return "\ubcc0\uc218 \ucd94\ub860 \ub2e8\uacc4 \ub85c\uadf8\uc640 variable_metadata.xlsx\ub97c \ud655\uc778\ud558\uc138\uc694."
    if item.startswith("file:auto_analysis_plan") or item.startswith("file:auto_variable_map"):
        return "\ubd84\uc11d\uacc4\ud68d \uc0dd\uc131 \ub2e8\uacc4\uac00 \uc131\uacf5\ud588\ub294\uc9c0 \ud655\uc778\ud558\uc138\uc694."
    if item.startswith("file:coefficients") or item.startswith("file:fit_statistics"):
        return "\ubaa8\ub378 \uc2e4\ud589 \ub2e8\uacc4\uac00 \uc131\uacf5\ud588\ub294\uc9c0 \ud655\uc778\ud558\uace0, --plan-only\ub85c \uacc4\ud68d\uc744 \uba3c\uc800 \uc810\uac80\ud558\uc138\uc694."
    if item.startswith("yaml:auto_analysis_plan"):
        return "auto_analysis_plan.yaml\uc5d0\uc11c dependent\uc640 regression.enabled \uac12\uc744 \ud655\uc778\ud558\uc138\uc694."
    if item.startswith("yaml:auto_variable_map"):
        return "auto_variable_map.yaml\uc774 \ube44\uc5b4 \uc788\uac70\ub098 \uc190\uc0c1\ub418\uc9c0 \uc54a\uc558\ub294\uc9c0 \ud655\uc778\ud558\uc138\uc694."
    return "\ud574\ub2f9 \ub2e8\uacc4\uc758 \uc774\uc804 \uc0b0\ucd9c\ubb3c\uacfc \uacbd\uace0 \uba54\uc2dc\uc9c0\ub97c \ud655\uc778\ud558\uc138\uc694."

def _artifact_exists(runtime: PipelineRuntime, key: str) -> bool:
    try:
        runtime.get_artifact(key)
    except KeyError:
        return False
    return True


def _add_item(
    items: list[AutoRunValidationItem],
    item: str,
    passed: bool,
    evidence: str,
    suggestion: str | None = None,
) -> None:
    items.append(
        AutoRunValidationItem(
            item=item,
            passed=passed,
            evidence=evidence,
            suggestion="" if passed else (suggestion or _suggestion_for_item(item)),
        )
    )


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
        "auto_final_report.md",
        "output_manifest.xlsx",
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
