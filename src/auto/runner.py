"""One-call automatic rawdata analysis runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

from src.auto.analysis_plan import AutoAnalysisPlanStep
from src.auto.multi_outcome import AutoMultiOutcomeAnalysisPlanStep
from src.auto.overrides import (
    apply_variable_role_overrides,
    build_auto_variable_role_overrides,
)
from src.auto.pipeline import (
    AutoMultiOutcomePipelineBuildResult,
    AutoMultiOutcomePipelineRunResult,
    AutoRegressionPipelineBuildResult,
    build_auto_multi_outcome_regression_orchestrators,
    build_auto_regression_orchestrator,
    run_auto_multi_outcome_regression_orchestrators,
)
from src.auto.rawdata_loader import AutoRawDataLoadingStep
from src.auto.validation import auto_run_validation_report_to_dataframe, validate_auto_run_outputs
from src.auto.variable_inference import AutoVariableInferenceStep, variable_map_to_dataframe
from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import OrchestratorResult, ResearchOrchestrator
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import StepResult


@dataclass(slots=True)
class AutoRawDataAnalysisResult:
    success: bool
    context: ResearchContext
    runtime: PipelineRuntime
    setup_step_results: list[StepResult] = field(default_factory=list)
    pipeline_build_result: AutoRegressionPipelineBuildResult | None = None
    multi_outcome_pipeline_build_result: AutoMultiOutcomePipelineBuildResult | None = None
    multi_outcome_pipeline_run_result: AutoMultiOutcomePipelineRunResult | None = None
    orchestrator: ResearchOrchestrator | None = None
    orchestrator_result: OrchestratorResult | None = None
    output_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failed_stage: str | None = None


def _record_step_result(
    *,
    context: ResearchContext,
    result: StepResult,
    output_files: list[str],
    warnings: list[str],
) -> None:
    for path in result.output_files:
        context.add_generated_file(path)
        output_files.append(path)
    for warning in result.warnings:
        context.add_warning(f"{result.stage_name}: {warning}")
        warnings.append(warning)
    if result.success:
        context.mark_stage_completed(result.stage_name)


def _apply_plan_to_context(context: ResearchContext, analysis_plan: AnalysisPlan, variable_map: VariableMap) -> None:
    context.dependent_variables = list(analysis_plan.variables.dependent)
    context.independent_variables = list(analysis_plan.variables.independent)
    context.mediator_variables = list(analysis_plan.variables.mediators)
    context.moderator_variables = list(analysis_plan.variables.moderators)
    context.control_variables = list(analysis_plan.variables.controls)
    context.analysis_plan = analysis_plan.model_dump(mode="json")
    context.variable_map = variable_map.model_dump(mode="json")


def _write_auto_run_summary(
    *,
    working_directory: Path,
    result: AutoRawDataAnalysisResult,
) -> str:
    import pandas as pd

    output_dir = working_directory / "result" / "00_auto_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "auto_run_summary.xlsx"
    rows: list[dict[str, object]] = []
    for step_result in result.setup_step_results:
        rows.append(
            {
                "stage_name": step_result.stage_name,
                "success": step_result.success,
                "output_file_count": len(step_result.output_files),
                "warning_count": len(step_result.warnings),
            }
        )
    if result.pipeline_build_result is not None:
        rows.append(
            {
                "stage_name": "04_auto_pipeline_registration",
                "success": result.pipeline_build_result.success,
                "output_file_count": 0,
                "warning_count": len(result.pipeline_build_result.warnings),
            }
        )
    if result.orchestrator_result is not None:
        rows.append(
            {
                "stage_name": "05_auto_pipeline_execution",
                "success": result.orchestrator_result.success,
                "output_file_count": len(result.context.generated_files),
                "warning_count": len(result.orchestrator_result.warnings),
            }
        )
    if result.multi_outcome_pipeline_build_result is not None:
        rows.append(
            {
                "stage_name": "04b_auto_multi_outcome_pipeline_registration",
                "success": result.multi_outcome_pipeline_build_result.success,
                "output_file_count": 0,
                "warning_count": len(result.multi_outcome_pipeline_build_result.warnings),
            }
        )
    if result.multi_outcome_pipeline_run_result is not None:
        rows.append(
            {
                "stage_name": "05b_auto_multi_outcome_pipeline_execution",
                "success": result.multi_outcome_pipeline_run_result.success,
                "output_file_count": len(result.multi_outcome_pipeline_run_result.completed_models),
                "warning_count": len(result.multi_outcome_pipeline_run_result.warnings),
            }
        )
    pd.DataFrame(rows).to_excel(summary_path, index=False)
    return str(summary_path)


def _format_bool(value: bool) -> str:
    return "\uc131\uacf5" if value else "\uc2e4\ud328"


def _artifact_or_none(runtime: PipelineRuntime, key: str) -> object | None:
    try:
        return runtime.get_artifact(key)
    except KeyError:
        return None


def _write_auto_run_markdown(
    *,
    working_directory: Path,
    result: AutoRawDataAnalysisResult,
) -> str:
    output_dir = working_directory / "result" / "00_auto_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "auto_run_report.md"

    rawdata = _artifact_or_none(result.runtime, "auto_rawdata_load_result")
    analysis_plan = _artifact_or_none(result.runtime, "auto_analysis_plan")
    variable_map = _artifact_or_none(result.runtime, "auto_variable_map")
    registration = result.pipeline_build_result.registration if result.pipeline_build_result else None

    lines = [
        "# \uc790\ub3d9 \ubd84\uc11d \uc2e4\ud589 \uc694\uc57d",
        "",
        f"- \ud504\ub85c\uc81d\ud2b8\uba85: {result.context.project_name}",
        f"- \uc804\uccb4 \uc0c1\ud0dc: {_format_bool(result.success)}",
        f"- \uc2e4\ud328 \ub2e8\uacc4: {result.failed_stage or '-'}",
        f"- \uacbd\uace0 \uc218: {len(result.warnings)}",
        "",
        "## \uc6d0\uc790\ub8cc \uc120\ud0dd",
    ]
    if rawdata is not None:
        candidate = rawdata.selected_candidate
        lines.extend(
            [
                f"- \ud30c\uc77c: `{candidate.source_path}`",
                f"- Sheet: {candidate.sheet_name or '-'}",
                f"- \ud589 \uc218: {candidate.row_count}",
                f"- \uc5f4 \uc218: {candidate.column_count}",
                f"- \ud6c4\ubcf4 \uc218: {len(rawdata.candidates)}",
            ]
        )
    else:
        lines.append("- \uc6d0\uc790\ub8cc\ub97c \ubd88\ub7ec\uc624\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4.")

    lines.extend(["", "## \uc790\ub3d9 \ubcc0\uc218 \uc5ed\ud560"])
    if isinstance(variable_map, VariableMap):
        role_rows = [
            (name, definition.role, definition.measurement_level)
            for name, definition in variable_map.variables.items()
        ]
        lines.extend(["| \ubcc0\uc218 | \uc5ed\ud560 | \uce21\uc815\uc218\uc900 |", "| --- | --- | --- |"])
        lines.extend(f"| `{name}` | {role} | {level} |" for name, role, level in role_rows)
    else:
        lines.append("- \ubcc0\uc218 \uc5ed\ud560\uc744 \ucd94\ub860\ud558\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4.")

    lines.extend(["", "## \uc790\ub3d9 \ubd84\uc11d\uacc4\ud68d"])
    if isinstance(analysis_plan, AnalysisPlan):
        lines.extend(
            [
                f"- \uc885\uc18d\ubcc0\uc218: {', '.join(analysis_plan.variables.dependent) or '-'}",
                f"- \ub3c5\ub9bd\ubcc0\uc218: {', '.join(analysis_plan.variables.independent) or '-'}",
                f"- \ud1b5\uc81c\ubcc0\uc218: {', '.join(analysis_plan.variables.controls) or '-'}",
                f"- \uad70\uc9d1\ubcc0\uc218: {', '.join(analysis_plan.variables.clusters) or '-'}",
                f"- \uac00\uc911\uce58 \ubcc0\uc218: {', '.join(analysis_plan.variables.weights) or '-'}",
                f"- \ud68c\uadc0\ubd84\uc11d \ud65c\uc131\ud654: {_format_bool(analysis_plan.analyses.regression.enabled)}",
                f"- Panel \ubd84\uc11d \ud65c\uc131\ud654: {_format_bool(analysis_plan.analyses.panel.enabled)}",
                f"- \uac15\uac74\uc131 \ubd84\uc11d \ud65c\uc131\ud654: {_format_bool(analysis_plan.analyses.robustness.enabled)}",
                f"- \ud68c\uadc0 \uc635\uc158: `{analysis_plan.analyses.regression.options}`",
            ]
        )
    else:
        lines.append("- \ubd84\uc11d\uacc4\ud68d\uc744 \uc0dd\uc131\ud558\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4.")

    lines.extend(["", "## \ub4f1\ub85d\ub41c \ubaa8\ub378"])
    if registration is not None:
        lines.extend(
            [
                f"- \ubaa8\ub378 ID: {registration.model_id}",
                f"- \ubaa8\ub378 \uc720\ud615: {registration.model_type}",
                f"- \uc885\uc18d\ubcc0\uc218: {registration.dependent_variable}",
                f"- \ub3c5\ub9bd\ubcc0\uc218: {', '.join(registration.independent_variables)}",
                f"- \uc9c4\ub2e8 \ub4f1\ub85d: {_format_bool(registration.diagnostics_registered)}",
                f"- \ud6a8\uacfc\ud06c\uae30 \ub4f1\ub85d: {_format_bool(registration.effect_size_registered)}",
                f"- \ubcf4\uace0 \ub4f1\ub85d: {_format_bool(registration.reporting_registered)}",
                f"- \uc2dc\uac01\ud654 \ub4f1\ub85d: {_format_bool(registration.visualization_registered)}",
                f"- Audit \ub4f1\ub85d: {_format_bool(registration.audit_registered)}",
            ]
        )
    else:
        lines.append("- \ubaa8\ub378 \ud30c\uc774\ud504\ub77c\uc778\uc774 \ub4f1\ub85d\ub418\uc9c0 \uc54a\uc558\uc2b5\ub2c8\ub2e4.")

    lines.extend(["", "## \ub2e8\uacc4\ubcc4 \uacb0\uacfc", "| \ub2e8\uacc4 | \uc0c1\ud0dc | \uc0b0\ucd9c\ubb3c \uc218 | \uacbd\uace0 \uc218 |", "| --- | --- | ---: | ---: |"])
    for step_result in result.setup_step_results:
        lines.append(
            f"| {step_result.stage_name} | {_format_bool(step_result.success)} | "
            f"{len(step_result.output_files)} | {len(step_result.warnings)} |"
        )
    if result.pipeline_build_result is not None:
        lines.append(
            "| 04_auto_pipeline_registration | "
            f"{_format_bool(result.pipeline_build_result.success)} | 0 | "
            f"{len(result.pipeline_build_result.warnings)} |"
        )
    if result.orchestrator_result is not None:
        lines.append(
            "| 05_auto_pipeline_execution | "
            f"{_format_bool(result.orchestrator_result.success)} | "
            f"{len(result.context.generated_files)} | {len(result.orchestrator_result.warnings)} |"
        )

    if result.warnings:
        lines.extend(["", "## \uacbd\uace0"])
        lines.extend(f"- {warning}" for warning in result.warnings)

    lines.extend(["", "## \uc8fc\uc694 \uc0b0\ucd9c\ubb3c"])
    lines.extend(f"- `{output_file}`" for output_file in result.output_files)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)



def _format_number(value: object) -> str:
    if isinstance(value, Real):
        return f"{value:.3f}"
    return str(value)

def _sort_p_value(value: object) -> float:
    if value is None:
        return 1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _artifact_keys(runtime: PipelineRuntime, prefix: str) -> list[str]:
    return sorted(key for key in runtime.artifacts if key.startswith(prefix))


def _append_model_result_summary(lines: list[str], runtime: PipelineRuntime, *, title: str) -> None:
    result_keys = _artifact_keys(runtime, "regression_result:")
    if not result_keys:
        return
    lines.extend(["", f"## {title}"])
    for key in result_keys:
        model_id = key.split(":", 1)[1]
        regression = runtime.artifacts.get(key)
        publication = runtime.artifacts.get(f"regression_publication_report:{model_id}")
        effects = runtime.artifacts.get(f"effect_size_report:{model_id}")
        lines.extend(
            [
                "",
                f"### {model_id}",
                f"- model_type: {getattr(regression, 'model_type', '-')}",
                f"- dependent: {getattr(regression, 'dependent_variable', '-')}",
                f"- sample_size: {getattr(regression, 'sample_size', '-')}",
                f"- converged: {_format_bool(bool(getattr(regression, 'converged', False)))}",
            ]
        )
        fit_statistics = getattr(regression, "fit_statistics", {}) or {}
        important_stats = [
            "r_squared",
            "adjusted_r_squared",
            "pseudo_r_squared_mcfadden",
            "pseudo_r_squared_deviance",
            "aic",
            "bic",
            "log_likelihood",
            "dispersion_ratio",
            "group_count",
            "cluster_count",
        ]
        statistic_rows = [
            (name, fit_statistics.get(name))
            for name in important_stats
            if fit_statistics.get(name) is not None
        ]
        if statistic_rows:
            lines.extend(["", "| statistic | value |", "| --- | ---: |"])
            lines.extend(f"| {name} | {_format_number(value)} |" for name, value in statistic_rows)

        coefficients = [
            coefficient
            for coefficient in getattr(regression, "coefficients", [])
            if str(getattr(coefficient, "term", "")).lower() not in {"const", "intercept"}
        ]
        if coefficients:
            top_coefficients = sorted(coefficients, key=lambda item: _sort_p_value(getattr(item, "p_value", None)))[:5]
            lines.extend(["", "| term | estimate | p_value |", "| --- | ---: | ---: |"])
            lines.extend(
                f"| {coefficient.term} | {_format_number(coefficient.estimate)} | "
                f"{_format_number(coefficient.p_value)} |"
                for coefficient in top_coefficients
            )

        effect_items = getattr(effects, "effects", []) if effects is not None else []
        if effect_items:
            top_effects = sorted(
                effect_items,
                key=lambda item: _sort_p_value(getattr(item, "p_value", None)),
            )[:5]
            lines.extend(["", "| effect term | effect_type | estimate | magnitude |", "| --- | --- | ---: | --- |"])
            lines.extend(
                f"| {effect.term} | {effect.effect_type} | {_format_number(effect.estimate)} | "
                f"{effect.magnitude or '-'} |"
                for effect in top_effects
            )

        narrative = getattr(publication, "narrative", "") if publication is not None else ""
        if narrative:
            lines.extend(["", "Narrative:", narrative])
        warnings = list(getattr(regression, "warnings", []) or [])
        if warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in warnings[:5])



def _classify_output_file(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "00_auto_run" in parts:
        return "auto_run"
    if "01_auto_import" in parts:
        return "rawdata_import"
    if "02_auto_variables" in parts:
        return "variable_inference"
    if "03_auto_plan" in parts:
        return "analysis_plan"
    if "09_regression_analysis" in parts or name in {"coefficients.xlsx", "fit_statistics.xlsx"}:
        return "regression_result"
    if "10_diagnostics" in parts:
        return "diagnostics"
    if "13_effect_size" in parts:
        return "effect_size"
    if "14_regression_reporting" in parts:
        return "reporting"
    if "15_regression_visualization" in parts:
        return "visualization"
    if "16_research_audit" in parts:
        return "research_audit"
    if "multi_outcome_runs" in parts:
        return "multi_outcome"
    return "other"


def _output_manifest_priority(category: str) -> int:
    order = {
        "auto_run": 0,
        "rawdata_import": 1,
        "variable_inference": 2,
        "analysis_plan": 3,
        "regression_result": 4,
        "diagnostics": 5,
        "effect_size": 6,
        "reporting": 7,
        "visualization": 8,
        "research_audit": 9,
        "multi_outcome": 10,
        "other": 99,
    }
    return order.get(category, 99)

def _describe_output_file(path: Path, category: str) -> str:
    name = path.name.lower()
    descriptions = {
        "auto_final_report.md": "Start here: final narrative summary and key model results.",
        "auto_run_report.md": "Automatic setup and pipeline registration summary.",
        "auto_run_summary.xlsx": "Stage-level execution status table.",
        "auto_validation_report.xlsx": "Validation checklist with evidence and repair suggestions.",
        "output_manifest.xlsx": "Index of generated output files and recommended reading order.",
        "analysis_base.parquet": "Clean analysis dataset selected from rawdata.",
        "rawdata_candidates.xlsx": "Candidate raw datasets and sheet selection scores.",
        "variable_role_inference.xlsx": "Variable role and measurement-level inference details.",
        "inferred_variable_map.xlsx": "Inferred variable map in spreadsheet form.",
        "analysis_plan_summary.xlsx": "Auto-generated analysis plan summary.",
        "auto_analysis_plan.yaml": "Machine-readable analysis plan used by the pipeline.",
        "auto_variable_map.yaml": "Machine-readable variable map used by the pipeline.",
        "coefficients.xlsx": "Model coefficient table.",
        "fit_statistics.xlsx": "Model fit statistics table.",
    }
    if name in descriptions:
        return descriptions[name]
    category_descriptions = {
        "diagnostics": "Model diagnostic output.",
        "effect_size": "Effect-size output.",
        "reporting": "Publication reporting output.",
        "visualization": "Generated visualization output.",
        "research_audit": "Research audit output.",
        "multi_outcome": "Output from an automatically generated multi-outcome model.",
    }
    return category_descriptions.get(category, "Generated analysis output.")


def _is_recommended_output(path: Path) -> bool:
    return path.name.lower() in {
        "auto_final_report.md",
        "auto_validation_report.xlsx",
        "output_manifest.xlsx",
        "coefficients.xlsx",
        "fit_statistics.xlsx",
    }



def _write_auto_validation_report(
    *,
    working_directory: Path,
    validation_report: object,
) -> str:
    output_dir = working_directory / "result" / "00_auto_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "auto_validation_report.xlsx"
    auto_run_validation_report_to_dataframe(validation_report).to_excel(report_path, index=False)
    return str(report_path)

def _write_output_manifest(*, working_directory: Path, result: AutoRawDataAnalysisResult) -> str:
    import pandas as pd

    output_dir = working_directory / "result" / "00_auto_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "output_manifest.xlsx"
    output_paths = list(dict.fromkeys([*result.output_files, str(manifest_path)]))
    rows: list[dict[str, object]] = []
    for output_file in output_paths:
        path = Path(output_file)
        category = _classify_output_file(path)
        try:
            relative_path = path.resolve().relative_to(working_directory).as_posix()
        except ValueError:
            relative_path = str(path)
        exists = path.exists()
        rows.append(
            {
                "category": category,
                "recommended_order": _output_manifest_priority(category),
                "recommended": _is_recommended_output(path),
                "description": _describe_output_file(path, category),
                "filename": path.name,
                "relative_path": relative_path,
                "absolute_path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
            }
        )
    rows.sort(key=lambda row: (int(row["recommended_order"]), str(row["relative_path"])))
    pd.DataFrame(rows).to_excel(manifest_path, index=False)
    return str(manifest_path)

def _relative_output_path(path: Path, working_directory: Path) -> str:
    try:
        return path.resolve().relative_to(working_directory).as_posix()
    except ValueError:
        return str(path)


def _append_recommended_outputs(
    lines: list[str],
    *,
    working_directory: Path,
    result: AutoRawDataAnalysisResult,
) -> None:
    rows: list[tuple[int, str, str, str, str]] = []
    for output_file in dict.fromkeys(result.output_files):
        path = Path(output_file)
        if not _is_recommended_output(path):
            continue
        category = _classify_output_file(path)
        rows.append(
            (
                _output_manifest_priority(category),
                _relative_output_path(path, working_directory),
                path.name,
                category,
                _describe_output_file(path, category),
            )
        )
    if not rows:
        return
    lines.extend(
        [
            "",
            "## Recommended outputs",
            "| file | category | path | description |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, relative_path, filename, category, description in sorted(rows):
        lines.append(f"| {filename} | {category} | `{relative_path}` | {description} |")

def _write_auto_final_report(
    *,
    working_directory: Path,
    result: AutoRawDataAnalysisResult,
) -> str:
    output_dir = working_directory / "result" / "00_auto_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "auto_final_report.md"

    rawdata = _artifact_or_none(result.runtime, "auto_rawdata_load_result")
    registration = result.pipeline_build_result.registration if result.pipeline_build_result else None
    validation = _artifact_or_none(result.runtime, "auto_run_validation_report")

    lines = [
        "# \ucd5c\uc885 \uc790\ub3d9 \ubd84\uc11d \ub9ac\ud3ec\ud2b8",
        "",
        f"- \ud504\ub85c\uc81d\ud2b8\uba85: {result.context.project_name}",
        f"- \uc804\uccb4 \uc0c1\ud0dc: {_format_bool(result.success)}",
        f"- \uc2e4\ud328 \ub2e8\uacc4: {result.failed_stage or '-'}",
        f"- \uc0b0\ucd9c\ubb3c \uc218: {len(result.output_files)}",
        f"- \uacbd\uace0 \uc218: {len(result.warnings)}",
    ]
    _append_recommended_outputs(lines, working_directory=working_directory, result=result)
    lines.extend([
        "",
        "## \uc6d0\uc790\ub8cc",
    ])
    if rawdata is not None:
        candidate = rawdata.selected_candidate
        lines.extend(
            [
                f"- \uc120\ud0dd \ud30c\uc77c: `{candidate.source_path}`",
                f"- \ud589/\uc5f4: {candidate.row_count} / {candidate.column_count}",
                f"- Sheet: {candidate.sheet_name or '-'}",
            ]
        )
    else:
        lines.append("- \uc6d0\uc790\ub8cc \uc120\ud0dd \uc815\ubcf4\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.")

    lines.extend(["", "## Main model"])
    if registration is not None:
        main_status = result.orchestrator_result.success if result.orchestrator_result is not None else result.pipeline_build_result.success
        lines.extend(
            [
                "| model_id | status | model_type | dependent | independent |",
                "| --- | --- | --- | --- | --- |",
                f"| {registration.model_id} | {_format_bool(bool(main_status))} | {registration.model_type} | "
                f"{registration.dependent_variable} | {', '.join(registration.independent_variables)} |",
            ]
        )
    else:
        lines.append("- Main model was not registered.")

    lines.extend(["", "## Multi-outcome models"])
    if result.multi_outcome_pipeline_build_result is not None:
        run_result = result.multi_outcome_pipeline_run_result
        lines.extend(
            [
                "| model_id | status | model_type | dependent | independent |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for model_id, build_result in result.multi_outcome_pipeline_build_result.model_results.items():
            registration = build_result.registration
            if run_result is None:
                status = build_result.success
            elif model_id in run_result.completed_models:
                status = True
            else:
                status = False
            model_type = registration.model_type if registration is not None else "-"
            dependent = registration.dependent_variable if registration is not None else "-"
            independent = ", ".join(registration.independent_variables) if registration is not None else "-"
            lines.append(f"| {model_id} | {_format_bool(status)} | {model_type} | {dependent} | {independent} |")
    else:
        lines.append("- Multi-outcome analysis was not enabled.")

    _append_model_result_summary(lines, result.runtime, title="Main model results")
    if result.multi_outcome_pipeline_build_result is not None:
        for model_id, model_runtime in result.multi_outcome_pipeline_build_result.runtimes.items():
            _append_model_result_summary(lines, model_runtime, title=f"Multi-outcome result: {model_id}")

    lines.extend(
        [
            "",
            "## \ub2e8\uacc4\ubcc4 \uc694\uc57d",
            "| stage | status | outputs | warnings |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for step_result in result.setup_step_results:
        lines.append(
            f"| {step_result.stage_name} | {_format_bool(step_result.success)} | "
            f"{len(step_result.output_files)} | {len(step_result.warnings)} |"
        )
    if result.pipeline_build_result is not None:
        lines.append(
            "| 04_auto_pipeline_registration | "
            f"{_format_bool(result.pipeline_build_result.success)} | 0 | {len(result.pipeline_build_result.warnings)} |"
        )
    if result.orchestrator_result is not None:
        lines.append(
            "| 05_auto_pipeline_execution | "
            f"{_format_bool(result.orchestrator_result.success)} | "
            f"{len(result.context.generated_files)} | {len(result.orchestrator_result.warnings)} |"
        )
    if result.multi_outcome_pipeline_build_result is not None:
        lines.append(
            "| 04b_auto_multi_outcome_pipeline_registration | "
            f"{_format_bool(result.multi_outcome_pipeline_build_result.success)} | 0 | "
            f"{len(result.multi_outcome_pipeline_build_result.warnings)} |"
        )
    if result.multi_outcome_pipeline_run_result is not None:
        lines.append(
            "| 05b_auto_multi_outcome_pipeline_execution | "
            f"{_format_bool(result.multi_outcome_pipeline_run_result.success)} | "
            f"{len(result.multi_outcome_pipeline_run_result.completed_models)} | "
            f"{len(result.multi_outcome_pipeline_run_result.warnings)} |"
        )

    if validation is not None:
        lines.extend(["", "## \uac80\uc99d \uc694\uc57d", f"- \uc0c1\ud0dc: {_format_bool(validation.passed)}"])
        if validation.warnings:
            lines.extend(f"- {warning}" for warning in validation.warnings)

    if result.warnings:
        lines.extend(["", "## \uacbd\uace0"])
        lines.extend(f"- {warning}" for warning in result.warnings)

    lines.extend(["", "## \uc0b0\ucd9c\ubb3c"])
    lines.extend(f"- `{output_file}`" for output_file in result.output_files)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)

def run_auto_rawdata_analysis(
    working_directory: str | Path = ".",
    *,
    rawdata_dir: str | Path = "rawdata",
    source_file: str | Path | None = None,
    auto_merge: bool = True,
    codebook_dir: str | Path = "codebook",
    questionnaire_dir: str | Path = "questionnaire",
    project_name: str = "auto_rawdata_analysis",
    enable_robustness: bool = False,
    run_analysis: bool = True,
    model_id: str = "main_model",
    dependent_variable: str | None = None,
    independent_variables: list[str] | tuple[str, ...] | None = None,
    control_variables: list[str] | tuple[str, ...] | None = None,
    cluster_variable: str | None = None,
    weight_variable: str | None = None,
    id_variable: str | None = None,
    time_variable: str | None = None,
    enable_multi_outcome: bool = False,
    max_outcomes: int = 3,
) -> AutoRawDataAnalysisResult:
    """Run the rawdata-only workflow through automatic planning and analysis."""
    root = Path(working_directory).expanduser().resolve()
    context = ResearchContext(project_name=project_name)
    runtime = PipelineRuntime()
    output_files: list[str] = []
    warnings: list[str] = []
    setup_results: list[StepResult] = []
    role_overrides = build_auto_variable_role_overrides(
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        control_variables=control_variables,
        cluster_variable=cluster_variable,
        weight_variable=weight_variable,
        id_variable=id_variable,
        time_variable=time_variable,
    )

    setup_steps = [
        AutoRawDataLoadingStep(
            runtime,
            rawdata_dir=rawdata_dir,
            source_file=source_file,
            auto_merge=auto_merge,
            codebook_dir=codebook_dir,
            questionnaire_dir=questionnaire_dir,
        ),
        AutoVariableInferenceStep(runtime),
        AutoAnalysisPlanStep(runtime, enable_robustness=enable_robustness),
    ]
    if enable_multi_outcome:
        setup_steps.append(
            AutoMultiOutcomeAnalysisPlanStep(
                runtime,
                max_outcomes=max_outcomes,
                model_id_prefix=model_id,
                enable_robustness=enable_robustness,
            )
        )
    for step in setup_steps:
        try:
            step_result = step.run(context, root)
        except Exception as error:  # noqa: BLE001 - convert setup failures into a structured auto-run result.
            step_result = StepResult(
                stage_name=step.name,
                success=False,
                warnings=[str(error)],
                metadata={"error_message": str(error)},
            )
        setup_results.append(step_result)
        _record_step_result(
            context=context,
            result=step_result,
            output_files=output_files,
            warnings=warnings,
        )
        if step_result.success and step.name == "02_auto_variable_inference" and role_overrides.has_overrides():
            try:
                overridden_map = apply_variable_role_overrides(
                    runtime.get_artifact("auto_variable_map"),
                    role_overrides,
                )
                runtime.set_artifact("auto_variable_map", overridden_map)
                override_dir = root / "result" / "02_auto_variables"
                override_dir.mkdir(parents=True, exist_ok=True)
                override_path = override_dir / "overridden_variable_map.xlsx"
                variable_map_to_dataframe(overridden_map).to_excel(override_path, index=False)
                output_files.append(str(override_path))
                context.add_generated_file(override_path)
            except Exception as error:  # noqa: BLE001 - convert invalid overrides into structured failure.
                step_result = StepResult(
                    stage_name="02_auto_variable_role_overrides",
                    success=False,
                    warnings=[str(error)],
                    metadata={"error_message": str(error)},
                )
                setup_results.append(step_result)
                _record_step_result(
                    context=context,
                    result=step_result,
                    output_files=output_files,
                    warnings=warnings,
                )

        if not step_result.success:
            result = AutoRawDataAnalysisResult(
                success=False,
                context=context,
                runtime=runtime,
                setup_step_results=setup_results,
                output_files=output_files,
                warnings=warnings,
                failed_stage=step_result.stage_name,
            )
            summary_path = _write_auto_run_summary(working_directory=root, result=result)
            report_path = _write_auto_run_markdown(working_directory=root, result=result)
            final_report_path = _write_auto_final_report(working_directory=root, result=result)
            result.output_files.extend([summary_path, report_path, final_report_path])
            context.add_generated_file(summary_path)
            context.add_generated_file(report_path)
            context.add_generated_file(final_report_path)
            manifest_path = _write_output_manifest(working_directory=root, result=result)
            result.output_files.append(manifest_path)
            context.add_generated_file(manifest_path)
            validation_report = validate_auto_run_outputs(
                runtime=runtime,
                output_files=result.output_files,
            )
            runtime.set_artifact("auto_run_validation_report", validation_report)
            validation_report_path = _write_auto_validation_report(
                working_directory=root,
                validation_report=validation_report,
            )
            result.output_files.append(validation_report_path)
            context.add_generated_file(validation_report_path)
            result.warnings.extend(validation_report.warnings)
            _write_auto_final_report(working_directory=root, result=result)
            _write_output_manifest(working_directory=root, result=result)
            return result

    analysis_plan = runtime.get_artifact("auto_analysis_plan")
    variable_map = runtime.get_artifact("auto_variable_map")
    if isinstance(analysis_plan, AnalysisPlan) and isinstance(variable_map, VariableMap):
        _apply_plan_to_context(context, analysis_plan, variable_map)

    orchestrator, build_result = build_auto_regression_orchestrator(
        context=context,
        runtime=runtime,
        working_directory=root,
        model_id=model_id,
    )
    warnings.extend(build_result.warnings)
    for warning in build_result.warnings:
        context.add_warning(f"04_auto_pipeline_registration: {warning}")

    orchestrator_result: OrchestratorResult | None = None
    multi_outcome_build_result: AutoMultiOutcomePipelineBuildResult | None = None
    multi_outcome_run_result: AutoMultiOutcomePipelineRunResult | None = None
    success = build_result.success
    failed_stage = None if success else "04_auto_pipeline_registration"
    if build_result.success and run_analysis:
        orchestrator_result = orchestrator.run()
        success = orchestrator_result.success
        failed_stage = orchestrator_result.failed_stage
        warnings.extend(orchestrator_result.warnings)
        output_files = list(dict.fromkeys(output_files + context.generated_files))

    if enable_multi_outcome:
        multi_outcome_build_result = build_auto_multi_outcome_regression_orchestrators(
            context=context,
            runtime=runtime,
            working_directory=root,
        )
        warnings.extend(multi_outcome_build_result.warnings)
        for warning in multi_outcome_build_result.warnings:
            context.add_warning(f"04b_auto_multi_outcome_pipeline_registration: {warning}")
        if not multi_outcome_build_result.success:
            success = False
            failed_stage = failed_stage or "04b_auto_multi_outcome_pipeline_registration"
        elif run_analysis and build_result.success:
            multi_outcome_run_result = run_auto_multi_outcome_regression_orchestrators(
                multi_outcome_build_result,
                runtime=runtime,
            )
            success = success and multi_outcome_run_result.success
            if not multi_outcome_run_result.success:
                failed_stage = failed_stage or "05b_auto_multi_outcome_pipeline_execution"
            warnings.extend(multi_outcome_run_result.warnings)
            multi_output_files = [
                str(generated_file)
                for model_orchestrator in multi_outcome_build_result.orchestrators.values()
                for generated_file in model_orchestrator.context.generated_files
            ]
            output_files = list(dict.fromkeys(output_files + context.generated_files + multi_output_files))

    result = AutoRawDataAnalysisResult(
        success=success,
        context=context,
        runtime=runtime,
        setup_step_results=setup_results,
        pipeline_build_result=build_result,
        multi_outcome_pipeline_build_result=multi_outcome_build_result,
        multi_outcome_pipeline_run_result=multi_outcome_run_result,
        orchestrator=orchestrator,
        orchestrator_result=orchestrator_result,
        output_files=list(dict.fromkeys(output_files)),
        warnings=warnings,
        failed_stage=failed_stage,
    )
    summary_path = _write_auto_run_summary(working_directory=root, result=result)
    report_path = _write_auto_run_markdown(working_directory=root, result=result)
    final_report_path = _write_auto_final_report(working_directory=root, result=result)
    result.output_files.extend([summary_path, report_path, final_report_path])
    context.add_generated_file(summary_path)
    context.add_generated_file(report_path)
    context.add_generated_file(final_report_path)
    manifest_path = _write_output_manifest(working_directory=root, result=result)
    result.output_files.append(manifest_path)
    context.add_generated_file(manifest_path)
    validation_report = validate_auto_run_outputs(
        runtime=runtime,
        output_files=result.output_files,
        require_model_outputs=run_analysis,
    )
    runtime.set_artifact("auto_run_validation_report", validation_report)
    validation_report_path = _write_auto_validation_report(
        working_directory=root,
        validation_report=validation_report,
    )
    result.output_files.append(validation_report_path)
    context.add_generated_file(validation_report_path)
    if validation_report.warnings:
        result.warnings.extend(validation_report.warnings)
        for warning in validation_report.warnings:
            context.add_warning(f"auto_run_validation: {warning}")
    _write_auto_final_report(working_directory=root, result=result)
    _write_output_manifest(working_directory=root, result=result)
    return result
