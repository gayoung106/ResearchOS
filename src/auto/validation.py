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
        return "원자료 파일이 rawdata 폴더에 있는지, 파일이 읽을 수 있는 CSV/Excel 형식인지 확인하세요."
    if item.startswith("artifact:auto_variable_map"):
        return "변수 추론 단계가 성공했는지 확인하고, 코드북/설문지 또는 수동 변수 보정 옵션을 추가하세요."
    if item.startswith("artifact:auto_analysis_plan"):
        return "종속변수와 독립변수가 하나 이상 추론되도록 변수명, 코드북 role 힌트, 수동 보정 옵션을 확인하세요."
    if item.startswith("artifact:auto_regression_pipeline_build_result"):
        return "자동 분석계획이 생성된 뒤 Builder 등록 단계가 실행되었는지 확인하세요."
    if item.startswith("file:auto_run_report.md") or item.startswith("file:auto_final_report.md"):
        return "자동 실행이 중간에 중단되었는지 확인하고 result/00_auto_run 폴더 쓰기 권한을 확인하세요."
    if item.startswith("file:analysis_base.parquet"):
        return "원자료 로딩 단계가 완료되었는지 확인하고 pyarrow/parquet 저장 의존성을 확인하세요."
    if item.startswith("file:variable_role_inference") or item.startswith("file:inferred_variable_map"):
        return "변수 추론 단계 로그와 variable_metadata.xlsx를 확인하세요."
    if item.startswith("file:auto_analysis_plan") or item.startswith("file:auto_variable_map"):
        return "분석계획 생성 단계가 성공했는지 확인하세요."
    if item.startswith("file:coefficients") or item.startswith("file:fit_statistics"):
        return "모델 실행 단계가 성공했는지 확인하고, --plan-only로 계획을 먼저 점검하세요."
    if item.startswith("yaml:auto_analysis_plan"):
        return "auto_analysis_plan.yaml에서 dependent와 regression.enabled 값을 확인하세요."
    if item.startswith("yaml:auto_variable_map"):
        return "auto_variable_map.yaml이 비어 있거나 손상되지 않았는지 확인하세요."
    return "해당 단계의 이전 산출물과 경고 메시지를 확인하세요."

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
