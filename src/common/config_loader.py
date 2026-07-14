"""YAML 설정 로딩, 검증 및 ResearchContext 변환."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from src.common.config_exceptions import (
    ConfigFileNotFoundError,
    ConfigValidationError,
)
from src.common.config_models import AnalysisPlan, ResearchPlan, VariableMap
from src.pipeline.context import ResearchContext


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """UTF-8 YAML 파일을 딕셔너리로 읽는다."""
    file_path = Path(path)

    if not file_path.exists():
        raise ConfigFileNotFoundError(f"설정 파일을 찾을 수 없습니다: {file_path}")

    if not file_path.is_file():
        raise ConfigValidationError(f"설정 경로가 파일이 아닙니다: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8-sig") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ConfigValidationError(f"YAML 문법 오류가 있습니다: {file_path}\n{error}") from error

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigValidationError(f"YAML 최상위 구조는 객체여야 합니다: {file_path}")

    return data


def load_model[ModelType: BaseModel](
    path: str | Path,
    model_type: type[ModelType],
) -> ModelType:
    """YAML 파일을 지정된 Pydantic 모델로 검증한다."""
    data = load_yaml_file(path)

    try:
        return model_type.model_validate(data)
    except ValidationError as error:
        messages = []

        for item in error.errors():
            location = ".".join(str(part) for part in item["loc"])
            message = item["msg"]
            messages.append(f"- {location}: {message}")

        details = "\n".join(messages)
        raise ConfigValidationError(f"설정값 검증에 실패했습니다: {path}\n{details}") from error


def load_research_plan(path: str | Path) -> ResearchPlan:
    """research_plan.yaml을 읽고 검증한다."""
    return load_model(path, ResearchPlan)


def load_analysis_plan(path: str | Path) -> AnalysisPlan:
    """analysis_plan.yaml을 읽고 검증한다."""
    return load_model(path, AnalysisPlan)


def load_variable_map(path: str | Path) -> VariableMap:
    """variable_map.yaml을 읽고 검증한다."""
    return load_model(path, VariableMap)


def validate_cross_config(
    research_plan: ResearchPlan,
    analysis_plan: AnalysisPlan,
    variable_map: VariableMap,
) -> list[str]:
    """
    여러 설정 파일 사이의 논리적 불일치를 검사한다.

    오류는 예외로 처리하고, 아직 확정되지 않은 사항은 경고 목록으로 반환한다.
    """
    warnings: list[str] = []

    if not research_plan.project.title.strip():
        warnings.append("프로젝트 제목이 비어 있습니다.")

    if not research_plan.research.topic.strip():
        warnings.append("연구주제가 비어 있습니다.")

    if not research_plan.research.research_questions:
        warnings.append("연구질문이 등록되지 않았습니다.")

    groups = analysis_plan.variables
    referenced_variables = set(
        groups.dependent
        + groups.independent
        + groups.mediators
        + groups.moderators
        + groups.controls
        + groups.fixed_effects
        + groups.weights
        + groups.clusters
    )

    defined_variables = set(variable_map.variables)
    missing_definitions = sorted(referenced_variables - defined_variables)

    if missing_definitions:
        warnings.append(
            "variable_map에 정의되지 않은 분석 변수가 있습니다: " + ", ".join(missing_definitions)
        )

    if analysis_plan.analyses.regression.enabled and not groups.dependent:
        raise ConfigValidationError("회귀분석이 활성화되었지만 종속변수가 지정되지 않았습니다.")

    if analysis_plan.analyses.mediation.enabled and not groups.mediators:
        raise ConfigValidationError("매개분석이 활성화되었지만 매개변수가 지정되지 않았습니다.")

    if analysis_plan.analyses.moderation.enabled and not groups.moderators:
        raise ConfigValidationError("조절분석이 활성화되었지만 조절변수가 지정되지 않았습니다.")

    if analysis_plan.analyses.multilevel.enabled and not (
        groups.clusters or research_plan.data.cluster_variable
    ):
        raise ConfigValidationError("다층분석이 활성화되었지만 군집변수가 지정되지 않았습니다.")

    if analysis_plan.analyses.panel.enabled and not (
        research_plan.data.id_variable and research_plan.data.time_variable
    ):
        raise ConfigValidationError(
            "패널분석이 활성화되었지만 ID 변수와 시점 변수가 모두 지정되지 않았습니다."
        )

    return warnings


def build_research_context(
    research_plan: ResearchPlan,
    analysis_plan: AnalysisPlan,
    variable_map: VariableMap,
) -> ResearchContext:
    """검증된 설정 모델을 ResearchContext로 변환한다."""
    warnings = validate_cross_config(
        research_plan,
        analysis_plan,
        variable_map,
    )

    project_name = (
        research_plan.project.title.strip()
        or research_plan.project.short_name.strip()
        or "미지정 연구"
    )

    context = ResearchContext(
        project_name=project_name,
        research_topic=research_plan.research.topic,
        research_questions=research_plan.research.research_questions,
        hypotheses=research_plan.research.hypotheses,
        raw_data_files=research_plan.data.raw_files,
        questionnaire_files=research_plan.data.questionnaire_files,
        codebook_files=research_plan.data.codebook_files,
        dependent_variables=analysis_plan.variables.dependent,
        independent_variables=analysis_plan.variables.independent,
        mediator_variables=analysis_plan.variables.mediators,
        moderator_variables=analysis_plan.variables.moderators,
        control_variables=analysis_plan.variables.controls,
        analysis_plan=analysis_plan.model_dump(),
        variable_map=variable_map.model_dump(),
        warnings=warnings,
    )

    return context
