"""연구계획 및 분석계획 설정 모델."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

VariableRole = Literal[
    "dependent",
    "independent",
    "mediator",
    "moderator",
    "control",
    "fixed_effect",
    "weight",
    "cluster",
    "strata",
    "id",
    "time",
    "other",
]

MeasurementLevel = Literal[
    "binary",
    "nominal",
    "ordinal",
    "continuous",
    "count",
    "proportion",
    "datetime",
    "string",
    "multi_response",
    "scale_item",
    "unknown",
]


class StrictModel(BaseModel):
    """알 수 없는 설정 키를 허용하지 않는 기본 모델."""

    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    """프로젝트 기본정보."""

    title: str = ""
    short_name: str = ""
    field: str = ""
    language: str = "ko"
    author: str = ""
    created_at: str = ""


class ResearchConfig(StrictModel):
    """연구주제 및 이론정보."""

    topic: str = ""
    purpose: str = ""
    research_questions: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    theoretical_framework: str = ""
    contribution: str = ""


class DesignConfig(StrictModel):
    """연구설계 정보."""

    study_type: str = ""
    unit_of_analysis: str = ""
    data_structure: str = ""
    time_structure: str = ""
    sampling_design: str = ""
    causal_claim_allowed: bool = False


class DataConfig(StrictModel):
    """입력자료 및 표본설계 변수."""

    raw_files: list[str] = Field(default_factory=list)
    questionnaire_files: list[str] = Field(default_factory=list)
    codebook_files: list[str] = Field(default_factory=list)
    id_variable: str | None = None
    time_variable: str | None = None
    weight_variable: str | None = None
    cluster_variable: str | None = None
    strata_variable: str | None = None


class ResearchReviewConfig(StrictModel):
    """연구설계 단계의 미확정 사항."""

    unresolved_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ResearchPlan(StrictModel):
    """research_plan.yaml 전체 모델."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    design: DesignConfig = Field(default_factory=DesignConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    review: ResearchReviewConfig = Field(default_factory=ResearchReviewConfig)


class VariableGroups(StrictModel):
    """분석에서 사용하는 변수 역할별 목록."""

    dependent: list[str] = Field(default_factory=list)
    independent: list[str] = Field(default_factory=list)
    mediators: list[str] = Field(default_factory=list)
    moderators: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    fixed_effects: list[str] = Field(default_factory=list)
    weights: list[str] = Field(default_factory=list)
    clusters: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_role_duplicates(self) -> VariableGroups:
        """동일 변수가 상충하는 역할에 중복 지정되는지 검사한다."""
        role_map = {
            "dependent": self.dependent,
            "independent": self.independent,
            "mediators": self.mediators,
            "moderators": self.moderators,
            "controls": self.controls,
        }

        occurrences: dict[str, list[str]] = {}
        for role, variables in role_map.items():
            for variable in variables:
                occurrences.setdefault(variable, []).append(role)

        conflicts = {variable: roles for variable, roles in occurrences.items() if len(roles) > 1}

        if conflicts:
            details = "; ".join(
                f"{variable}: {', '.join(roles)}" for variable, roles in conflicts.items()
            )
            raise ValueError(f"동일 변수가 여러 핵심 역할에 중복 지정되었습니다: {details}")

        return self


class PreprocessingConfig(StrictModel):
    """전처리 설정."""

    missing_value_rules: list[dict[str, Any]] = Field(default_factory=list)
    recoding_rules: list[dict[str, Any]] = Field(default_factory=list)
    reverse_items: list[str] = Field(default_factory=list)
    scale_rules: list[dict[str, Any]] = Field(default_factory=list)
    derived_variables: list[dict[str, Any]] = Field(default_factory=list)
    exclusions: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisBlock(StrictModel):
    """개별 분석영역 설정."""

    enabled: bool = False
    options: dict[str, Any] = Field(default_factory=dict)
    methods: list[str] = Field(default_factory=list)
    models: list[dict[str, Any]] = Field(default_factory=list)
    scales: list[dict[str, Any]] = Field(default_factory=list)
    checks: list[dict[str, Any]] = Field(default_factory=list)


class AnalysesConfig(StrictModel):
    """전체 분석 설정."""

    descriptive: AnalysisBlock = Field(default_factory=lambda: AnalysisBlock(enabled=True))
    reliability: AnalysisBlock = Field(default_factory=AnalysisBlock)
    validity: AnalysisBlock = Field(default_factory=AnalysisBlock)
    regression: AnalysisBlock = Field(default_factory=AnalysisBlock)
    mediation: AnalysisBlock = Field(default_factory=AnalysisBlock)
    moderation: AnalysisBlock = Field(default_factory=AnalysisBlock)
    multilevel: AnalysisBlock = Field(default_factory=AnalysisBlock)
    panel: AnalysisBlock = Field(default_factory=AnalysisBlock)
    robustness: AnalysisBlock = Field(default_factory=lambda: AnalysisBlock(enabled=True))


class TableOutputConfig(StrictModel):
    """표 출력 설정."""

    formats: list[str] = Field(default_factory=lambda: ["xlsx", "docx", "html", "md"])


class FigureOutputConfig(StrictModel):
    """그림 출력 설정."""

    formats: list[str] = Field(default_factory=lambda: ["png", "svg"])


class ManuscriptOutputConfig(StrictModel):
    """논문 초안 출력 설정."""

    methods_draft: bool = True
    results_draft: bool = True


class OutputsConfig(StrictModel):
    """전체 출력 설정."""

    tables: TableOutputConfig = Field(default_factory=TableOutputConfig)
    figures: FigureOutputConfig = Field(default_factory=FigureOutputConfig)
    manuscript: ManuscriptOutputConfig = Field(default_factory=ManuscriptOutputConfig)


class AnalysisReviewConfig(StrictModel):
    """분석 종료 후 검토 역할."""

    required_roles: list[str] = Field(default_factory=list)


class AnalysisPlan(StrictModel):
    """analysis_plan.yaml 전체 모델."""

    variables: VariableGroups = Field(default_factory=VariableGroups)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    analyses: AnalysesConfig = Field(default_factory=AnalysesConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    review: AnalysisReviewConfig = Field(default_factory=AnalysisReviewConfig)


class VariableDefinition(StrictModel):
    """개별 변수 정의."""

    original_name: str = ""
    korean_name: str = ""
    label: str = ""
    question_text: str = ""
    role: VariableRole = "other"
    measurement_level: MeasurementLevel = "unknown"
    coding: dict[str, Any] = Field(default_factory=dict)
    missing_values: list[Any] = Field(default_factory=list)
    reverse_coded: bool = False
    scale_name: str | None = None
    preprocessing: list[dict[str, Any]] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    review_status: str = "unreviewed"
    notes: str = ""


class VariableMap(StrictModel):
    """variable_map.yaml 전체 모델."""

    variables: dict[str, VariableDefinition] = Field(default_factory=dict)
