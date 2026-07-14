"""측정수준과 연구설계에 따른 분석방법 지식베이스."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnalysisKnowledge:
    """분석방법에 관한 기본 지식."""

    method_id: str
    korean_name: str
    applicable_outcomes: tuple[str, ...]
    description: str
    required_diagnostics: tuple[str, ...]
    common_outputs: tuple[str, ...]
    cautions: tuple[str, ...]


ANALYSIS_KNOWLEDGE_BASE: dict[str, AnalysisKnowledge] = {
    "ols": AnalysisKnowledge(
        method_id="ols",
        korean_name="선형회귀분석",
        applicable_outcomes=("continuous",),
        description="연속형 종속변수의 조건부 평균을 설명하는 기본 회귀모형",
        required_diagnostics=(
            "선형성",
            "이분산성",
            "잔차 진단",
            "다중공선성",
            "영향력 관측치",
        ),
        common_outputs=(
            "비표준화 계수",
            "표준오차",
            "신뢰구간",
            "결정계수",
            "표본 수",
        ),
        cautions=(
            "비선형 관계를 선형으로 단정하지 않는다.",
            "관찰자료에서는 계수를 인과효과로 단정하지 않는다.",
        ),
    ),
    "binary_logit": AnalysisKnowledge(
        method_id="binary_logit",
        korean_name="이항 로지스틱 회귀분석",
        applicable_outcomes=("binary",),
        description="이분형 종속변수의 사건발생 확률을 설명하는 모형",
        required_diagnostics=(
            "완전 또는 준완전분리",
            "희소범주",
            "연속변수의 로짓 선형성",
            "다중공선성",
            "모형 적합도",
        ),
        common_outputs=(
            "로짓계수",
            "오즈비",
            "신뢰구간",
            "예측확률",
            "평균한계효과",
        ),
        cautions=(
            "오즈비를 확률비로 해석하지 않는다.",
            "사건 수가 부족한 경우 추정이 불안정할 수 있다.",
        ),
    ),
    "ordered_logit": AnalysisKnowledge(
        method_id="ordered_logit",
        korean_name="순서형 로지스틱 회귀분석",
        applicable_outcomes=("ordinal",),
        description="순서가 있는 범주형 종속변수를 분석하는 모형",
        required_diagnostics=(
            "비례오즈 가정",
            "희소범주",
            "다중공선성",
            "모형 수렴",
        ),
        common_outputs=(
            "계수",
            "오즈비",
            "임계값",
            "예측확률",
            "평균한계효과",
        ),
        cautions=(
            "범주 간 간격이 동일하다고 가정하지 않는다.",
            "비례오즈 가정 위반 시 대안모형을 검토한다.",
        ),
    ),
    "multinomial_logit": AnalysisKnowledge(
        method_id="multinomial_logit",
        korean_name="다항 로지스틱 회귀분석",
        applicable_outcomes=("nominal",),
        description="순서가 없는 다범주 종속변수를 분석하는 모형",
        required_diagnostics=(
            "기준범주 설정",
            "희소범주",
            "다중공선성",
            "IIA 가정 검토",
            "모형 수렴",
        ),
        common_outputs=(
            "범주별 계수",
            "상대위험비",
            "예측확률",
            "평균한계효과",
        ),
        cautions=(
            "기준범주에 따라 계수 해석이 달라진다.",
            "IIA 가정의 적절성을 검토한다.",
        ),
    ),
    "poisson": AnalysisKnowledge(
        method_id="poisson",
        korean_name="포아송 회귀분석",
        applicable_outcomes=("count",),
        description="0 이상의 횟수형 종속변수를 분석하는 기본 모형",
        required_diagnostics=(
            "과산포",
            "영과다",
            "노출량 또는 오프셋",
            "모형 적합도",
        ),
        common_outputs=(
            "계수",
            "발생률비",
            "신뢰구간",
            "예측횟수",
        ),
        cautions=(
            "평균과 분산이 크게 다르면 음이항 모형을 검토한다.",
            "노출시간이 다르면 오프셋을 고려한다.",
        ),
    ),
    "fractional_logit": AnalysisKnowledge(
        method_id="fractional_logit",
        korean_name="분수형 로짓 모형",
        applicable_outcomes=("proportion",),
        description="0과 1 사이의 비율형 종속변수를 분석하는 모형",
        required_diagnostics=(
            "종속변수 범위",
            "함수형태",
            "강건표준오차",
        ),
        common_outputs=(
            "계수",
            "예측비율",
            "평균한계효과",
        ),
        cautions=("단순 OLS보다 경계값과 비선형성을 적절히 처리한다.",),
    ),
}


OUTCOME_TO_METHODS: dict[str, tuple[str, ...]] = {
    "continuous": ("ols",),
    "binary": ("binary_logit",),
    "ordinal": ("ordered_logit",),
    "nominal": ("multinomial_logit",),
    "count": ("poisson",),
    "proportion": ("fractional_logit",),
}


def get_analysis_knowledge(method_id: str) -> AnalysisKnowledge:
    """분석방법 지식정보를 반환한다."""
    try:
        return ANALYSIS_KNOWLEDGE_BASE[method_id]
    except KeyError as error:
        raise KeyError(f"등록되지 않은 분석방법입니다: {method_id}") from error


def get_methods_for_outcome(measurement_level: str) -> tuple[str, ...]:
    """종속변수 측정수준에 대응하는 기본 분석방법을 반환한다."""
    return OUTCOME_TO_METHODS.get(measurement_level, ())
