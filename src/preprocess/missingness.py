"""결측치 진단 및 처리전략 추천 모듈."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class MissingnessRecommendation:
    """결측치 처리전략 추천."""

    strategy: str
    priority: str
    reason: str
    cautions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MissingnessReport:
    """결측치 진단 결과."""

    row_count: int
    column_count: int
    total_missing_count: int
    overall_missing_rate: float
    variable_summary: pd.DataFrame
    case_summary: pd.DataFrame
    pattern_summary: pd.DataFrame
    recommendations: list[MissingnessRecommendation]
    warnings: list[str]


def variable_missingness_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """변수별 결측치 수와 비율을 계산한다."""
    row_count = len(dataframe)

    rows: list[dict[str, Any]] = []

    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())
        missing_rate = missing_count / row_count if row_count > 0 else 0.0

        rows.append(
            {
                "variable_name": str(column),
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "non_missing_count": row_count - missing_count,
                "unique_count": int(dataframe[column].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["missing_rate", "variable_name"],
        ascending=[False, True],
        ignore_index=True,
    )


def case_missingness_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """사례별 결측치 수와 비율을 계산한다."""
    column_count = len(dataframe.columns)

    missing_count = dataframe.isna().sum(axis=1)
    missing_rate = missing_count / column_count if column_count > 0 else 0.0

    return pd.DataFrame(
        {
            "row_index": dataframe.index,
            "missing_count": missing_count.to_numpy(),
            "missing_rate": missing_rate.to_numpy(),
            "complete_case": (missing_count == 0).to_numpy(),
        }
    )


def missingness_pattern_summary(
    dataframe: pd.DataFrame,
    *,
    max_patterns: int = 50,
) -> pd.DataFrame:
    """
    변수별 결측 여부 조합을 패턴으로 요약한다.

    1은 결측, 0은 관측을 의미한다.
    """
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "pattern",
                "frequency",
                "percentage",
                "missing_variable_count",
            ]
        )

    indicator = dataframe.isna().astype(int)
    pattern = indicator.astype(str).agg("".join, axis=1)

    summary = pattern.value_counts().rename_axis("pattern").reset_index(name="frequency")
    summary["percentage"] = summary["frequency"] / len(dataframe) * 100
    summary["missing_variable_count"] = summary["pattern"].str.count("1")

    return summary.head(max_patterns).reset_index(drop=True)


def recommend_missingness_strategy(
    variable_summary: pd.DataFrame,
    *,
    overall_missing_rate: float,
) -> list[MissingnessRecommendation]:
    """결측률 수준에 따라 처리전략 후보를 제안한다."""
    recommendations: list[MissingnessRecommendation] = []

    if variable_summary.empty:
        return [
            MissingnessRecommendation(
                strategy="no_action",
                priority="low",
                reason="진단할 변수가 없습니다.",
            )
        ]

    max_rate = float(variable_summary["missing_rate"].max())
    variables_over_20 = variable_summary.loc[
        variable_summary["missing_rate"] >= 0.20,
        "variable_name",
    ].tolist()

    if overall_missing_rate == 0:
        recommendations.append(
            MissingnessRecommendation(
                strategy="no_action",
                priority="low",
                reason="전체 데이터에 결측값이 없습니다.",
            )
        )
        return recommendations

    if max_rate < 0.05 and overall_missing_rate < 0.02:
        recommendations.append(
            MissingnessRecommendation(
                strategy="complete_case_review",
                priority="normal",
                reason=(
                    "변수별 결측률과 전체 결측률이 낮아 완전사례 분석을 우선 검토할 수 있습니다."
                ),
                cautions=[
                    "결측이 완전무작위라는 가정을 자동으로 충족하는 것은 아닙니다.",
                    "분석별 표본 감소폭을 함께 확인해야 합니다.",
                ],
            )
        )

    if 0.05 <= max_rate < 0.20:
        recommendations.append(
            MissingnessRecommendation(
                strategy="multiple_imputation_review",
                priority="high",
                reason=("일부 변수의 결측률이 5% 이상이므로 다중대체 가능성을 검토해야 합니다."),
                cautions=[
                    "대체모형에는 분석모형 변수와 결측 관련 보조변수를 포함해야 합니다.",
                    "단순 평균대체를 기본값으로 사용하지 않습니다.",
                ],
            )
        )

    if variables_over_20:
        recommendations.append(
            MissingnessRecommendation(
                strategy="high_missingness_variable_review",
                priority="high",
                reason=("결측률이 20% 이상인 변수가 있습니다: " + ", ".join(variables_over_20)),
                cautions=[
                    "변수 제외 여부를 결측률만으로 결정하지 않습니다.",
                    "설문 분기, 구조적 결측, 조사설계를 먼저 확인해야 합니다.",
                ],
            )
        )

    recommendations.append(
        MissingnessRecommendation(
            strategy="missingness_mechanism_review",
            priority="high",
            reason=("MCAR·MAR·MNAR 가능성을 이론과 관측자료를 함께 고려해 검토해야 합니다."),
            cautions=[
                "통계검정 하나만으로 결측 메커니즘을 확정하지 않습니다.",
            ],
        )
    )

    return recommendations


def build_missingness_report(
    dataframe: pd.DataFrame,
) -> MissingnessReport:
    """결측치 진단 보고서를 생성한다."""
    row_count = len(dataframe)
    column_count = len(dataframe.columns)
    total_cells = row_count * column_count
    total_missing_count = int(dataframe.isna().sum().sum())
    overall_missing_rate = total_missing_count / total_cells if total_cells > 0 else 0.0

    variable_summary = variable_missingness_summary(dataframe)
    case_summary = case_missingness_summary(dataframe)
    pattern_summary = missingness_pattern_summary(dataframe)

    warnings: list[str] = []

    if row_count == 0:
        warnings.append("데이터에 사례가 없습니다.")

    if column_count == 0:
        warnings.append("데이터에 변수가 없습니다.")

    complete_case_count = int(case_summary["complete_case"].sum()) if not case_summary.empty else 0

    if row_count > 0 and complete_case_count == 0:
        warnings.append("완전사례가 없습니다.")

    recommendations = recommend_missingness_strategy(
        variable_summary,
        overall_missing_rate=overall_missing_rate,
    )

    return MissingnessReport(
        row_count=row_count,
        column_count=column_count,
        total_missing_count=total_missing_count,
        overall_missing_rate=overall_missing_rate,
        variable_summary=variable_summary,
        case_summary=case_summary,
        pattern_summary=pattern_summary,
        recommendations=recommendations,
        warnings=warnings,
    )


def recommendations_to_dataframe(
    recommendations: list[MissingnessRecommendation],
) -> pd.DataFrame:
    """추천 목록을 검토용 데이터프레임으로 변환한다."""
    rows: list[dict[str, Any]] = []

    for recommendation in recommendations:
        row = asdict(recommendation)
        row["cautions"] = " | ".join(recommendation.cautions)
        rows.append(row)

    return pd.DataFrame(rows)


def missingness_report_summary(
    report: MissingnessReport,
) -> dict[str, Any]:
    """결측치 보고서 요약을 반환한다."""
    complete_case_count = (
        int(report.case_summary["complete_case"].sum()) if not report.case_summary.empty else 0
    )

    return {
        "row_count": report.row_count,
        "column_count": report.column_count,
        "total_missing_count": report.total_missing_count,
        "overall_missing_rate": report.overall_missing_rate,
        "complete_case_count": complete_case_count,
        "pattern_count": len(report.pattern_summary),
        "recommendation_count": len(report.recommendations),
        "warning_count": len(report.warnings),
    }
