"""파이프라인 단계 사이에서 데이터와 중간결과를 공유하는 런타임 저장소."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class PipelineRuntime:
    """파이프라인 실행 중 사용하는 메모리 기반 저장소."""

    dataframe: pd.DataFrame | None = None
    variable_metadata: pd.DataFrame | None = None
    detections: list[Any] = field(default_factory=list)
    resolved_levels: list[Any] = field(default_factory=list)
    preprocessing_plan: Any | None = None
    preprocessing_result: Any | None = None
    missingness_report: Any | None = None
    outlier_report: Any | None = None
    scale_definitions: list[Any] = field(default_factory=list)
    scale_records: list[Any] = field(default_factory=list)
    reliability_results: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def require_dataframe(self) -> pd.DataFrame:
        """현재 데이터프레임을 반환하거나 오류를 발생시킨다."""
        if self.dataframe is None:
            raise RuntimeError("파이프라인 런타임에 데이터프레임이 없습니다.")
        return self.dataframe

    def set_artifact(self, key: str, value: Any) -> None:
        """중간 산출물을 저장한다."""
        self.artifacts[key] = value

    def get_artifact(self, key: str) -> Any:
        """저장된 중간 산출물을 반환한다."""
        if key not in self.artifacts:
            raise KeyError(f"런타임 산출물이 없습니다: {key}")
        return self.artifacts[key]
