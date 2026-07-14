"""회귀분석 인터페이스."""

from __future__ import annotations

import pandas as pd


def run_regression_models(dataframe: pd.DataFrame) -> dict[str, object]:
    """연구설계에 따른 회귀모형을 실행한다."""
    raise NotImplementedError
