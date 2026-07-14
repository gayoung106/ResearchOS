"""전처리 규칙 적용 인터페이스."""

from __future__ import annotations

import pandas as pd


def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """승인된 전처리 규칙을 적용한다."""
    raise NotImplementedError
