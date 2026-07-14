"""?곌뎄 ?곗씠?곗? 遺꾩꽍 寃곌낵瑜??ㅼ뼇???뺤떇?쇰줈 ??ν븯??紐⑤뱢."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyreadstat

SUPPORTED_OUTPUT_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".sav",
    ".dta",
    ".parquet",
    ".json",
}


def write_data_file(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    *,
    column_labels: dict[str, str] | None = None,
    variable_value_labels: dict[str, dict[Any, str]] | None = None,
    index: bool = False,
) -> Path:
    """?뺤옣?먮? 湲곗??쇰줈 ?곗씠?고봽?덉엫????ν븳??"""
    output = Path(output_path).expanduser().resolve()
    suffix = output.suffix.lower()

    if suffix not in SUPPORTED_OUTPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_EXTENSIONS))
        raise ValueError(f"吏?먰븯吏 ?딅뒗 異쒕젰 ?뺤떇?낅땲?? {suffix}. 吏???뺤떇: {supported}")

    output.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".csv":
        dataframe.to_csv(output, index=index, encoding="utf-8-sig")
    elif suffix == ".xlsx":
        dataframe.to_excel(output, index=index, engine="xlsxwriter")
    elif suffix == ".sav":
        pyreadstat.write_sav(
            dataframe,
            output,
            column_labels=column_labels,
            variable_value_labels=variable_value_labels,
        )
    elif suffix == ".dta":
        dataframe.to_stata(output, write_index=index)
    elif suffix == ".parquet":
        dataframe.to_parquet(output, index=index)
    elif suffix == ".json":
        dataframe.to_json(
            output,
            orient="records",
            force_ascii=False,
            indent=2,
        )

    return output


def write_excel_sheets(
    sheets: dict[str, pd.DataFrame],
    output_path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """?щ윭 ?곗씠?고봽?덉엫???섎굹??Excel ?뚯씪????ν븳??"""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() != ".xlsx":
        raise ValueError("?ㅼ쨷 ?쒗듃 異쒕젰 ?뚯씪? .xlsx ?뺤떇?댁뼱???⑸땲??")

    used_names: set[str] = set()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sequence, (sheet_name, dataframe) in enumerate(
            sheets.items(),
            start=1,
        ):
            safe_name = _make_safe_sheet_name(
                sheet_name,
                used_names,
                fallback=f"sheet_{sequence}",
            )
            dataframe.to_excel(
                writer,
                sheet_name=safe_name,
                index=index,
            )
            used_names.add(safe_name)

    return output


def _make_safe_sheet_name(
    name: str,
    used_names: set[str],
    *,
    fallback: str,
) -> str:
    """Excel ?쒖빟??留욌뒗 以묐났 ?녿뒗 ?쒗듃紐낆쓣 ?앹꽦?쒕떎."""
    invalid_characters = set("[]:*?/\\")
    cleaned = "".join("_" if char in invalid_characters else char for char in name)
    cleaned = cleaned.strip()[:31] or fallback

    candidate = cleaned
    sequence = 1

    while candidate in used_names:
        suffix = f"_{sequence}"
        candidate = f"{cleaned[: 31 - len(suffix)]}{suffix}"
        sequence += 1

    return candidate
