"""Automatic rawdata discovery and loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.file_metadata import build_variable_metadata
from src.common.file_reader import ReadResult, find_data_files, read_data_file
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult

_EXCEL_EXTENSIONS = {".xlsx", ".xls"}
_DATA_EXTENSIONS = {".csv", ".txt", ".xlsx", ".xls", ".sav", ".dta", ".sas7bdat", ".parquet", ".json"}
_ID_NAME_KEYWORDS = {"id", "caseid", "case_id", "person_id", "participant_id", "respondent_id", "subject_id", "student_id", "user_id", "uid", "pid", "아이디", "식별", "응답자"}


@dataclass(slots=True)
class RawDatasetCandidate:
    source_path: Path
    file_type: str
    sheet_name: str | int | None
    row_count: int
    column_count: int
    non_missing_cell_count: int
    missing_rate: float
    duplicate_row_count: int
    score: float
    warnings: list[str] = field(default_factory=list)

    @property
    def source_label(self) -> str:
        if self.sheet_name is None:
            return str(self.source_path)
        return f"{self.source_path}::{self.sheet_name}"


@dataclass(slots=True)
class AutoRawDataLoadResult:
    dataframe: pd.DataFrame
    selected_candidate: RawDatasetCandidate
    candidates: list[RawDatasetCandidate]
    read_result: ReadResult
    variable_metadata: pd.DataFrame
    metadata_files: list[Path] = field(default_factory=list)
    merge_key: str | None = None
    merged_candidate_labels: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_METADATA_FILE_NAME_KEYWORDS = {
    "codebook",
    "code_book",
    "dictionary",
    "data_dictionary",
    "questionnaire",
    "survey_questions",
    "metadata",
    "변수",
    "코드북",
    "설문",
    "문항",
}
_VARIABLE_NAME_ALIASES = {
    "variable_name",
    "variable",
    "varname",
    "var_name",
    "name",
    "column",
    "column_name",
    "field",
    "item",
    "변수명",
    "변수",
    "문항번호",
}
_METADATA_COLUMN_ALIASES = {
    "variable_label": {
        "variable_label",
        "label",
        "description",
        "desc",
        "변수라벨",
        "변수명_한글",
        "한글명",
        "설명",
    },
    "label": {"label", "라벨", "표시명"},
    "korean_name": {"korean_name", "korean", "한글명", "변수명_한글"},
    "question_text": {"question_text", "question", "item_text", "문항", "문항내용", "질문", "설문문항"},
    "questionnaire_text": {"questionnaire_text", "questionnaire", "survey_question", "설문지", "설문문항"},
    "role_hint": {"role", "role_hint", "역할", "변수역할"},
    "measurement_level_hint": {"measurement_level", "level", "type", "측정수준", "변수유형", "척도"},
    "codebook_note": {"note", "notes", "codebook_note", "비고", "메모"},
}


def discover_metadata_files(
    working_directory: str | Path,
    *,
    codebook_dir: str | Path = "codebook",
    questionnaire_dir: str | Path = "questionnaire",
) -> list[Path]:
    root = Path(working_directory).expanduser().resolve()
    directories = [Path(codebook_dir), Path(questionnaire_dir)]
    files: list[Path] = []
    for directory in directories:
        path = directory if directory.is_absolute() else root / directory
        if path.exists() and path.is_dir():
            files.extend(find_data_files(path))

    rawdata_path = root / "rawdata"
    if rawdata_path.exists() and rawdata_path.is_dir():
        for path in find_data_files(rawdata_path):
            lowered = _normalize_metadata_key(path.stem)
            if any(keyword in lowered for keyword in _METADATA_FILE_NAME_KEYWORDS):
                files.append(path)

    if root.exists() and root.is_dir():
        for path in root.iterdir():
            if not path.is_file() or path.suffix.lower() not in _DATA_EXTENSIONS:
                continue
            lowered = _normalize_metadata_key(path.stem)
            if any(keyword in lowered for keyword in _METADATA_FILE_NAME_KEYWORDS):
                files.append(path)

    return sorted(dict.fromkeys(files))


def _normalize_metadata_key(value: object) -> str:
    return "_".join(str(value).strip().lower().replace("-", "_").split())


def _find_column(dataframe: pd.DataFrame, aliases: set[str]) -> str | None:
    alias_keys = {_normalize_metadata_key(alias) for alias in aliases}
    for column in dataframe.columns:
        if _normalize_metadata_key(column) in alias_keys:
            return str(column)
    return None


def _read_metadata_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in _EXCEL_EXTENSIONS:
        frames: list[pd.DataFrame] = []
        excel = pd.ExcelFile(path)
        for sheet_name in excel.sheet_names:
            frame = read_data_file(path, sheet_name=sheet_name).dataframe
            if not frame.empty:
                frame = frame.copy()
                frame["metadata_sheet_name"] = str(sheet_name)
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return read_data_file(path).dataframe


def _metadata_rows_from_file(path: Path) -> list[dict[str, Any]]:
    dataframe = _read_metadata_file(path)
    if dataframe.empty:
        return []
    variable_column = _find_column(dataframe, _VARIABLE_NAME_ALIASES)
    if variable_column is None:
        return []

    column_map = {
        target: _find_column(dataframe, aliases)
        for target, aliases in _METADATA_COLUMN_ALIASES.items()
    }
    rows: list[dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        raw_name = row.get(variable_column)
        if pd.isna(raw_name) or not str(raw_name).strip():
            continue
        item: dict[str, Any] = {
            "variable_name": str(raw_name).strip(),
            "metadata_source_file": str(path),
        }
        sheet_name = row.get("metadata_sheet_name")
        if pd.notna(sheet_name) and str(sheet_name).strip():
            item["metadata_sheet_name"] = str(sheet_name).strip()
        for target, source_column in column_map.items():
            if source_column is None:
                continue
            value = row.get(source_column)
            if pd.notna(value) and str(value).strip():
                item[target] = str(value).strip()
        rows.append(item)
    return rows


def _merge_metadata_rows(base_metadata: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows or base_metadata.empty or "variable_name" not in base_metadata.columns:
        return base_metadata
    output = base_metadata.copy()
    for column in [
        "label",
        "korean_name",
        "question_text",
        "questionnaire_text",
        "role_hint",
        "measurement_level_hint",
        "codebook_note",
        "metadata_source_files",
    ]:
        if column not in output.columns:
            output[column] = None

    index_by_normalized_name = {
        _normalize_metadata_key(variable_name): index
        for index, variable_name in output["variable_name"].items()
    }
    for row in rows:
        normalized_name = _normalize_metadata_key(row["variable_name"])
        if normalized_name not in index_by_normalized_name:
            continue
        index = index_by_normalized_name[normalized_name]
        source_file = row.get("metadata_source_file")
        if source_file:
            existing = output.at[index, "metadata_source_files"]
            sources = [] if pd.isna(existing) or not str(existing).strip() else str(existing).split(" | ")
            if str(source_file) not in sources:
                sources.append(str(source_file))
            output.at[index, "metadata_source_files"] = " | ".join(sources)
        for column in [
            "variable_label",
            "label",
            "korean_name",
            "question_text",
            "questionnaire_text",
            "role_hint",
            "measurement_level_hint",
            "codebook_note",
        ]:
            value = row.get(column)
            if value is None or not str(value).strip():
                continue
            existing = output.at[index, column] if column in output.columns else None
            if pd.isna(existing) or not str(existing).strip():
                output.at[index, column] = value
    return output


def enrich_variable_metadata_from_files(
    variable_metadata: pd.DataFrame,
    metadata_files: list[Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in metadata_files:
        rows.extend(_metadata_rows_from_file(path))
    return _merge_metadata_rows(variable_metadata, rows)



def _is_metadata_named_file(path: Path) -> bool:
    lowered = _normalize_metadata_key(path.stem)
    return any(keyword in lowered for keyword in _METADATA_FILE_NAME_KEYWORDS)

def discover_rawdata_files(
    working_directory: str | Path,
    *,
    rawdata_dir: str | Path = "rawdata",
) -> list[Path]:
    root = Path(working_directory).expanduser().resolve()
    raw_path = Path(rawdata_dir)
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    return [path for path in find_data_files(raw_path) if not _is_metadata_named_file(path)]


def _candidate_score(dataframe: pd.DataFrame) -> float:
    if dataframe.empty or len(dataframe.columns) == 0:
        return float("-inf")
    non_missing = int(dataframe.notna().sum().sum())
    missing_rate = float(dataframe.isna().mean().mean())
    duplicate_count = int(dataframe.duplicated().sum())
    usable_columns = sum(dataframe[column].nunique(dropna=True) > 1 for column in dataframe.columns)
    base_score = non_missing + len(dataframe) * 10 + len(dataframe.columns) * 5 + usable_columns * 25
    return float(base_score - missing_rate * 100.0 - duplicate_count * 2.0)


def _summarize_candidate(
    read_result: ReadResult,
    *,
    sheet_name: str | int | None = None,
    warnings: list[str] | None = None,
) -> RawDatasetCandidate:
    dataframe = read_result.dataframe
    cell_count = max(int(dataframe.shape[0] * dataframe.shape[1]), 1)
    non_missing = int(dataframe.notna().sum().sum())
    return RawDatasetCandidate(
        source_path=read_result.source_path,
        file_type=read_result.file_type,
        sheet_name=sheet_name,
        row_count=int(len(dataframe)),
        column_count=int(len(dataframe.columns)),
        non_missing_cell_count=non_missing,
        missing_rate=float(1.0 - non_missing / cell_count),
        duplicate_row_count=int(dataframe.duplicated().sum()),
        score=_candidate_score(dataframe),
        warnings=list(warnings or []),
    )


def _read_excel_candidates(path: Path) -> list[tuple[ReadResult, RawDatasetCandidate]]:
    output: list[tuple[ReadResult, RawDatasetCandidate]] = []
    excel = pd.ExcelFile(path)
    for sheet_name in excel.sheet_names:
        read_result = read_data_file(path, sheet_name=sheet_name)
        candidate = _summarize_candidate(read_result, sheet_name=sheet_name)
        output.append((read_result, candidate))
    return output


def _read_candidates(path: Path) -> list[tuple[ReadResult, RawDatasetCandidate]]:
    if path.suffix.lower() in _EXCEL_EXTENSIONS:
        return _read_excel_candidates(path)
    read_result = read_data_file(path)
    return [(read_result, _summarize_candidate(read_result))]


def _rank_candidates(candidates: list[tuple[ReadResult, RawDatasetCandidate]]) -> tuple[ReadResult, RawDatasetCandidate]:
    usable = [item for item in candidates if item[1].row_count > 0 and item[1].column_count > 0]
    if not usable:
        raise ValueError("No readable rawdata candidate contains rows and columns.")
    return max(
        usable,
        key=lambda item: (
            item[1].score,
            item[1].row_count,
            item[1].column_count,
            str(item[1].source_path),
            str(item[1].sheet_name),
        ),
    )




def _normalized_column_map(dataframe: pd.DataFrame) -> dict[str, str]:
    return {_normalize_metadata_key(column): str(column) for column in dataframe.columns}


def _candidate_merge_keys(dataframe: pd.DataFrame) -> list[str]:
    keys: list[str] = []
    for column in dataframe.columns:
        normalized = _normalize_metadata_key(column)
        if normalized not in _ID_NAME_KEYWORDS and not normalized.endswith("_id"):
            continue
        series = dataframe[column].dropna()
        if series.empty or series.duplicated().any():
            continue
        keys.append(str(column))
    return keys


def _find_shared_merge_key(base: pd.DataFrame, other: pd.DataFrame) -> tuple[str, str] | None:
    base_keys = _candidate_merge_keys(base)
    other_map = _normalized_column_map(other)
    for base_key in base_keys:
        normalized = _normalize_metadata_key(base_key)
        other_key = other_map.get(normalized)
        if other_key is not None and other_key in _candidate_merge_keys(other):
            return base_key, other_key
    return None


def _merge_read_candidates(
    selected_read_result: ReadResult,
    selected_candidate: RawDatasetCandidate,
    read_candidates: list[tuple[ReadResult, RawDatasetCandidate]],
) -> tuple[ReadResult, RawDatasetCandidate, str | None, list[str], list[str]]:
    base = selected_read_result.dataframe.copy()
    merge_key: str | None = None
    merged_labels: list[str] = []
    warnings: list[str] = []

    for read_result, candidate in read_candidates:
        if candidate.source_label == selected_candidate.source_label:
            continue
        shared_key = _find_shared_merge_key(base, read_result.dataframe)
        if shared_key is None:
            continue
        base_key, other_key = shared_key
        overlap_rate = float(base[base_key].isin(read_result.dataframe[other_key]).mean())
        if overlap_rate < 0.8:
            warnings.append(
                f"Skipped merge candidate {candidate.source_label}: key overlap for {base_key} was {overlap_rate:.2f}."
            )
            continue
        columns_to_add = [column for column in read_result.dataframe.columns if column != other_key]
        if not columns_to_add:
            continue
        suffix = f"_{_normalize_metadata_key(candidate.source_path.stem) or 'merged'}"
        other = read_result.dataframe[[other_key, *columns_to_add]].copy()
        rename_map = {
            column: f"{column}{suffix}"
            for column in columns_to_add
            if column in base.columns
        }
        other = other.rename(columns=rename_map)
        base = base.merge(other, how="left", left_on=base_key, right_on=other_key)
        if other_key != base_key and other_key in base.columns:
            base = base.drop(columns=[other_key])
        merge_key = base_key
        merged_labels.append(candidate.source_label)

    if not merged_labels:
        return selected_read_result, selected_candidate, None, [], warnings

    merged_read_result = ReadResult(
        dataframe=base,
        source_path=selected_read_result.source_path,
        file_type=selected_read_result.file_type,
        metadata=selected_read_result.metadata,
    )
    merged_candidate = _summarize_candidate(
        merged_read_result,
        sheet_name=selected_candidate.sheet_name,
        warnings=selected_candidate.warnings + warnings,
    )
    return merged_read_result, merged_candidate, merge_key, merged_labels, warnings

def load_rawdata_project(
    working_directory: str | Path = ".",
    *,
    rawdata_dir: str | Path = "rawdata",
    source_file: str | Path | None = None,
    auto_merge: bool = True,
    codebook_dir: str | Path = "codebook",
    questionnaire_dir: str | Path = "questionnaire",
) -> AutoRawDataLoadResult:
    root = Path(working_directory).expanduser().resolve()
    warnings: list[str] = []

    if source_file is not None:
        source_path = Path(source_file)
        if not source_path.is_absolute():
            source_path = root / source_path
        files = [source_path]
    else:
        files = discover_rawdata_files(root, rawdata_dir=rawdata_dir)

    if not files:
        raw_path = Path(rawdata_dir)
        if not raw_path.is_absolute():
            raw_path = root / raw_path
        raise FileNotFoundError(f"No supported data files were found in rawdata directory: {raw_path}")

    read_candidates: list[tuple[ReadResult, RawDatasetCandidate]] = []
    for path in files:
        try:
            read_candidates.extend(_read_candidates(path))
        except Exception as error:  # noqa: BLE001 - keep scanning other rawdata candidates.
            warnings.append(f"Skipped unreadable rawdata file {path}: {error}")

    if not read_candidates:
        raise ValueError("No rawdata files could be read successfully. " + "; ".join(warnings))

    selected_read_result, selected_candidate = _rank_candidates(read_candidates)
    merge_key: str | None = None
    merged_candidate_labels: list[str] = []
    if auto_merge and source_file is None:
        selected_read_result, selected_candidate, merge_key, merged_candidate_labels, merge_warnings = _merge_read_candidates(
            selected_read_result,
            selected_candidate,
            read_candidates,
        )
        warnings.extend(merge_warnings)
    variable_metadata = build_variable_metadata(
        selected_read_result.dataframe,
        source_metadata=selected_read_result.metadata,
    )
    metadata_files = discover_metadata_files(
        root,
        codebook_dir=codebook_dir,
        questionnaire_dir=questionnaire_dir,
    )
    variable_metadata = enrich_variable_metadata_from_files(variable_metadata, metadata_files)
    return AutoRawDataLoadResult(
        dataframe=selected_read_result.dataframe,
        selected_candidate=selected_candidate,
        candidates=[candidate for _, candidate in read_candidates],
        read_result=selected_read_result,
        variable_metadata=variable_metadata,
        metadata_files=metadata_files,
        merge_key=merge_key,
        merged_candidate_labels=merged_candidate_labels,
        warnings=warnings,
    )


class AutoRawDataLoadingStep(PipelineStep):
    """Load the best available dataset from rawdata into PipelineRuntime."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        rawdata_dir: str | Path = "rawdata",
        source_file: str | Path | None = None,
        auto_merge: bool = True,
        codebook_dir: str | Path = "codebook",
        questionnaire_dir: str | Path = "questionnaire",
        order: int = 10,
    ) -> None:
        super().__init__(name="01_auto_rawdata_loading", order=order, required=True)
        self.runtime = runtime
        self.rawdata_dir = rawdata_dir
        self.source_file = source_file
        self.auto_merge = auto_merge
        self.codebook_dir = codebook_dir
        self.questionnaire_dir = questionnaire_dir

    def run(self, context: ResearchContext, working_directory: Path) -> StepResult:
        load_result = load_rawdata_project(
            working_directory,
            rawdata_dir=self.rawdata_dir,
            source_file=self.source_file,
            auto_merge=self.auto_merge,
            codebook_dir=self.codebook_dir,
            questionnaire_dir=self.questionnaire_dir,
        )
        self.runtime.dataframe = load_result.dataframe
        self.runtime.variable_metadata = load_result.variable_metadata
        self.runtime.set_artifact("auto_rawdata_load_result", load_result)
        self.runtime.set_artifact("read_result", load_result.read_result)

        output_dir = working_directory / "result" / "01_auto_import"
        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / "analysis_base.parquet"
        metadata_path = output_dir / "variable_metadata.xlsx"
        candidates_path = output_dir / "rawdata_candidates.xlsx"

        load_result.dataframe.to_parquet(parquet_path, index=False)
        load_result.variable_metadata.to_excel(metadata_path, index=False)
        pd.DataFrame(
            [
                {
                    "source_path": str(candidate.source_path),
                    "file_type": candidate.file_type,
                    "sheet_name": candidate.sheet_name,
                    "row_count": candidate.row_count,
                    "column_count": candidate.column_count,
                    "non_missing_cell_count": candidate.non_missing_cell_count,
                    "missing_rate": candidate.missing_rate,
                    "duplicate_row_count": candidate.duplicate_row_count,
                    "score": candidate.score,
                    "selected": candidate.source_label == load_result.selected_candidate.source_label,
                }
                for candidate in load_result.candidates
            ]
        ).to_excel(candidates_path, index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(parquet_path), str(metadata_path), str(candidates_path)],
            warnings=load_result.warnings,
            metadata={
                "source_file": str(load_result.selected_candidate.source_path),
                "sheet_name": load_result.selected_candidate.sheet_name,
                "row_count": load_result.selected_candidate.row_count,
                "column_count": load_result.selected_candidate.column_count,
                "candidate_count": len(load_result.candidates),
                "metadata_file_count": len(load_result.metadata_files),
                "merge_key": load_result.merge_key,
                "merged_candidate_count": len(load_result.merged_candidate_labels),
            },
        )
