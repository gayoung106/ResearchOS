"""Automatic rawdata discovery and loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.common.file_metadata import build_variable_metadata
from src.common.file_reader import ReadResult, find_data_files, read_data_file
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult

_EXCEL_EXTENSIONS = {".xlsx", ".xls"}


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
    warnings: list[str] = field(default_factory=list)


def discover_rawdata_files(
    working_directory: str | Path,
    *,
    rawdata_dir: str | Path = "rawdata",
) -> list[Path]:
    root = Path(working_directory).expanduser().resolve()
    raw_path = Path(rawdata_dir)
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    return find_data_files(raw_path)


def _candidate_score(dataframe: pd.DataFrame) -> float:
    if dataframe.empty or len(dataframe.columns) == 0:
        return float("-inf")
    non_missing = int(dataframe.notna().sum().sum())
    missing_rate = float(dataframe.isna().mean().mean())
    duplicate_count = int(dataframe.duplicated().sum())
    usable_columns = sum(dataframe[column].nunique(dropna=True) > 1 for column in dataframe.columns)
    return float(non_missing + len(dataframe) * 10 + len(dataframe.columns) * 5 + usable_columns * 25)
    - missing_rate * 100.0 - duplicate_count * 2.0


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


def load_rawdata_project(
    working_directory: str | Path = ".",
    *,
    rawdata_dir: str | Path = "rawdata",
    source_file: str | Path | None = None,
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
    variable_metadata = build_variable_metadata(
        selected_read_result.dataframe,
        source_metadata=selected_read_result.metadata,
    )
    return AutoRawDataLoadResult(
        dataframe=selected_read_result.dataframe,
        selected_candidate=selected_candidate,
        candidates=[candidate for _, candidate in read_candidates],
        read_result=selected_read_result,
        variable_metadata=variable_metadata,
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
        order: int = 10,
    ) -> None:
        super().__init__(name="01_auto_rawdata_loading", order=order, required=True)
        self.runtime = runtime
        self.rawdata_dir = rawdata_dir
        self.source_file = source_file

    def run(self, context: ResearchContext, working_directory: Path) -> StepResult:
        load_result = load_rawdata_project(
            working_directory,
            rawdata_dir=self.rawdata_dir,
            source_file=self.source_file,
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
            },
        )
