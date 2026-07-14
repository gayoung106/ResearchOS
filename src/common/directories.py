"""프로젝트 디렉터리 생성 및 점검 모듈."""

from pathlib import Path

from src.common.paths import (
    CONFIG_DIR,
    LOG_DIR,
    MANUSCRIPT_DIR,
    QUESTIONNAIRE_DIR,
    RAW_DATA_DIR,
    RESULT_DIR,
    SRC_DIR,
    TESTS_DIR,
)

BASE_DIRECTORIES: tuple[Path, ...] = (
    RAW_DATA_DIR,
    QUESTIONNAIRE_DIR,
    CONFIG_DIR,
    SRC_DIR,
    RESULT_DIR,
    LOG_DIR,
    MANUSCRIPT_DIR,
    TESTS_DIR,
)

RESULT_DIRECTORIES: tuple[str, ...] = (
    "01_conversion",
    "02_diagnostics",
    "03_preprocessing",
    "04_variables",
    "05_descriptive",
    "06_measurement",
    "07_models",
    "08_robustness",
    "09_figures",
    "10_tables",
    "11_review",
)


def create_base_directories() -> None:
    """프로젝트 기본 디렉터리를 생성한다."""
    for directory in BASE_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def create_result_directories() -> None:
    """분석 단계별 결과 디렉터리를 생성한다."""
    for directory_name in RESULT_DIRECTORIES:
        directory = RESULT_DIR / directory_name
        directory.mkdir(parents=True, exist_ok=True)


def initialize_project_directories() -> None:
    """프로젝트에 필요한 모든 디렉터리를 생성한다."""
    create_base_directories()
    create_result_directories()
