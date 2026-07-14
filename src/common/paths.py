"""프로젝트 전역 경로 정의 모듈."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "rawdata"
QUESTIONNAIRE_DIR = PROJECT_ROOT / "questionnaire"
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"
RESULT_DIR = PROJECT_ROOT / "result"
LOG_DIR = RESULT_DIR / "logs"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
TESTS_DIR = PROJECT_ROOT / "tests"


def get_result_dir(step_name: str) -> Path:
    """분석 단계별 결과 디렉터리를 반환한다."""
    if not step_name.strip():
        raise ValueError("step_name은 비어 있을 수 없습니다.")

    return RESULT_DIR / step_name
