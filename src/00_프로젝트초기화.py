"""연구분석 프로젝트의 기본 폴더를 초기화한다."""

from src.common.directories import initialize_project_directories
from src.common.logger import setup_logger
from src.common.paths import PROJECT_ROOT


def main() -> None:
    """프로젝트 디렉터리를 생성하고 결과를 기록한다."""
    logger = setup_logger("project_initialize")

    logger.info("프로젝트 초기화를 시작합니다.")
    logger.info("프로젝트 루트: %s", PROJECT_ROOT)

    initialize_project_directories()

    logger.info("기본 폴더와 분석 결과 폴더 생성이 완료되었습니다.")


if __name__ == "__main__":
    main()
