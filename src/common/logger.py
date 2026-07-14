"""분석 실행 로그 설정 모듈."""

import logging
from datetime import datetime
from pathlib import Path

from src.common.paths import LOG_DIR


def setup_logger(
    name: str,
    log_filename: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """콘솔과 파일에 동시에 기록하는 로거를 생성한다."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{name}_{timestamp}.log"

    log_path = LOG_DIR / log_filename
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        filename=log_path,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_log_path(filename: str) -> Path:
    """로그 파일의 전체 경로를 반환한다."""
    if not filename.strip():
        raise ValueError("filename은 비어 있을 수 없습니다.")

    return LOG_DIR / filename
