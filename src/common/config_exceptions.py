"""연구 설정 관련 예외 정의."""


class ConfigError(Exception):
    """연구 설정 처리 중 발생하는 기본 예외."""


class ConfigFileNotFoundError(ConfigError):
    """필수 설정 파일을 찾지 못한 경우."""


class ConfigValidationError(ConfigError):
    """설정값이 연구 규칙에 맞지 않는 경우."""
