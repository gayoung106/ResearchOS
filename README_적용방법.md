# Research Orchestrator 적용

다음 파일을 프로젝트에 복사합니다.

- `src/pipeline/step.py`
- `src/pipeline/registry.py`
- `src/pipeline/orchestrator.py`
- `src/pipeline/builtin_steps.py`
- `tests/test_orchestrator.py`

적용 후 실행:

```powershell
ruff format src tests
ruff check src tests --fix
python -m pytest
```

기존 테스트가 모두 유지되어 있다면 신규 테스트 8개가 추가됩니다.

이 오케스트레이터는 각 단계 실행 전후에 다음 파일을 저장합니다.

- `config/research_context.yaml`
- `result/pipeline_state.json`

현재 버전은 단계 실행 기반을 구현한 1차 버전입니다.
실제 파일변환, 진단, 전처리, 통계 모듈은 다음 단계에서
PipelineStep 구현체로 연결합니다.
