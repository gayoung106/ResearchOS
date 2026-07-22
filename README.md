# ResearchOS

ResearchOS는 회귀 분석을 중심으로 한 연구 분석 파이프라인입니다. 현재 코드는 Builder, Selector, Diagnostics, Effect Size, Reporting, Visualization, Research Audit가 한 흐름에서 함께 작동하도록 구성되어 있습니다.

이 문서는 설치, 기본 사용법, 파이프라인 등록, 지원 모델, 주요 옵션, 결과물, 테스트 방법을 설명합니다.

## 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

설치 후 다음 명령으로 환경을 확인합니다.

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest -q --basetemp C:\tmp\pytest-basetemp-readme -o cache_dir=C:\tmp\pytest-cache-readme
```

## 사용 방법 1: 함수 직접 호출

단일 모델을 dol/li/go 싶을 때는 `fit_regression_by_level`을 사용합니다.

```python
import pandas as pd

from src.reporting.regression import build_regression_publication_report
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


data = pd.read_excel("rawdata/example.xlsx")

result = fit_regression_by_level(
    data,
    dependent_variable="y",
    independent_variables=["x1", "x2"],
    measurement_level="continuous",
    model_type="ols",
)

effects = build_regression_effect_size_report(result)
report = build_regression_publication_report(result, effects)
visual = build_regression_visualizations(result, output_directory="result/figures")

print(result.model_type)
print(result.fit_statistics)
print(report.narrative)
print(visual.output_files)
```

`RegressionResult`의 주요 필드.

- `model_id`: 모델 식별자
- `model_type`: 적합된 모델 유형
- `sample_size`: 분석 표본 수
- `coefficients`: 계수, 표준 오차, 통계량, p-value, 신뢰구간
- `fit_statistics`: 모델별 적합도 통계
- `metadata`: 진단, 보고, 시각화에 필요한 보조 정보
- `warnings`: 해석 전에 검토할 경고

## 사용 방법 2: Pipeline Builder

분석 계획과 변수 정의를 기반으로 전체 흐름을 등록하려면 Builder를 사용합니다.

```python
import pandas as pd

from src.common.config_models import AnalysisPlan, VariableDefinition, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.regression_builder import register_regression_pipeline
from src.pipeline.runtime import PipelineRuntime


data = pd.read_excel("rawdata/example.xlsx")
runtime = PipelineRuntime(dataframe=data)

plan = AnalysisPlan.model_validate(
    {
        "variables": {
            "dependent": ["y"],
            "independent": ["x1", "x2"],
            "controls": ["age", "gender"],
        },
        "analyses": {
            "regression": {
                "enabled": True,
                "options": {"estimator": "ols", "covariance_type": "HC3"},
            },
            "robustness": {"enabled": True},
        },
    }
)

variable_map = VariableMap(
    variables={
        "y": VariableDefinition(role="dependent", measurement_level="continuous"),
        "x1": VariableDefinition(role="independent", measurement_level="continuous"),
        "x2": VariableDefinition(role="independent", measurement_level="continuous"),
        "age": VariableDefinition(role="control", measurement_level="continuous"),
        "gender": VariableDefinition(role="control", measurement_level="nominal"),
    }
)

orchestrator = ResearchOrchestrator(
    context=ResearchContext(project_name="example study"),
    working_directory=".",
)

registration = register_regression_pipeline(
    orchestrator=orchestrator,
    runtime=runtime,
    analysis_plan=plan,
    variable_map=variable_map,
)

run_result = orchestrator.run(rerun_completed=True)
```

등록되는 주요 단계.

- `09_regression_analysis`: 모델 적합, 계수표, 적합도 통계 저장
- `10_regression_diagnostics`: 모델 진단 표 저장
- Effect size: 효과크기 계산
- Reporting: 출판용 표와 결과 서술문 생성
- Visualization: 계수 그림, 잔차 그림, QQ plot 생성
- Audit: 연구 감사 리포트 생성

## 자동 모델 선택

`model_type`을 생략하면 `measurement_level`에 따라 기본 모델이 선택됩니다.

| `measurement_level` | 기본 모델 |
| --- | --- |
| `continuous` | `ols` |
| `binary` | `binary_logit` |
| `ordinal` | `ordered_logit` |
| `nominal` | `multinomial_logit` |
| `count` | `count_regression` |
| `proportion` | `fractional_logit` |

## 지원 모델

### 연속형 결과

| `model_type` | 설명 | 주요 옵션 |
| --- | --- | --- |
| `ols` | OLS 회귀 | `covariance_type`, `add_intercept` |
| `weighted_least_squares` | 가중 최소 제곱 | `weight_variable`, `covariance_type` |
| `robust_regression` | 로버스트 회귀 | `norm`, `maximum_iterations` |
| `regularized_regression` | 규제화 회귀 | `penalty`, `alpha`, `l1_ratio` |
| `quantile_regression` | 분위수 회귀 | `quantile` |
| `boxcox_regression` | 양수 종속 변수 Box-Cox 변환 OLS | `lambda_value`, `boxcox_lambda` |
| `tobit_regression` | 검열 연속형 모델 | `lower_limit`, `upper_limit` |
| `truncated_regression` | 절단 표본 정규 회귀 | `lower_limit`, `upper_limit` |
| `heckman_selection` | 표본 선택 보정 | `selection_variable`, `exclusion_restrictions` |
| `iv_2sls_regression` | 도구 변수 2SLS | `endogenous_variables`, `instrument_variables` |

### 이항, 비율, 범주, count

| 유형 | `model_type` |
| --- | --- |
| 이항 | `binary_logit`, `binary_probit`, `binary_cloglog`, `linear_probability_model`, `modified_poisson`, `log_binomial`, `quasi_binomial` |
| 비율 | `fractional_logit`, `beta_regression` |
| 순서형 | `ordered_logit`, `ordered_probit` |
| 명목형 | `multinomial_logit` |
| count | `poisson`, `negative_binomial`, `generalized_poisson`, `quasi_poisson`, `zero_inflated_poisson`, `zero_inflated_negative_binomial`, `hurdle_poisson`, `hurdle_negative_binomial` |

### GEE

| `model_type` | 설명 |
| --- | --- |
| `gee_gaussian` | 연속형 GEE |
| `gee_logit` | 이항 GEE |
| `gee_poisson` | Poisson GEE |
| `gee_negative_binomial` | 음이항 GEE |
| `gee_gamma` | Gamma GEE |
| `gee_inverse_gaussian` | inverse Gaussian GEE |
| `gee_tweedie` | Tweedie GEE |

```python
result = fit_regression_by_level(
    data,
    dependent_variable="y",
    independent_variables=["x1", "x2"],
    measurement_level="count",
    model_type="gee_negative_binomial",
    group_variable="cluster_id",
    mixed_effects_options={"covariance_structure": "exchangeable"},
)
```

### Mixed Effects / GLMM

| `model_type` | 필요 옵션 |
| --- | --- |
| `mixed_random_intercept` | `group_variable` |
| `mixed_random_slope` | `group_variable`, `random_slope_variable` |
| `mixed_three_level` | `level2_group`, `level3_group` |
| `mixed_binary_logit_random_intercept` | `group_variable` |
| `mixed_binary_logit_random_slope` | `group_variable`, `random_slope_variable` |
| `mixed_binary_logit_three_level` | `level2_group`, `level3_group` |
| `mixed_poisson_random_intercept` | `group_variable` |
| `mixed_poisson_random_slope` | `group_variable`, `random_slope_variable` |
| `mixed_poisson_three_level` | `level2_group`, `level3_group` |
| `mixed_negative_binomial_random_intercept` | `group_variable` |
| `mixed_negative_binomial_random_slope` | `group_variable`, `random_slope_variable` |
| `mixed_negative_binomial_three_level` | `level2_group`, `level3_group` |

### Panel 모델

| `model_type` | 설명 | 주요 옵션 |
| --- | --- | --- |
| `panel_fixed_effects` | entity/time fixed effects | `entity_variable`, `time_variable` |
| `panel_random_effects` | random intercept panel | `entity_variable`, `time_variable` |
| `panel_correlated_random_effects` | Mundlak CRE | `entity_variable`, `time_variable` |
| `panel_between_effects` | between estimator | `entity_variable`, `time_variable` |
| `panel_first_difference` | first difference | `entity_variable`, `time_variable` |
| `panel_pooled_ols` | pooled panel OLS | `entity_variable`, `time_variable` |

```python
result = fit_regression_by_level(
    data,
    dependent_variable="y",
    independent_variables=["x1", "x2"],
    measurement_level="continuous",
    model_type="panel_correlated_random_effects",
    mixed_effects_options={"entity_variable": "person_id", "time_variable": "wave"},
)
```

### Survival 모델

`cox_proportional_hazards`, `stratified_cox`, `clustered_cox`, `left_truncated_cox`, `time_varying_cox`, `cause_specific_cox`, `weibull_ph`, `weibull_aft`, `exponential_aft`, `loglogistic_aft`, `lognormal_aft`, `piecewise_exponential`, `discrete_time_hazard`를 지원합니다.

## Builder 옵션 예시

### Box-Cox

```python
"analyses": {
    "regression": {"enabled": True, "options": {"estimator": "boxcox", "boxcox_lambda": None}},
    "robustness": {"enabled": False},
}
```

종속 변수는 반드시 양수여야 합니다. `boxcox_lambda`가 `None`이면 lambda/reul 데이터에서 추정합니다.

### Tobit

```python
"analyses": {
    "regression": {"enabled": True, "options": {"estimator": "tobit", "lower_limit": 0.0}},
    "robustness": {"enabled": False},
}
```

경계값에서 검열된 연속형 결과에 사용합니다.

### Truncated Regression

```python
"analyses": {
    "regression": {"enabled": True, "options": {"estimator": "truncated", "lower_limit": 0.0}},
    "robustness": {"enabled": False},
}
```

경계 바깥 관측치가 표본에 없는 절단 표본에 사용합니다.

### Panel FE / CRE

```python
"analyses": {
    "regression": {"enabled": True, "options": {"estimator": "panel_cre"}},
    "panel": {"enabled": True, "options": {"entity_variable": "person_id", "time_variable": "wave"}},
    "robustness": {"enabled": False},
}
```

`estimator`를 `panel_fe`, `panel_re`, `panel_cre`, `panel_be`, `panel_fd`, `panel_pooled`로 바꿀 수 있습니다.

## 결과물

- `result/09_models/*_coefficients.xlsx`: 계수표
- `result/09_models/*_fit_statistics.xlsx`: 적합도 통계
- `result/10_diagnostics/<model_id>/*.xlsx`: 진단 표
- Reporting output: 출판용 표와 narrative
- Visualization output: 계수 그림, 잔차 그림, QQ plot
- Audit output: Research Audit 리포트

## 변수 설정

`measurement_level`: `continuous`, `binary`, `ordinal`, `nominal`, `count`, `proportion`을 주로 사용합니다.

`role`: `dependent`, `independent`, `control`, `fixed_effect`, `weight`, `cluster`, `id`, `time`, `strata`를 사용합니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest -q --basetemp C:\tmp\pytest-basetemp-full -o cache_dir=C:\tmp\pytest-cache-full
```

특정 모델 테스트만 실행할 때.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_boxcox_regression.py -q --basetemp C:\tmp\pytest-basetemp-boxcox -o cache_dir=C:\tmp\pytest-cache-boxcox
.\.venv\Scripts\python.exe -m pytest tests\test_panel_fixed_effects.py -q --basetemp C:\tmp\pytest-basetemp-panel -o cache_dir=C:\tmp\pytest-cache-panel
.\.venv\Scripts\python.exe -m pytest tests\test_gee_regression.py -q --basetemp C:\tmp\pytest-basetemp-gee -o cache_dir=C:\tmp\pytest-cache-gee
```

일부 statsmodels 경고는 테스트 통과시 실패가 아닙니다.

## 문제 해결

`ModuleNotFoundError`가 나오면 다시 설치합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

pytest cache 문제가 잍으면 다음처럼 임시 포더를 명시합니다.

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp C:\tmp\pytest-basetemp-local -o cache_dir=C:\tmp\pytest-cache-local
```

GitHub Actions에서 `pip install -e .`가 실패하면 로컬에서 먼저 설치, ruff, pytest/reul 확인합니다.
