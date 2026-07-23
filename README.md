# ResearchOS

ResearchOS는 원자료(raw data)를 기준으로 연구 분석 파이프라인을 자동 구성하고 실행하는 Python 프로젝트입니다. Rawdata Loader, Variable Inference, Analysis Plan, Builder, Selector, Diagnostics, Effect Size, Reporting, Visualization, Research Audit이 한 흐름으로 연결됩니다.

가장 중요한 사용 방식은 다음입니다.

1. rawdata 폴더에 CSV 또는 Excel 원자료를 넣습니다.
2. 자동 실행 명령을 실행합니다.
3. result 폴더에서 분석 계획, 모델 결과, 진단, 보고서, 시각화, 감사 리포트를 확인합니다.

## 설치

Windows PowerShell 기준입니다.

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt
    python -m pip install -e .

기본 확인 명령입니다.

    .\.venv\Scripts\python.exe -m ruff check src tests
    .\.venv\Scripts\python.exe -m pytest -q --basetemp C:\tmp\pytest-basetemp-full -o cache_dir=C:\tmp\pytest-cache-full

## 가장 쉬운 사용법: 원자료만 넣고 자동 분석

프로젝트 폴더 안에 rawdata 폴더를 만들고 원자료 파일을 넣습니다.

    python-layout/
      rawdata/
        survey.xlsx

그 다음 실행합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --project-name my_study

이 명령은 다음을 자동으로 수행합니다.

1. rawdata 안에서 읽을 수 있는 데이터 파일을 찾습니다.
2. 가장 분석 가능한 데이터셋 또는 Excel sheet를 선택합니다.
3. 변수의 측정수준과 역할을 추론합니다.
4. 종속변수, 독립변수, 통제변수, 군집변수, 가중치변수 등을 추정합니다.
5. 자동 분석계획과 변수맵을 생성합니다.
6. Builder를 통해 회귀 파이프라인을 등록합니다.
7. Selector가 적절한 통계모델을 선택하거나 지정된 모델을 실행합니다.
8. Diagnostics, Effect Size, Reporting, Visualization, Research Audit을 생성합니다.
9. 최종 요약 리포트와 산출물 manifest를 저장합니다.

## 먼저 계획만 확인하기

처음에는 실제 모델 실행 전에 자동 추론 결과만 확인하는 것이 좋습니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --project-name my_study --plan-only

주요 확인 파일은 다음입니다.

- result/01_auto_import/analysis_base.parquet
- result/02_auto_variables/variable_role_inference.xlsx
- result/02_auto_variables/inferred_variable_map.xlsx
- result/03_auto_plan/auto_analysis_plan.yaml
- result/03_auto_plan/auto_variable_map.yaml
- result/00_auto_run/auto_run_summary.xlsx
- result/00_auto_run/auto_run_report.md
- result/00_auto_run/auto_final_report.md
- result/00_auto_run/auto_validation_report.xlsx
- result/00_auto_run/output_manifest.xlsx

## 특정 파일만 지정하기

rawdata 안에 파일이 여러 개 있을 때 특정 파일만 분석하려면 --source-file을 사용합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --source-file rawdata\survey.xlsx --project-name survey_study

## 여러 rawdata 파일 자동 병합

rawdata 폴더에 여러 데이터 파일이 있고, 파일들이 같은 ID 변수를 공유하면 안전한 경우에만 자동 병합을 시도합니다. 예를 들어 outcomes.csv와 demographics.csv가 모두 person_id를 가지고 있고 각 파일에서 person_id가 중복되지 않으면 main 데이터에 보조 변수를 left join합니다.

자동 병합은 아래 조건을 만족할 때만 수행됩니다.

- 공통 ID 변수명이 있습니다. 예: id, person_id, respondent_id, student_id
- 각 파일에서 ID 값이 중복되지 않습니다.
- base 데이터의 ID 대부분이 보조 파일에 존재합니다.
- 충돌하는 열 이름은 보조 파일명 suffix를 붙여 보존합니다.

자동 병합을 끄고 한 파일만 선택하게 하려면 --no-auto-merge를 사용합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --project-name single_file_study --no-auto-merge

## 코드북과 설문지 파일 함께 사용하기

변수명이 q1, q2처럼 짧거나 의미를 알기 어려운 경우에는 코드북 또는 설문지 파일을 함께 넣는 것이 좋습니다.

    python-layout/
      rawdata/
        survey.xlsx
      codebook/
        survey_codebook.xlsx
      questionnaire/
        questionnaire.xlsx

코드북 또는 설문지 파일에는 아래 열 중 일부가 있으면 됩니다.

| 열 이름 예시 | 의미 |
| --- | --- |
| variable_name, variable, column_name, 변수명 | 원자료의 변수명 |
| variable_label, label, description, 한글명 | 변수 라벨 또는 설명 |
| question_text, question, 문항, 질문 | 설문 문항 텍스트 |
| role, 역할 | dependent, independent, control 같은 역할 힌트 |
| measurement_level, type, 측정수준, 척도 | continuous, binary, ordinal 같은 측정수준 힌트 |
| note, 비고 | 코드북 메모 |

기본 폴더명은 codebook, questionnaire입니다. 다른 폴더명을 쓰려면 아래처럼 지정합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --codebook-dir metadata\codebook --questionnaire-dir metadata\questionnaire --project-name codebook_study --plan-only

이 정보는 result/01_auto_import/variable_metadata.xlsx에 병합되고, 이후 변수 역할 추론과 multi-outcome 후보 선택에 사용됩니다. 특히 role과 measurement_level 힌트가 있으면 자동 추론보다 우선 반영됩니다.

## 자동 추론을 수동으로 보정하기

자동으로 잡힌 종속변수나 독립변수가 마음에 들지 않으면 CLI 옵션으로 직접 지정할 수 있습니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --project-name override_study --dependent-variable final_score --independent-variables baseline_score treatment --control-variables age gender --cluster-variable school_id --weight-variable sample_weight

지원하는 보정 옵션은 다음입니다.

| 옵션 | 의미 |
| --- | --- |
| --dependent-variable | 종속변수 직접 지정 |
| --independent-variables | 독립변수 직접 지정 |
| --control-variables | 통제변수 직접 지정 |
| --cluster-variable | 군집 또는 그룹 변수 지정 |
| --weight-variable | 가중치 변수 지정 |
| --id-variable | 패널 entity/id 변수 지정 |
| --time-variable | 패널 time 변수 지정 |

수동 보정을 적용하면 result/02_auto_variables/overridden_variable_map.xlsx가 생성됩니다.

## 여러 종속변수를 자동으로 분석하기

설문 데이터처럼 결과변수가 여러 개 있을 수 있으면 --multi-outcome을 사용합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --project-name multi_outcome_study --multi-outcome --max-outcomes 3

이 기능은 outcome 후보를 여러 개 찾고, 후보별로 별도의 분석계획과 모델 파이프라인을 만듭니다.

생성되는 주요 파일은 다음입니다.

- result/03_auto_plan/multi_outcome/outcome_candidates.xlsx
- result/03_auto_plan/multi_outcome/outcome_analysis_plans.xlsx
- result/03_auto_plan/multi_outcome/model_id/analysis_plan.yaml
- result/03_auto_plan/multi_outcome/model_id/variable_map.yaml
- result/multi_outcome_runs/model_id/...
- result/00_auto_run/auto_final_report.md
- result/00_auto_run/output_manifest.xlsx

## 강건성 분석 켜기

자동 분석에 강건성 검토를 포함하려면 --enable-robustness를 사용합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --project-name robust_study --enable-robustness

강건성 분석은 모델 종류와 데이터 구조에 따라 가능한 항목만 등록됩니다.

## Python 코드에서 자동 실행하기

CLI 대신 Python 함수로도 실행할 수 있습니다.

    from src.auto.runner import run_auto_rawdata_analysis

    result = run_auto_rawdata_analysis(
        '.',
        project_name='my_study',
        run_analysis=True,
        enable_multi_outcome=True,
        max_outcomes=3,
    )

    print(result.success)
    print(result.output_files)

## 결과 폴더 구조

분석이 끝나면 보통 아래 구조가 생성됩니다.

    result/
      00_auto_run/
        auto_run_summary.xlsx
        auto_run_report.md
        auto_final_report.md
        auto_validation_report.xlsx
        output_manifest.xlsx
      01_auto_import/
        rawdata_candidates.xlsx
        variable_metadata.xlsx
        analysis_base.parquet
      02_auto_variables/
        variable_role_inference.xlsx
        inferred_variable_map.xlsx
        overridden_variable_map.xlsx
      03_auto_plan/
        analysis_plan_summary.xlsx
        auto_analysis_plan.yaml
        auto_variable_map.yaml
        multi_outcome/
      09_models/
        *_coefficients.xlsx
        *_fit_statistics.xlsx
      10_diagnostics/
      13_effect_sizes/
      14_reports/
      15_visualizations/
      16_research_audit/

가장 먼저 볼 파일은 result/00_auto_run/auto_final_report.md입니다. 이 파일에는 원자료 선택, main model, multi-outcome model, 모델 결과, 단계별 상태, 검증 결과, 경고, 산출물 목록이 정리됩니다. 검증 세부표는 result/00_auto_run/auto_validation_report.xlsx, 전체 산출물 위치와 분류는 result/00_auto_run/output_manifest.xlsx에서 확인할 수 있습니다.

## 자동 모델 선택 기준

model_type을 직접 지정하지 않으면 변수 측정수준과 데이터 구조를 기준으로 기본 모델을 선택합니다.

| 측정수준 | 기본 모델 |
| --- | --- |
| continuous | ols |
| binary | binary_logit |
| ordinal | ordered_logit |
| nominal | multinomial_logit |
| count | count_regression |
| proportion | fractional_logit |

## 지원하는 주요 모델

### 연속형 결과

- ols
- weighted_least_squares
- robust_regression
- regularized_regression
- quantile_regression
- boxcox_regression
- tobit_regression
- truncated_regression
- heckman_selection
- iv_2sls_regression

### 이항, 비율, 범주, count

- 이항: binary_logit, binary_probit, binary_cloglog, linear_probability_model, modified_poisson, log_binomial, quasi_binomial
- 비율: fractional_logit, beta_regression
- 순서형: ordered_logit, ordered_probit
- 명목형: multinomial_logit
- count: poisson, negative_binomial, generalized_poisson, quasi_poisson, zero_inflated_poisson, zero_inflated_negative_binomial, hurdle_poisson, hurdle_negative_binomial

### GEE

- gee_gaussian
- gee_logit
- gee_poisson
- gee_negative_binomial
- gee_gamma
- gee_inverse_gaussian
- gee_tweedie

### Mixed Effects / GLMM

- mixed_random_intercept
- mixed_random_slope
- mixed_three_level
- mixed_binary_logit_random_intercept
- mixed_binary_logit_random_slope
- mixed_binary_logit_three_level
- mixed_poisson_random_intercept
- mixed_poisson_random_slope
- mixed_poisson_three_level
- mixed_negative_binomial_random_intercept
- mixed_negative_binomial_random_slope
- mixed_negative_binomial_three_level

### Panel

- panel_fixed_effects
- panel_random_effects
- panel_correlated_random_effects
- panel_between_effects
- panel_first_difference
- panel_pooled_ols

### Survival

- cox_proportional_hazards
- stratified_cox
- clustered_cox
- left_truncated_cox
- time_varying_cox
- cause_specific_cox
- weibull_ph
- weibull_aft
- exponential_aft
- loglogistic_aft
- lognormal_aft
- piecewise_exponential
- discrete_time_hazard

## 직접 모델을 호출하는 방법

자동 실행이 아니라 특정 모델을 코드에서 직접 호출할 수도 있습니다.

    import pandas as pd

    from src.statistics.selector import fit_regression_by_level

    data = pd.read_csv('rawdata/survey.csv')
    result = fit_regression_by_level(
        data,
        dependent_variable='y',
        independent_variables=['x1', 'x2'],
        measurement_level='continuous',
    )
    print(result.model_type)

## 개발 및 테스트

자주 쓰는 확인 명령입니다.

    .\.venv\Scripts\python.exe -m ruff check src tests
    .\.venv\Scripts\python.exe -m pytest tests\test_auto_runner.py tests\test_auto_validation.py -q --basetemp C:\tmp\pytest-basetemp-auto -o cache_dir=C:\tmp\pytest-cache-auto
    .\.venv\Scripts\python.exe -m pytest -q --basetemp C:\tmp\pytest-basetemp-full -o cache_dir=C:\tmp\pytest-cache-full

## 주의사항

- 자동 추론 결과는 --plan-only로 먼저 확인하는 것이 좋습니다.
- 변수명이 불명확하면 코드북 또는 설문지를 함께 제공하세요.
- 종속변수나 주요 독립변수가 잘못 잡히면 CLI 보정 옵션을 사용하세요.
- auto_final_report.md는 가장 먼저 읽는 요약 파일입니다.
- output_manifest.xlsx는 생성된 산출물의 위치, 분류, 추천 여부, 설명을 담습니다.
