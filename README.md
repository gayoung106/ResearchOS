# ResearchOS 사용 설명서

ResearchOS는 원자료(raw data)를 넣으면 연구 분석 파이프라인을 자동으로 구성하고 실행하는 Python 프로젝트입니다. 원자료 로딩, 변수 추론, 분석계획 생성, Builder 등록, Selector 기반 모델 선택, 진단, 효과크기, 보고서, 시각화, Research Audit, 최종 리포트까지 한 번에 이어지도록 구성되어 있습니다.

이 문서는 Git clone부터 실제 분석 실행까지 처음 사용하는 사람이 그대로 따라 할 수 있도록 작성했습니다.

## 1. 준비물

Windows PowerShell 기준입니다.

필요한 것:

- Git
- Python 3.11 이상 권장
- 이 저장소에 접근 가능한 GitHub URL
- 분석할 CSV 또는 Excel 원자료

Git 설치 확인:

    git --version

Python 설치 확인:

    python --version

Python 명령이 동작하지 않으면 Windows에서는 아래 명령도 확인합니다.

    py --version

## 2. Git clone으로 프로젝트 받기

먼저 작업할 폴더로 이동합니다. 예시는 바탕화면 아래 작업 폴더입니다.

    cd C:\Users\KMI\Desktop\gayoung

GitHub 저장소를 clone합니다.

    git clone https://github.com/USER/REPOSITORY.git

위 URL은 실제 저장소 주소로 바꿔야 합니다. 예를 들어 저장소 주소가 https://github.com/myname/ResearchOS.git 라면 아래처럼 실행합니다.

    git clone https://github.com/myname/ResearchOS.git

clone이 끝나면 프로젝트 폴더로 이동합니다.

    cd REPOSITORY

현재 폴더에 pyproject.toml, README.md, src, tests 폴더가 보이면 정상입니다.

    dir

이미 clone한 프로젝트를 최신 상태로 갱신하려면 프로젝트 폴더에서 아래 명령을 실행합니다.

    git pull

## 3. 가상환경 만들기

프로젝트마다 독립된 Python 환경을 쓰기 위해 가상환경을 만듭니다.

    python -m venv .venv

python 명령이 없으면 아래처럼 실행합니다.

    py -m venv .venv

가상환경을 활성화합니다.

    .\.venv\Scripts\Activate.ps1

PowerShell 실행 정책 때문에 막히면 현재 터미널 세션에서만 허용한 뒤 다시 활성화합니다.

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

활성화되면 프롬프트 앞에 (.venv)가 표시됩니다.

## 4. 의존성 설치

pip을 먼저 업데이트합니다.

    .\.venv\Scripts\python.exe -m pip install --upgrade pip

실행 의존성을 설치합니다.

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

개발 및 테스트 의존성을 설치합니다.

    .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

현재 프로젝트를 editable 모드로 설치합니다.

    .\.venv\Scripts\python.exe -m pip install -e .

설치가 끝났는지 간단히 확인합니다.

    .\.venv\Scripts\python.exe -m ruff check src tests

전체 테스트는 시간이 걸릴 수 있습니다.

    .\.venv\Scripts\python.exe -m pytest -q --basetemp C:\tmp\pytest-basetemp-full -o cache_dir=C:\tmp\pytest-cache-full

## 5. 원자료 폴더 만들기

프로젝트 루트에 rawdata 폴더를 만듭니다.

    mkdir rawdata

분석할 원자료 파일을 rawdata 폴더에 넣습니다.

예시 구조:

    REPOSITORY/
      rawdata/
        survey.xlsx

CSV도 사용할 수 있습니다.

    REPOSITORY/
      rawdata/
        survey.csv

지원하는 대표 형식:

- csv
- txt
- xlsx
- xls
- sav
- dta
- sas7bdat
- parquet
- json

## 6. 코드북과 설문지 넣기 선택 사항

변수명이 q1, q2처럼 의미를 알기 어렵다면 코드북이나 설문지를 함께 넣는 것이 좋습니다.

기본 폴더명:

    codebook
    questionnaire

예시 구조:

    REPOSITORY/
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

다른 폴더명을 쓰려면 실행할 때 옵션으로 지정합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --codebook-dir metadata\codebook --questionnaire-dir metadata\questionnaire --plan-only

## 7. 먼저 계획만 확인하기 권장

처음에는 모델을 바로 실행하지 말고 plan-only로 자동 추론 결과를 확인하는 것이 좋습니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --project-name my_study --plan-only

이 명령은 다음만 수행합니다.

1. rawdata 파일 선택
2. rawdata 품질 리포트 생성
3. 변수 역할과 측정수준 추론
4. 자동 분석계획 생성
5. 모델 파이프라인 등록
6. 최종 요약 파일 생성

plan-only 후 가장 먼저 볼 파일:

    result\00_auto_run\auto_final_report.md

함께 확인하면 좋은 파일:

    result\01_auto_import\rawdata_quality_report.xlsx
    result\02_auto_variables\variable_role_inference.xlsx
    result\03_auto_plan\auto_analysis_plan.yaml
    result\03_auto_plan\auto_variable_map.yaml
    result\00_auto_run\auto_validation_report.xlsx
    result\00_auto_run\auto_recovery_guide.xlsx
    result\00_auto_run\output_manifest.xlsx

## 8. 전체 자동 분석 실행

plan-only 결과가 괜찮으면 실제 모델 실행까지 진행합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --project-name my_study

전체 실행은 다음을 수행합니다.

1. 원자료 로딩
2. rawdata 품질 점검
3. 변수 추론
4. 분석계획 생성
5. 회귀 파이프라인 등록
6. 모델 실행
7. 진단 산출물 생성
8. 효과크기 산출물 생성
9. 보고서 생성
10. 시각화 생성
11. Research Audit 생성
12. 최종 리포트, 검증 리포트, 복구 가이드, manifest 생성

실행이 끝나면 CLI 출력에서 아래 경로를 먼저 확인합니다.

    Final report: ...\auto_final_report.md
    Output manifest: ...\output_manifest.xlsx
    Recovery guide: ...\auto_recovery_guide.xlsx

## 9. 특정 rawdata 파일만 분석하기

rawdata 폴더에 파일이 여러 개 있을 때 특정 파일만 지정하려면 --source-file을 사용합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --source-file rawdata\survey.xlsx --project-name survey_study

## 10. 여러 rawdata 파일 자동 병합

rawdata 폴더에 여러 파일이 있고 공통 ID 변수가 있으면 안전한 경우에만 자동 병합을 시도합니다.

예시:

    rawdata/
      outcomes.csv
      demographics.csv

자동 병합 조건:

- 공통 ID 변수명이 있음 예: id, person_id, respondent_id, student_id
- 각 파일에서 ID 값이 중복되지 않음
- base 데이터의 ID 대부분이 보조 파일에 존재함
- 충돌하는 열 이름은 보조 파일명 suffix를 붙여 보존함

자동 병합을 끄려면 --no-auto-merge를 사용합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --project-name single_file_study --no-auto-merge

## 11. 변수 역할을 직접 지정하기

자동 추론이 원하는 변수 구성을 고르지 못하면 CLI 옵션으로 직접 지정할 수 있습니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --project-name override_study --dependent-variable final_score --independent-variables baseline_score treatment --control-variables age gender --cluster-variable school_id --weight-variable sample_weight

지원 옵션:

| 옵션 | 의미 |
| --- | --- |
| --dependent-variable | 종속변수 직접 지정 |
| --independent-variables | 독립변수 직접 지정. 여러 개 가능 |
| --control-variables | 통제변수 직접 지정. 여러 개 가능 |
| --cluster-variable | 군집 또는 그룹 변수 지정 |
| --weight-variable | 가중치 변수 지정 |
| --id-variable | 패널 entity/id 변수 지정 |
| --time-variable | 패널 time 변수 지정 |

수동 보정 결과는 아래 파일에 저장됩니다.

    result\02_auto_variables\overridden_variable_map.xlsx

## 12. 여러 종속변수 자동 분석

설문 데이터처럼 outcome 후보가 여러 개 있으면 --multi-outcome을 사용합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --project-name multi_outcome_study --multi-outcome --max-outcomes 3

이 기능은 outcome 후보별로 별도의 분석계획과 모델 파이프라인을 만듭니다.

주요 산출물:

    result\03_auto_plan\multi_outcome\outcome_candidates.xlsx
    result\03_auto_plan\multi_outcome\outcome_analysis_plans.xlsx
    result\03_auto_plan\multi_outcome\MODEL_ID\analysis_plan.yaml
    result\03_auto_plan\multi_outcome\MODEL_ID\variable_map.yaml
    result\multi_outcome_runs\MODEL_ID\...

## 13. 강건성 분석 켜기

가능한 강건성 검토까지 포함하려면 --enable-robustness를 사용합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --project-name robust_study --enable-robustness

강건성 분석은 모델 종류와 데이터 구조에 따라 가능한 항목만 등록됩니다.

## 14. 결과 폴더 구조

대표적인 결과 구조입니다.

    result/
      00_auto_run/
        auto_run_summary.xlsx
        auto_run_report.md
        auto_final_report.md
        auto_validation_report.xlsx
        auto_recovery_guide.xlsx
        output_manifest.xlsx
      01_auto_import/
        rawdata_candidates.xlsx
        variable_metadata.xlsx
        analysis_base.parquet
        rawdata_quality_report.xlsx
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
      10_diagnostics/
      13_effect_sizes/
      14_reports/
      15_visualizations/
      16_research_audit/

## 15. 가장 먼저 볼 파일

1순위:

    result\00_auto_run\auto_final_report.md

이 파일에는 다음이 들어 있습니다.

- 원자료 선택 정보
- 추천 산출물
- rawdata 품질 요약
- main model 요약
- multi-outcome model 요약
- 실제 모델 결과 요약
- 단계별 실행 상태
- Recovery guide
- Next steps
- 검증 요약
- 경고
- 전체 산출물 목록

2순위:

    result\00_auto_run\output_manifest.xlsx

모든 산출물의 위치, 분류, 추천 여부, 설명이 들어 있습니다.

3순위:

    result\00_auto_run\auto_validation_report.xlsx

필수 산출물이 빠졌는지, yaml 파일이 정상인지, 모델 산출물이 필요한 상황에서 생성됐는지 확인합니다.

4순위:

    result\00_auto_run\auto_recovery_guide.xlsx

실패했거나 검증 항목이 부족할 때 우선순위별 조치 방법을 보여줍니다.

5순위:

    result\01_auto_import\rawdata_quality_report.xlsx

결측, 상수열, ID 후보, 날짜 후보, high-cardinality 변수 등을 확인합니다.

## 16. Python 코드에서 실행하기

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

## 17. 자동 모델 선택 기준

model_type을 직접 지정하지 않으면 변수 측정수준과 데이터 구조를 기준으로 기본 모델을 선택합니다.

| 측정수준 | 기본 모델 |
| --- | --- |
| continuous | ols |
| binary | binary_logit |
| ordinal | ordered_logit |
| nominal | multinomial_logit |
| count | count_regression |
| proportion | fractional_logit |

## 18. 지원하는 주요 모델

연속형 결과:

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

이항, 비율, 범주, count:

- binary_logit
- binary_probit
- binary_cloglog
- linear_probability_model
- modified_poisson
- log_binomial
- quasi_binomial
- fractional_logit
- beta_regression
- ordered_logit
- ordered_probit
- multinomial_logit
- poisson
- negative_binomial
- generalized_poisson
- quasi_poisson
- zero_inflated_poisson
- zero_inflated_negative_binomial
- hurdle_poisson
- hurdle_negative_binomial

GEE:

- gee_gaussian
- gee_logit
- gee_poisson
- gee_negative_binomial
- gee_gamma
- gee_inverse_gaussian
- gee_tweedie

Mixed Effects / GLMM:

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

Panel:

- panel_fixed_effects
- panel_random_effects
- panel_correlated_random_effects
- panel_between_effects
- panel_first_difference
- panel_pooled_ols

Survival:

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

## 19. 개발 및 테스트 명령

자동 분석 관련 테스트:

    .\.venv\Scripts\python.exe -m pytest tests\test_auto_cli.py tests\test_auto_runner.py tests\test_auto_rawdata_loader.py tests\test_auto_validation.py -q --basetemp C:\tmp\pytest-basetemp-auto -o cache_dir=C:\tmp\pytest-cache-auto

전체 ruff:

    .\.venv\Scripts\python.exe -m ruff check src tests

전체 pytest:

    .\.venv\Scripts\python.exe -m pytest -q --basetemp C:\tmp\pytest-basetemp-full -o cache_dir=C:\tmp\pytest-cache-full

## 20. 자주 생기는 문제

### pip install -e . 실패

먼저 의존성을 다시 설치합니다.

    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
    .\.venv\Scripts\python.exe -m pip install -e .

### rawdata를 찾지 못함

확인할 것:

- 프로젝트 루트에 rawdata 폴더가 있는지 확인
- rawdata 안에 CSV 또는 Excel 파일이 있는지 확인
- 다른 폴더를 쓴다면 --rawdata-dir 옵션을 지정했는지 확인
- 특정 파일만 분석하려면 --source-file을 사용

### 종속변수가 이상하게 잡힘

먼저 plan-only로 확인합니다.

    .\.venv\Scripts\python.exe -m src.auto.cli --working-directory . --plan-only

그 다음 아래 파일을 확인합니다.

    result\02_auto_variables\variable_role_inference.xlsx
    result\03_auto_plan\auto_analysis_plan.yaml

필요하면 수동 보정 옵션을 사용합니다.

    --dependent-variable
    --independent-variables
    --control-variables

### 결과 파일이 너무 많아서 헷갈림

가장 먼저 아래 두 파일만 보면 됩니다.

    result\00_auto_run\auto_final_report.md
    result\00_auto_run\output_manifest.xlsx

### 테스트는 통과하지만 warning이 뜸

statsmodels에서 나오는 일부 경고는 모델 특성상 정상일 수 있습니다. pytest 결과가 failed가 아니고 passed라면 보통 괜찮습니다.
