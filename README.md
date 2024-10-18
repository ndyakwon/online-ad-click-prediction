# online-ad-click-prediction

DACON 웹 광고 클릭률 예측 AI 경진대회   
https://dacon.io/competitions/official/236258/overview/description

**1) 분석 목적**

- 로그 데이터를 활용하여 웹 광고 클릭률을 예측하는 모형 개발
- 대용량, 클래스 불균형, 고차원 등 웹 로그 데이터가 가지는 특수성을 고려

**2) 분석 과정**

- 대용량 데이터 로드를 위해 polars 라이브러리 활용하고 메모리 최적화 실시
- 시간 순서를 보존한 35% 샘플 구성 (Systematic + Under Sampling)
- 피처 중요도 기준으로 하위 5개 피처 제외

**3) 모형**

- 속도 측면에 큰 이점이 있는 LightGBM 모형 선택
- 시계열 구조를 유지하는 Time-Series Split Cross-Validation
- Optuna로 하이퍼파라미터 튜닝 (evaluation metric : CV AUC score)

**4) 분석 결과**

- 35% 샘플 이용하여 AUC 0.7812 로 양호한 성능의 예측값 도출
- DACON 웹 광고 클릭률 예측 AI 경진대회 LB 상위 10% 랭크
- 하드웨어가 제한적인 상황에서 대용량 데이터 처리를 위한 다양한 방법을 시도