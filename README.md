3-1 AI_Programing Class
AI Competition Participation Task


### 로그 기반 보안 위험도 예측 AI 경진대회 참여 프로젝트

#### 1. 프로젝트 개요

- **대회명**: 로그 분석을 통한 보안 위험도 예측 AI 경진대회 (Dacon)
- **역할**: 개인 참여, 데이터 전처리·특징 추출·모델링 전 과정 단독 수행
- **최종 성능**: Public 0.8138 / Private 0.8206 (VotingClassifier 앙상블 기준)
- **핵심 문제**: 보안 로그(`full_log`)의 위험도 레벨(`level`)을 다중 클래스로 분류


#### 2. 버전별 특징 비교

| 항목 | 실험판 (`experiment_a_preprocessing_with_scaling.py`) | 최종 제출판 (`experiment_b_ensemble_final.py`) |
|---|---|---|
| 결측치 처리 | `fillna(0)` 직접 호출 | `SimpleImputer(strategy='constant', fill_value=0)` — sklearn 파이프라인 호환 |
| 다중공선성 제거 | VIF 기반 컬럼 제거 포함 | 미적용 (트리 모델의 다중공선성 내성을 근거로 제외) |
| 스케일링 | `StandardScaler` 적용 | 미적용 (트리 기반 모델은 스케일에 민감하지 않음) |
| 하이퍼파라미터 탐색 | `GridSearchCV`에 값 1개짜리 그리드 전달 (`{'n_estimators': [100]}`) — 형식만 그리드서치, 실질적 탐색 없음 | LightGBM·XGBoost에 실제 다중 값 그리드 적용, RandomForest는 기본값 유지 |
| 학습/검증 분리 | 리샘플링된 전체 데이터로 바로 학습 (홀드아웃 없음) | `train_test_split`으로 80/20 분리 후 학습 |
| 검증 평가 | `accuracy_score`, `f1_score` 직접 출력 | validation 예측값만 저장 (정량 평가 코드 없음 — 개선 필요) |
| 결과 | Public 0.55 / Private 0.75 / Private(7레벨 실험) 0.77 | **Public 0.8138 / Private 0.8206** |

**핵심 변경 이유**: experiment_a_preprocessing_with_scaling.py에서 다중공선성 제거와 스케일링을 추가했을 때 오히려 성능이 크게 하락했습니다. 원인은 (1) TF-IDF로 생성된 피처가 다중공선성 제거 과정에서 함께 삭제되어 텍스트 문맥 정보가 손실됐고, (2) RandomForest·LightGBM·XGBoost는 원래 다중공선성과 스케일에 내성이 있는 모델이라 해당 전처리가 불필요한 정보 손실만 유발했기 때문입니다. 이 비교 실험을 통해 "이론적으로 권장되는 전처리가 항상 성능 향상으로 이어지지는 않는다"는 결론을 얻었고, v2에서는 이 두 단계를 제거했습니다.

#### 3. 남아있는 개선 포인트 (자체 검토)

- v1의 GridSearchCV는 값이 1개뿐이라 실질적인 탐색 효과가 없음 — 실제로는 고정 파라미터 학습과 동일. 다음 버전에서는 그리드서치를 쓸지, 고정 파라미터로 명시할지 정리 필요.
- v2는 SMOTE를 `train_test_split` 이전에 적용해, 합성 샘플이 학습/검증 양쪽에 걸쳐 섞일 가능성 존재 (v1도 동일). 다음 버전에서는 분리 후 SMOTE를 적용하는 구조로 개선 예정.
- v2는 validation 성능을 정량적으로 출력하는 코드가 없어, 제출 전 로컬 검증 지표를 확인할 수 없음. `accuracy_score`/`f1_score` 출력 추가 권장.
