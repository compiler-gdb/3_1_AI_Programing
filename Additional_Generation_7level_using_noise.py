import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 데이터 로드
train = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
validation_sample = pd.read_csv("validation_sample.csv")
print("데이터 로드 완료")

# 'id' 컬럼 별도로 저장
test_ids = test_data['id'].copy() if 'id' in test_data.columns else None

# 2. 충돌된 로그 처리
def resolve_conflicts(data):
    pivot = data.pivot_table(index='full_log', columns='level', aggfunc='size', fill_value=0)
    pivot['count'] = (pivot > 0).sum(axis=1)
    conflicting_logs = pivot[pivot['count'] > 1].reset_index()

    def resolve(row):
        levels = data[data['full_log'] == row['full_log']]['level']
        counts = levels.value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            return counts.index.min()
        return counts.idxmax()

    conflicting_logs['resolved_level'] = conflicting_logs.apply(resolve, axis=1)
    resolved_data = data[~data['full_log'].isin(conflicting_logs['full_log'])]
    resolved_logs = conflicting_logs[['full_log', 'resolved_level']].rename(columns={'resolved_level': 'level'})
    return pd.concat([resolved_data, resolved_logs], ignore_index=True)

train = resolve_conflicts(train)
print("충돌된 로그 처리 완료")

# 3. 결측치 처리
train.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)
validation_sample.fillna(0, inplace=True)
print("결측치 제거 완료")

# 4. 특징 추출 (TF-IDF 포함)
def extract_features(data, tfidf=None):
    data['log_length'] = data['full_log'].apply(len)
    data['word_count'] = data['full_log'].apply(lambda x: len(x.split()))
    data['digit_count'] = data['full_log'].apply(lambda x: sum(c.isdigit() for c in x))
    data['special_char_count'] = data['full_log'].apply(lambda x: sum(c in "!@#$%^&*()_+-=~`[]{}|;:',.<>?/" for c in x))

    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(data['full_log'])
    else:
        tfidf_matrix = tfidf.transform(data['full_log'])

    tfidf_features = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
    data = pd.concat([data.reset_index(drop=True), tfidf_features.reset_index(drop=True)], axis=1)
    return data, tfidf

train, tfidf_vectorizer = extract_features(train)
test_data, _ = extract_features(test_data, tfidf_vectorizer)
validation_sample, _ = extract_features(validation_sample, tfidf_vectorizer)
print("TF-IDF 특징 추출 완료")

# 5. 데이터 준비
full_log = train['full_log']
y = train['level']
X = train.drop(columns=['full_log', 'level', 'id'], errors='ignore')

# 다중공선성 제거 대상 컬럼 리스트
removed_features = [...]  # 제공된 리스트 그대로 유지

# 제거된 피처 삭제
X = X.drop(columns=removed_features, errors='ignore')
test_data = test_data.drop(columns=removed_features, errors='ignore')
validation_sample = validation_sample.drop(columns=removed_features, errors='ignore')

# 컬럼 순서 동기화
test_data = test_data.reindex(columns=X.columns, fill_value=0)
validation_sample = validation_sample.reindex(columns=X.columns, fill_value=0)
print("데이터셋 동기화 완료")

# 6. 스케일링
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
test_data_scaled = pd.DataFrame(scaler.transform(test_data), columns=X.columns)
validation_sample_scaled = pd.DataFrame(scaler.transform(validation_sample), columns=X.columns)
print("스케일링 완료")

# 7. SMOTE 적용
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("SMOTE 적용 완료")

# 8. 모델 학습 및 예측
def grid_search_model(model, param_grid, X, y):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_weighted', verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

rf_model = grid_search_model(RandomForestClassifier(random_state=42), {'n_estimators': [100]}, X_resampled, y_resampled)
lgb_model = grid_search_model(LGBMClassifier(random_state=42), {'n_estimators': [100], 'learning_rate': [0.1]}, X_resampled, y_resampled)
xgb_model = grid_search_model(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'), {'n_estimators': [100]}, X_resampled, y_resampled)

ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('lgb', lgb_model), ('xgb', xgb_model)], voting='soft')
ensemble_model.fit(X_resampled, y_resampled)

# 9. Validation 데이터 평가
if 'level' in validation_sample.columns:
    validation_predictions = ensemble_model.predict(validation_sample_scaled)
    validation_accuracy = accuracy_score(validation_sample['level'], validation_predictions)
    validation_f1 = f1_score(validation_sample['level'], validation_predictions, average='weighted')

    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    print(f"Validation F1 Score: {validation_f1:.4f}")
else:
    print("Validation 데이터에 'level' 컬럼이 없어 평가를 생략합니다.")

# 10. Test 데이터 예측
if test_ids is not None:
    test_predictions = ensemble_model.predict(test_data_scaled)
    submission = pd.DataFrame({'id': test_ids, 'level': test_predictions})
    submission.to_csv('submission.csv', index=False)
    print("Submission 파일 생성 완료")
else:
    print("Test 데이터에 'id' 컬럼이 없어 제출 파일을 생성할 수 없습니다.")