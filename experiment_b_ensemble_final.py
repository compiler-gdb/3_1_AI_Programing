'''
0.813767091 public
0.8206124341 private
'''
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

# 데이터 로드
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
validation_sample = pd.read_csv("validation_sample.csv")

# 1. 충돌된 로그 처리
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

# 2. 특징 추출 (TF-IDF 최적화 포함)
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
test, _ = extract_features(test, tfidf_vectorizer)
validation_sample, _ = extract_features(validation_sample, tfidf_vectorizer)

# 3. 데이터 준비
X = train.drop(columns=['full_log', 'level', 'id'], errors='ignore')
y = train['level']
imputer = SimpleImputer(strategy='constant', fill_value=0)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# SMOTE로 데이터 불균형 처리
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 4. 하이퍼파라미터 튜닝 (GridSearch 적용)
def grid_search_model(model, param_grid, X, y):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_weighted', verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# RandomForest
rf_model = RandomForestClassifier(random_state=42)

# LightGBM
lgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 50]}
lgb_model = grid_search_model(LGBMClassifier(random_state=42), lgb_params, X_train, y_train)

# XGBoost
xgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 6]}
xgb_model = grid_search_model(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'), xgb_params, X_train, y_train)

# 5. 앙상블
ensemble_model = VotingClassifier(estimators=[
    ('random_forest', rf_model),
    ('lightgbm', lgb_model),
    ('xgboost', xgb_model)
], voting='soft')

ensemble_model.fit(X_train, y_train)

# Validation 데이터로 모델 성능 확인
validation_features = validation_sample.drop(columns=['full_log', 'id'], errors='ignore')
validation_features = validation_features.reindex(columns=X_train.columns, fill_value=0).fillna(0)
validation_sample['predicted_level'] = ensemble_model.predict(validation_features)

# Test 데이터에서 level 예측 및 제출 파일 생성
test_features = test.drop(columns=['full_log', 'id'], errors='ignore')
test_features = test_features.reindex(columns=X_train.columns, fill_value=0).fillna(0)

test['level'] = ensemble_model.predict(test_features)
submission = test[['id', 'level']]
submission.to_csv('submission.csv', index=False)

print("Test 데이터 기반으로 submission.csv 생성 완료.")