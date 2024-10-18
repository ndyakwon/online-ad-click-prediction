import pandas as pd
import numpy as np
import polars as pl
import os, gc, time, math, joblib

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import category_encoders as ce
import optuna

import warnings
warnings.filterwarnings("ignore")


# org
train = pl.read_csv("./dacon_weblog/train.csv")
train = train.to_pandas()
train = reduce_mem_usage(train)

# sample ver1
train_sample_sys = pl.read_parquet("./dacon_weblog/train_sample_sys.parquet")
train = train_sample_sys.to_pandas()
train = reduce_mem_usage(train)
train.shape

# sample ver2
train_sample_srs = pl.read_parquet("./dacon_weblog/train_sample_srs.parquet")
train = train_sample_srs.to_pandas()
train = reduce_mem_usage(train)
train.shape

# sample ver3
train_sample_bal = pl.read_parquet("./dacon_weblog/train_sample_down.parquet")
train = train_sample_bal.to_pandas()
train = reduce_mem_usage(train)
train.shape

test = pl.read_csv("./dacon_weblog/test.csv")
test = test.to_pandas()
test = reduce_mem_usage(test)
test.shape

# train.isnull().sum()
# train.Click.value_counts()


X = train.drop(columns=['ID', 'Click'])
y = train['Click']
X_test = test.drop(columns=['ID'])

X = X.drop(columns=['F22', 'F23', 'F30', 'F35', 'F38'])
X_test = X_test.drop(columns=['F22', 'F23', 'F30', 'F35', 'F38'])

numerical = []
categorical = []

for col in X.columns:
    if X[col].dtype == 'category' or X[col].dtype == 'object':
        categorical.append(col)
    else:
        numerical.append(col)

print("numerical features = ",numerical)
print("\ncategorical features = ",categorical)

for col in categorical:
    X[col] = X[col].astype('category').cat.add_categories(['Null'])
    X[col].fillna('Null', inplace=True)

for col in numerical:
    X[col].fillna(0, inplace=True)

for col in categorical:
    X_test[col] = X_test[col].astype('category').cat.add_categories(['Null'])
    X_test[col].fillna('Null', inplace=True)

for col in numerical:
    X_test[col].fillna(0, inplace=True)


# encoding_target = list(X.dtypes[X.dtypes == "category"].index)
# encoding_target
#
# enc = ce.CountEncoder(cols=encoding_target).fit(X, y)
# X_enc = enc.transform(X)
# X_test_enc = enc.transform(X_test)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

start_time = time.time()
model_lgb = LGBMClassifier(random_state=42, metric='auc', n_estimators=1000)
model_lgb.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100)])

lgb_proba = model_lgb.predict_proba(X_val)[:, 1]
lgb_auc_score = roc_auc_score(y_val, lgb_proba)
print(lgb_auc_score) # 0.7651 - 0.7735 (sys) / 0.7642 (srs) / 0.7634 (bal) / 0.7627 - 0.7794 (downsys)
end_time = time.time()
print(f"Time taken to read : {end_time - start_time} seconds") # 45.13 sec

feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': model_lgb.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)

new_features = sorted(feature_importance[:25].Feature)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.title('Feature Importance - LGBMClassifier')
# plt.gca().invert_yaxis()
# plt.show()

# count encoding : 0.7403 (sys) / 0.7400 (srs) / 0.7402 (bal)
#
# start_time = time.time()
# model_cat = CatBoostClassifier(random_state=42, cat_features=categorical)
# # model_cat = CatBoostClassifier(random_state=42, cat_features=categorical, task_type='GPU')
# model_cat.fit(X_train, y_train,
#               eval_set=[(X_val, y_val)],
#               early_stopping_rounds=50,
#               verbose=True)
#
# cat_proba = model_cat.predict_proba(X_val)[:, 1]
# cat_auc_score = roc_auc_score(y_val, cat_proba)
# print(cat_auc_score) # 0.7803 (sys) / 0.7799 (srs) / 0.7876 (bal)
# end_time = time.time()
# print(f"Time taken to read : {end_time - start_time} seconds") # 33507 sec



cv = TimeSeriesSplit(n_splits=3)

# preds = np.zeros(X_test.shape[0])
scores = []

for fold, (idx_train, idx_valid) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[idx_train], X.iloc[idx_valid]
    y_train, y_val = y.iloc[idx_train], y.iloc[idx_valid]

    model = LGBMClassifier(random_state=42, metric='auc', n_estimators=1000)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100)])

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    fold_score = roc_auc_score(y_val, y_pred_proba)
    scores.append(fold_score)
    gc.collect()

    print(f"Fold {fold + 1} AUC: {fold_score:.4f}")
    # preds += model.predict_proba(X_test)[:, 1] / cv.n_splits

print(f"\nOverall Validation Score: {np.mean(scores)}")

# 3fold default : 0.7609 / 7fold default : 0.7612
# 3fold 1000 auc : 0.7658 / 7fold 1000 auc : 0.7653
preds


def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 30, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        # 'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        # 'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        # 'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        # 'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'random_state': 42
    }

    # model = lgb.LGBMClassifier(**params)
    # model.fit(X_train, y_train,
    #           eval_set=[(X_val, y_val)],
    #           callbacks=[lgb.early_stopping(50)])
    #
    # y_pred_proba = model.predict_proba(X_val)[:, 1]
    # score = roc_auc_score(y_val, y_pred_proba)

    scores = []

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[idx_train], X.iloc[idx_valid]
        y_train, y_val = y.iloc[idx_train], y.iloc[idx_valid]

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50)])

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        fold_score = roc_auc_score(y_val, y_pred_proba)
        scores.append(fold_score)
        gc.collect()

    cv_score = np.mean(scores)

    return cv_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

### Evaluation
trial = study.best_trial
print(f'Best trial: {trial}')
print('  Value: {:.4f}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

best_params = study.best_params
best_model = LGBMClassifier(**best_params, random_state=42, bjective='binary', metric='auc', verbose=-1)
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(100)],
)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)

print("Best hyperparameters: ", best_params)
print("Best AUC: ", auc)

# Best hyperparameters:  {'num_leaves': 280, 'max_depth': 20, 'learning_rate': 0.017415711730705302, 'n_estimators': 1000, 'min_child_samples': 13}
# Best AUC:  0.7746813747237086

# Best hyperparameters:  {'num_leaves': 287, 'max_depth': 20, 'learning_rate': 0.01184565730088221, 'n_estimators': 1700, 'min_child_samples': 79, 'subsample': 0.7999735191831004, 'colsample_bytree': 0.6320646580737812, 'reg_alpha': 5.316185711001052, 'reg_lambda': 8.332904785664715}
# Best AUC:  0.7761515041101867

# Best parameters: {'num_leaves': 110, 'max_depth': 20, 'learning_rate': 0.014557869682012175, 'n_estimators': 1457, 'min_child_samples': 85}
# value: 0.766812818113539

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 110,
    "max_depth": 20,
    "learning_rate": 0.014557869682012175,
    "n_estimators": 1457,
    "min_child_samples": 85,
    "verbose": -1,
    "random_state": 42,
    # "scale_pos_weight": scale_pos_weight
}

scores = cross_val_score(model_lgb, X, y, cv=5, scoring='roc_auc')
print(f'Cross Validation AUC: {np.mean(scores):.5f}')

lgb_proba = model_lgb.predict_proba(X_val)[:, 1]
lgb_auc_score = roc_auc_score(y_val, lgb_proba)
print(lgb_auc_score) # 0.7746 (sys) / 0.7804 (full)
end_time = time.time()
print(f"Time taken to read : {end_time - start_time} seconds") # 45.13 sec / 4155 sec

# cv = StratifiedKFold(n_splits=7, shuffle=False)
cv = StratifiedKFold(n_splits=3, shuffle=True)
# cv = KFold(n_splits=7, shuffle=False)
# cv = TimeSeriesSplit(n_splits=7)

fold = 1
fitted_models = []

oof_pred = np.zeros(X.shape[0])

for idx_train, idx_valid in cv.split(X, y):
    X_train, X_val = X.iloc[idx_train], X.iloc[idx_valid]
    y_train, y_val = y.iloc[idx_train], y.iloc[idx_valid]

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100)])
    fitted_models.append(model)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    oof_pred[idx_valid] = y_pred_proba
    fold_score = roc_auc_score(y_val, y_pred_proba)
    gc.collect()

    print(f"Fold {fold} AUC: {fold_score:.4f}")
    fold += 1

oof_pred_score = roc_auc_score(y, oof_pred)
print(f"CV roc_auc_oof: {oof_pred_score:.4f}")

oof_models_dict = [(str(i), model) for i, model in enumerate(fitted_models)]
voting_clf = VotingClassifier(estimators=oof_models_dict, voting='soft')
voting_clf.estimators_ = fitted_models
voting_clf.le_ = LabelEncoder().fit(y)
voting_clf.classes_ = voting_clf.le_.classes_

y_pred_voting = voting_clf.predict_proba(X)[:, 1]
voting_auc_score = roc_auc_score(y, y_pred_voting)
print(f"VotingClassifier AUC: {voting_auc_score:.4f}") # 0.8311 (sys+7fold) / 0.8267 (sys+st3 with shuffle) / 0.8130 (sys+ts)



test_pred = voting_clf.predict_proba(X_test)
test_pred = model_lgb.predict_proba(X_test)

sample_submission = pd.read_csv('./dacon_weblog/sample_submission.csv')
sample_submission['Click'] = preds
sample_submission.to_csv('submission_lgb_ts3vot.csv', index=False)

joblib.dump(voting_clf, "model_lgb_7fold_voting.pkl")
joblib.dump(y_pred_voting, "pred_lgb_7fold_voting.pkl")