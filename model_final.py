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

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    Numeric_Int_types = ['int8', 'int16', 'int32', 'int64']
    Numeric_Float_types = ['float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) in ('category', 'datetime64[ms]', 'datetime64[ns]'):
            continue

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if col_type in Numeric_Int_types:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            elif col_type in Numeric_Float_types:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

train_sample_sys = pl.read_parquet("./dacon_weblog/train_sample_sys.parquet")
train = train_sample_sys.to_pandas()
train = reduce_mem_usage(train)
train.shape

train_sample_bal = pl.read_parquet("./dacon_weblog/train_sample_down.parquet")
train = train_sample_bal.to_pandas()
train = reduce_mem_usage(train)
train.shape

test = pl.read_csv("./dacon_weblog/test.csv")
test = test.to_pandas()
test = reduce_mem_usage(test)
test.shape

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



cv = TimeSeriesSplit(n_splits=3)

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
study.optimize(objective, n_trials=30)

trial = study.best_trial
print(f'Best trial: {trial}')
print('  Value: {:.4f}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

# best_params = study.best_params
# print("Best hyperparameters: ", best_params)

# Best parameters for sys sample : {'num_leaves': 110, 'max_depth': 20, 'learning_rate': 0.014557869682012175, 'n_estimators': 1457, 'min_child_samples': 85}
# value: 0.766812818113539

# value: 0.7747709072930622 and parameters: {'num_leaves': 286, 'learning_rate': 0.014911948416223737, 'n_estimators': 1101, 'min_child_samples': 54} best without depth
# value: 0.7743404019906425 and parameters: {'num_leaves': 112, 'max_depth': 17, 'learning_rate': 0.026886686934795918, 'n_estimators': 1729, 'min_child_samples': 39}
# value: 0.7744731869689004 and parameters: {'num_leaves': 141, 'max_depth': 19, 'learning_rate': 0.028095120127038273, 'n_estimators': 1677, 'min_child_samples': 40} best

best_params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 141,
    "max_depth": 19,
    "learning_rate": 0.028095120127038273,
    "n_estimators": 1677,
    "min_child_samples": 40,
    "verbose": -1,
    "random_state": 42,
}

best_model = LGBMClassifier(**best_params)
best_model.fit(X, y)

best_model = LGBMClassifier(random_state=42)
best_model.fit(X, y)

test_pred = best_model.predict_proba(X_test)

sample_submission = pd.read_csv('./dacon_weblog/sample_submission.csv')
sample_submission['Click'] = test_pred[:, 1]
sample_submission.to_csv('submission_lgb.csv', index=False)

