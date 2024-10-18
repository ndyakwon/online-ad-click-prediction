import pandas as pd
import numpy as np
import polars as pl
import os, gc, time, math, joblib
import optuna

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import category_encoders as ce


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

start_time = time.time()
# train = pl.read_parquet("./dacon_weblog/train_sample.parquet")
train = pl.read_parquet("./dacon_weblog/train_sample_sys.parquet")
# train = pl.read_parquet("./dacon_weblog/train_sample_bal.parquet")
train = train.to_pandas()
train = reduce_mem_usage(train)
end_time = time.time()
print(f"Time taken to read : {end_time - start_time} seconds") # 202 sec / 4603 mb

start_time = time.time()
test = pl.read_csv("./dacon_weblog/test.csv")
test = test.to_pandas()
test = reduce_mem_usage(test)
end_time = time.time()
print(f"Time taken to read : {end_time - start_time} seconds") # 18 sec / 743 mb

X = train.drop(columns=['ID', 'Click'])
y = train['Click']

### 결측치 처리
X.dtypes

for col in X.columns:
    if X[col].dtype == 'category':
       X[col] = X[col].astype('object')

for col in X.columns:
    if X[col].isnull().sum() != 0:
       X[col].fillna(0, inplace=True)

for col in X.columns:
    print(f"{col} ({X[col].dtype}) : {X[col].isnull().sum()}")

X_test = test.drop(columns=['ID'])

### 범주형 변수 처리 : count encoding
encoding_target = list(X.dtypes[X.dtypes == "object"].index)
encoding_target

enc = ce.CountEncoder(cols=encoding_target).fit(X, y)
X_enc = enc.transform(X)
X_test_enc = enc.transform(X_test)

### Training
X_train, X_val, y_train, y_val = train_test_split(X_enc, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 30, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'seed': 42
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50)])

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred_proba)

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

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
    callbacks=[lgb.early_stopping(50)],
)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)

print("Best hyperparameters: ", best_params)
# (srs) Best hyperparameters :  {'num_leaves': 248, 'learning_rate': 0.06750609707433469, 'n_estimators': 999, 'min_child_samples': 77}
# (sys) Best hyperparameters:  {'num_leaves': 249, 'learning_rate': 0.06745513122938426, 'n_estimators': 958, 'min_child_samples': 85}
# (sys+bal) Best hyperparameters:  {'num_leaves': 255, 'learning_rate': 0.09641453739596516, 'n_estimators': 975, 'min_child_samples': 55
# (sys3) Best hyperparameters:  {'num_leaves': 238, 'max_depth': 18, 'learning_rate': 0.048039377523140483, 'n_estimators': 595, 'min_child_samples': 37}
# (all) Best hyperparameters:  {'num_leaves': 300, 'max_depth': 16, 'learning_rate': 0.06050200574068186, 'n_estimators': 866, 'min_child_samples': 16}

# grid best parameters: {'boosting_type': 'gbdt', 'learning_rate': 0.05, 'max_depth': 15, 'metric': 'auc', 'n_estimators': 1000, 'num_leaves': 100, 'objective': 'binary', 'random_state': 42, 'verbose': -1}

print("Best AUC: ", auc) # 0.7694 (srs) / 0.7703 (sys) / 0.7740 (sys+bal) / 0.7650 (sys3) / / 0.7765 (all)

### Prediction
test_pred = best_model.predict_proba(X_test_enc)

sample_submission = pd.read_csv('./dacon_weblog/sample_submission.csv')
sample_submission['Click'] = test_pred[:,1]
sample_submission.to_csv('submission_lgb_opt_3.csv', index=False)

joblib.dump(best_model, "model_lgb_opt_3.pkl")
joblib.dump(y_pred_proba, "pred_lgb_opt_3.pkl")