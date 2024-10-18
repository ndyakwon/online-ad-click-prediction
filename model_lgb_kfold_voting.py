import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import os, gc, time, math, joblib
import optuna

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
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
train = pl.read_csv("./dacon_weblog/train.csv")
# train = pl.read_parquet("./dacon_weblog/train_sample_sys.parquet")
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
# X_train, X_val, y_train, y_val = train_test_split(X_enc, y, test_size=0.2, random_state=42)

# (sys) Best hyperparameters:  {'num_leaves': 249, 'learning_rate': 0.06745513122938426, 'n_estimators': 958, 'min_child_samples': 85}

# model_lgb = LGBMClassifier()
# model_lgb.fit(X_train, y_train,
#               eval_set=[(X_val, y_val)],
#               callbacks=[lgb.early_stopping(50)])
#
# y_pred_proba = model_lgb.predict_proba(X_val)[:, 1]
# score = roc_auc_score(y_val, y_pred_proba)

# lgb.plot_importance(model_lgb, max_num_features=20, importance_type='split')
# plt.show()

# params = {
#     "boosting_type": "gbdt",
#     "objective": "binary",
#     "metric": "auc",
#     "num_leaves": 249,
#     "learning_rate": 0.06745513122938426,
#     "n_estimators": 958,
#     "min_child_samples": 85,
#     "verbose": -1,
#     "random_state": 42,
# }

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 300,
    "max_depth": 16,
    "learning_rate": 0.06050200574068186,
    "n_estimators": 866,
    "min_child_samples": 16,
    "verbose": -1,
    "random_state": 42,
}

# cv = StratifiedKFold(n_splits=7, shuffle=False)
cv = KFold(n_splits=3, shuffle=False)

fold = 1
fitted_models = []

oof_pred = np.zeros(X_enc.shape[0])

for idx_train, idx_valid in cv.split(X_enc):
    X_train, X_val = X_enc.iloc[idx_train], X_enc.iloc[idx_valid]
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

y_pred_voting = voting_clf.predict_proba(X_enc)[:, 1]
voting_auc_score = roc_auc_score(y, y_pred_voting)
print(f"VotingClassifier AUC: {voting_auc_score:.4f}") # 0.8056 (sys5+st5) / 0.8060 (sys5+st7) / 0.7899 (sys3+st7) / 0.7819 (all+st7) / 0.7819 (all+k7)

joblib.dump(voting_clf, "model_lgb_kfold_voting_5.pkl")
joblib.dump(y_pred_voting, "pred_lgb_kfold_voting_5.pkl")