import pandas as pd
import numpy as np
import polars as pl
import os, gc, time, math

from sklearn.model_selection import train_test_split
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

def reduce_memory_usage(df):
    print(f"Memory usage of dataframe is {round(df.estimated_size('mb'), 2)} MB")
    Numeric_Int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    Numeric_Float_types = [pl.Float32, pl.Float64]

    for col in df.columns:
        try:
            col_type = df[col].dtype
            if col_type == pl.String:
                continue

            c_min = df[col].min()
            c_max = df[col].max()

            if col_type in Numeric_Int_types:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df = df.with_columns(df[col].cast(pl.Int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df = df.with_columns(df[col].cast(pl.Int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df = df.with_columns(df[col].cast(pl.Int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df = df.with_columns(df[col].cast(pl.Int64))

            elif col_type in Numeric_Float_types:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_columns(df[col].cast(pl.Float32))
                else:
                    pass
                pass
        except:
            pass
    print(f"Memory usage of dataframe became {round(df.estimated_size('mb'), 2)} MB")
    return df

start_time = time.time()
train = pl.read_parquet("./dacon_weblog/train_sample.parquet")
# train = pl.read_csv("./dacon_weblog/train.csv")
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
X_test = test.drop(columns=['ID'])

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

### 범주형 변수 처리 : count encoding
encoding_target = list(X.dtypes[X.dtypes == "object"].index)
encoding_target

enc = ce.CountEncoder(cols=encoding_target).fit(X, y)
X_enc = enc.transform(X)
X_test_enc = enc.transform(X_test)

### Training
X_train, X_val, y_train, y_val = train_test_split(X_enc, y, test_size=0.2, random_state=42)

# class_weights = {0: 1, 1: 4}
model_lgb = LGBMClassifier(random_state=42)
model_lgb.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50)])

model_cat = CatBoostClassifier(random_state=42)
model_cat.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,
              verbose=True)

### Evaluation
lgb_preds = model_lgb.predict(X_val)
lgb_acc_score = accuracy_score(y_val, lgb_preds)
print(lgb_acc_score) # 0.8143 (srs) / # 0.6843 (srs) + wt / 0.8145 (sys) / 0.6752 (bal) / 0.6753 (sys+bal)

score = model_lgb.predict_proba(X_val)[:, 1]
lgb_auc_score = roc_auc_score(y_val, score)
print(lgb_auc_score) # 0.7400 (srs) / # 0.7398 (srs) + wt / 0.7403 (sys) / 0.7402 (bal) / 0.7402 (sys+bal)

fpr, tpr, thresholds = roc_curve(y_val, score, pos_label=1)
lgb_auc_score = auc(fpr, tpr)
print(lgb_auc_score) # 0.7400 (srs) / # 0.7398 (srs) + wt / 0.7403 (sys) / 0.7402 (bal) / 0.7402 (sys+bal)

cat_preds = model_cat.predict(X_val)
cat_acc_score = accuracy_score(y_val, cat_preds)
print(cat_acc_score) # 0.8200 (srs) / 0.6950 (sys+bal)

# score = model_cat.predict_proba(X_test)[:, 1]
# fpr, tpr, thresholds = metrics.roc_curve(_val, score, pos_label=1)
# auc_score = metrics.auc(fpr, tpr)
# print(auc_score)

### Prediction
test_pred = model_lgb.predict_proba(X_test_enc)

sample_submission = pd.read_csv('./dacon_weblog/sample_submission.csv')
sample_submission['Click'] = test_pred[:,1]
sample_submission.to_csv('submission_baseline.csv', index=False)