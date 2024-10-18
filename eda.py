import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import os, gc, time, math

# import plotly.express as px
# import plotly.io as pio
# pio.renderers.default = 'browser'

pd.set_option('display.max_columns', 41)
pl.Config.set_tbl_cols(41)

# train = pl.read_csv("./dacon_weblog/train.csv")
# train.shape # (28605391, 41) 2800만
# train.columns # ID, Click, F01~F39
# train.drop('ID', 'Click').head
# train.estimated_size('mb') # 8108 mb

train = pl.read_parquet("./dacon_weblog/train_sample.parquet")

null_ratio = train.null_count()/len(train)
null_ratio.to_pandas().transpose()

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

train.shape # (28605391, 41) -> (5721079, 41)
test.shape # (4538541, 40)

# start_time = time.time()
# train = pl.read_csv("./dacon_weblog/train.csv")
# train = reduce_memory_usage(train)
# train = train.to_pandas()
# train = reduce_mem_usage(train)
# end_time = time.time()
# print(f"Time taken to read : {end_time - start_time} seconds") # 207 sec / 4603 mb
#
# start_time = time.time()
# train = pd.read_csv("./dacon_weblog/train.csv")
# train = reduce_mem_usage(train)
# end_time = time.time()
# print(f"Time taken to read : {end_time - start_time} seconds") # 352 sec / 4603 mb

gc.collect()

train.head()

click = train['Click'].value_counts(normalize=True).reset_index()
click.Click = ['Not Clicked : 0', 'Clicked : 1']

# click_figure = px.bar(click,
#        x=['Not Clicked : 0', 'Clicked : 1'],
#        y=click.values.tolist(),
#        labels={'x': 'Value', 'y': 'Percentage'},
#        width = 450, height = 500)
#
# click_figure.show()

plt.figure(figsize=(6, 6))
sns.barplot(x='Click', y='proportion', data=click, hue='Click', palette='Paired')
plt.show()

numeric_cols = train.select_dtypes(include='number').columns

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col], kde=True, bins=10)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(12, 18))
for i, col in enumerate(numeric_cols):
    r = i // 2
    c = i % 2

    sns.histplot(train[col], kde=True, bins=10, ax=axes[r, c])
    axes[r, c].set_title(f'Distribution of {col}')
    axes[r, c].set_xlabel(col)
    axes[r, c].set_ylabel('Frequency')

plt.show()