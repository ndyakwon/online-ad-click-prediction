import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import os, gc, time, math
from sklearn.utils import resample

pd.set_option('display.max_columns', 10)
pl.Config.set_tbl_cols(10)

### original data
train = pl.read_csv("./dacon_weblog/train.csv")
train.shape # (28605391, 41)

mean = []
window_size = 100000
for i in range(0, len(train)//window_size):
    mean.append(train[i*window_size:(i+1)*window_size, 'Click'].mean())
mean = np.array(mean)
plt.figure(figsize=(10, 3))
plt.plot(mean)
plt.show()

### random sampling
train = pl.read_csv("./dacon_weblog/train.csv")
train_sample = train.sample(math.ceil(len(train)*0.2), seed=42)
train_sample.shape # (5721079, 41)
train_sample.estimated_size('mb') # 1698 mb
train_sample.write_parquet("./dacon_weblog/train_sample.parquet")

train_sample = pl.read_parquet("./dacon_weblog/train_sample_srs.parquet")
train_sample.to_pandas()

### systematic sampling
def systematic_sampling(df, k):
    start = np.random.randint(0, k)
    indices = np.arange(start, len(df), step=k)
    sample = df[indices]
    return sample

train_sample_sys = systematic_sampling(train, 5)
train_sample_sys.shape # (5721078, 41)
train_sample_sys.write_parquet("./dacon_weblog/train_sample_sys.parquet")

train_sample_sys = systematic_sampling(train, 3)
train_sample_sys.shape # (9535131, 41)
train_sample_sys.write_parquet("./dacon_weblog/train_sample_sys.parquet")

train_sample_sys = pl.read_parquet("./dacon_weblog/train_sample_sys.parquet")
train_sample_sys.to_pandas()
train_sample_sys.write_csv("./dacon_weblog/train_sample_sys.csv")

mean_sys = []
window_size = 40000
for i in range(0, len(train_sample_sys)//window_size):
    mean_sys.append(train_sample_sys[i*window_size:(i+1)*window_size, 'Click'].mean())
mean = np.array(mean_sys)
plt.figure(figsize=(10, 3))
plt.plot(mean_sys)
plt.show()
train_index = train.with_row_count(name='original_index')

### sys + downsampling
minority_class = train_index.filter(pl.col('Click') == 1)
majority_class = train_index.filter(pl.col('Click') == 0)
len(minority_class) # 5569860
len(majority_class) # 23035531

sampled_majority_class = majority_class[::5, :]
train_sample_down = pl.concat([minority_class, sampled_majority_class], how='vertical').sort('original_index')
train_sample_down = train_sample_down.drop('original_index')
train_sample_down.write_parquet("./dacon_weblog/train_sample_down.parquet")

train_sample_down = pl.read_parquet("./dacon_weblog/train_sample_down.parquet")
train_sample_down.to_pandas()

# majority_class_down = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)
# train_sample_bal = pl.concat([majority_class_down, minority_class], how='vertical')
# train_sample_bal.shape # (11139720, 41)
# train_sample_bal.write_parquet("./dacon_weblog/train_sample_bal.parquet")




mean = []
window_size = 100000
for i in range(0, len(train)//window_size):
    mean.append(train[i*window_size:(i+1)*window_size, 'Click'].mean())
mean = np.array(mean)
plt.figure(figsize=(10, 3))
plt.plot(mean)
plt.show()

mean_srs = []
window_size = 20000
for i in range(0, len(train_sample)//window_size):
    mean_srs.append(train_sample[i*window_size:(i+1)*window_size, 'Click'].mean())
mean_srs = np.array(mean_srs)
plt.figure(figsize=(10, 3))
plt.plot(mean_srs)
plt.show()

train_sample_sys.shape
mean_sys = []
window_size = 20000
for i in range(0, len(train_sample_sys)//window_size):
    mean_sys.append(train_sample_sys[i*window_size:(i+1)*window_size, 'Click'].mean())
mean_sys = np.array(mean_sys)
plt.figure(figsize=(10, 3))
plt.plot(mean_sys)
plt.show()

mean_down = []
window_size = 35500
for i in range(0, len(train_sample_down)//window_size):
    mean_down.append(train_sample_down[i*window_size:(i+1)*window_size, 'Click'].mean())
mean_down = np.array(mean_down)
plt.figure(figsize=(10, 3))
plt.plot(mean_down)
plt.show()

train_index = train.with_row_count(name='original_index')



### Distribution check

train.describe()
train.filter(pl.col('Click')==0).describe()
train.filter(pl.col('Click')==1).describe()

train_sample.describe()
train_sample_sys.describe()
train_sample_bal.describe()
train_sample_sys_bal.describe()

train_sample.filter(pl.col('Click')==0).describe()
train_sample.filter(pl.col('Click')==1).describe()


### figure
fig, axes = plt.subplots(3, figsize=(12,5))
axes[0].plot(mean)
axes[1].plot(mean_srs)

axes[2].plot(mean_down)

for ax in axes:
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('mean')  # Replace with actual label
    ax.set_xticklabels([])  # Remove x-axis tick labels

plt.tight_layout()
plt.show()