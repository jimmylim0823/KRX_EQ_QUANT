import os
import pandas as pd
import numpy as np
import joblib

data_path_ = "data"
file_name_ = "data.sav"
file_path_ = os.path.join(data_path_, file_name_)
data_ = joblib.load(file_path_)
data_monthly_mkt_ = data_[2]
df_test_ = data_monthly_mkt_['CAP']

#%%
# operations for handling data
def to_monthly(x):
    x_m = x.resample('M').last()
    x_m = x_m[x_m.index>=data_monthly_mkt_['P'].index[0]]
    x_m.index = data_monthly_mkt_['P'].index
    return x_m

#%%
# time series operation
def ts_zscore(x, window):
    avg = x.rolling(window).mean()
    std = x.rolling(window).std()
    return (x - avg)/std

def ts_rank(x, window):
    return x.rolling(window).rank(pct=True)

def ts_mean(x, window):
    return x.rolling(window).mean()

def ts_std(x, window):
    return x.rolling(window).std()

def ts_delay(x, window):
    return x.shift(window)

def ts_change(x, window):
    return x.diff(window)

def ts_pct(x, window):
    return x.pct_change(window)

#%%
# cross section operation
def zscore(x):
    avg = x.mean()
    std = x.std()
    return (x.sub(avg)).div(std)

def rank(x):
    return x.rank(pct=True)

#%%
# basic operation
def signed_power(x, power):
    return np.sign(x) * x ** power