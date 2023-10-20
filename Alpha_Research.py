import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


class Alpha_Research():
    def __init__(self, data_path="data", file_name="data.sav"):
        file_path = os.path.join(data_path, file_name)
        datas = joblib.load(file_path)
        self.data_daily = datas[0]
        self.data_monthly_acc = datas[1]
        self.data_monthly_mkt = datas[2]
        self.data_index = datas[3]
        self.data_classification = datas[4]
        self.data_weight = datas[5]

        # delay-1 alpha
        self.df_target = (self.data_daily['ADJO'] + self.data_daily['ADJH']).copy()
        self.df_target += (self.data_daily['ADJL'] + self.data_daily['ADJC']).copy()
        self.df_target = self.df_target / 4
        self.df_target = self.df_target.resample('M').first().shift(-1).iloc[:-1,:]
        self.df_target = self.df_target[self.df_target.index>list(self.data_weight.keys())[0]]
        self.df_target.index = list(self.data_weight.keys())[:-1]
        self.df_target = (1 + self.df_target.pct_change()).apply(np.log)
        self.df_target = self.df_target.shift(-1).iloc[:-1,:]

        self.fgsc = self.data_classification['FGSC'].fillna(method='bfill').iloc[0]
        self.sector_dict = dict()
        for dt in self.df_target.index.to_list():
            temp_dict = dict()
            for sc in self.fgsc.unique():
                sc_temp = self.fgsc[self.fgsc==sc].index.to_list()
                temp_dict[sc] = [i for i in self.data_weight[dt].keys() if i in sc_temp]
            self.sector_dict[dt] = temp_dict
    
    
    def neutralize(self, x):
        demean = x - x.mean()
        return demean / demean.abs().sum()
    
    
    def sharpe_ratio(self, x):
        return x.mean() / x.std() * np.sqrt(12)


    def max_drawdown(self, x):
        cum_ret = (1 + x).cumprod()
        mdd = cum_ret / cum_ret.cummax() - 1
        return mdd.min()


    def ls_count(self, x):
        stats = {}
        stats['Min Long Count'] = x[x>0.000001].count(axis=1).resample('Y').min()
        stats['Min Short Count'] = x[x<-0.000001].count(axis=1).resample('Y').min()
        df = pd.DataFrame(stats)
        years = list(range(df.index[0].year, df.index[-1].year+1))
        df.index = years
        return df.T
    
    
    def monthly_resampling(self, x):
        x_m = x.resample('M').last()
        x_m = x_m[x_m.index>=self.data_monthly_mkt['P'].index[0]]
        x_m.index = self.data_monthly_mkt['P'].index
        return x_m

    
    def previous_diff(self, x, recent, previous):
        if x.iloc[-1] != x.shift(recent).iloc[-1]:
            p = x - x.shift(recent)
        else:
            p = x - x.shift(previous)
        return p
    
    def ts_zscore(self, x, window):
        avg = x.rolling(window).mean()
        std = x.rolling(window).std()
        return (x - avg)/std

        
    def backtest(self, features, operation=None, chart=False, trunc=0.05):
        if type(features) == pd.core.frame.DataFrame:
            features = [features]
        else:
            pass
        
        backtest_dict = dict()
        weight_dict = dict()
        for dt in list(self.sector_dict.keys()):
            temp_cross = [i.loc[dt] for i in features]
            backtest_dict[dt] = 0.0
            temp = []
            sectors = self.sector_dict[dt]
            for sc in list(sectors.keys()):
                temp_sector = [i[sectors[sc]] for i in temp_cross]
                if operation == None:
                    temp_scores = temp_sector[0]
                else:
                    temp_scores = operation(temp_sector)
                temp_target = self.df_target[sectors[sc]].loc[dt]
                temp_weight = self.neutralize(temp_scores) / len(sectors.keys())
                temp_weight = temp_weight.clip(lower=-trunc, upper=trunc)
                temp_weight = temp_weight.apply(lambda x: 0.0 if -1e-6 < x < 1e-6 else x)
                temp_result = temp_weight * temp_target
                backtest_dict[dt] += temp_result.sum()
                temp.append(temp_weight)
            weight_dict[dt] = temp

        bt_series = pd.Series(backtest_dict)
        weights = [pd.concat(weight_dict[i]) for i in list(weight_dict.keys())]
        df_weights = pd.concat(weights, axis=1).T
        
        if chart == True:
            fig, ax = plt.subplots()
            (1 + bt_series).cumprod().plot(ax=ax)
            sr = self.sharpe_ratio(bt_series)
            mdd = self.max_drawdown(bt_series)
            ax.set_title(f"Sharpe Ratio: {sr:.2f} Max-Drawdown: {mdd:.2f}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value")
            
            print(self.ls_count(df_weights))
            
        else:
            pass
        return bt_series, df_weights


    def alpha_pool(self, alphas, topn=5):
        rets = [alphas[i][0] for i in alphas.keys()]
        ws = [alphas[i][1] for i in alphas.keys()]

        # find top correlation pairs
        df_corr = pd.DataFrame(rets, index=alphas.keys()).T.corr()
        correlation_list = []
        for i in range(len(df_corr.columns)):
            for j in range(i+1, len(df_corr.columns)):
                pair = (df_corr.columns[i], df_corr.columns[j], df_corr.iloc[i, j])
                correlation_list.append(pair)
        correlation_list.sort(key=lambda x: abs(x[2]), reverse=True)
        i = 0
        for pair in correlation_list:
            if i >= topn:
                break
            print(f"{pair[0]} and {pair[1]} have correlation: {pair[2]:.2f}")
            i += 1

        # backtest equal weighted portfolio
        df_ew = pd.concat(ws, axis=1).groupby(level=0, axis=1).mean()
        _ = self.backtest(df_ew, chart=True)
    
    
if __name__ == "__main__":
    data_path = "data"
    file_name = "data.sav"
    file_path = os.path.join(data_path, file_name)
    datas = joblib.load(file_path)
    data_monthly_mkt = datas[2]

    ar = Alpha_Research()

    features = [data_monthly_mkt['RET1Y'], data_monthly_mkt['RET20']]

    def operation(features):
        return (-(features[0] - features[1])).rank(pct=True)

    ar.backtest(features, operation).plot()
