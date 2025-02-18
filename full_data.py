import import_ipynb
from oss_handler import OssClient
import sys, os
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from pylab import mpl
import seaborn as sns
from sklearn.linear_model import LinearRegression
import datetime
from datetime import datetime, timedelta
from oss2.exceptions import NoSuchKey
import warnings
warnings.filterwarnings('ignore')
oss_client = OssClient()

# idx000001 = oss_client.read_oss_pickle_file('ad_hoc_prod/index_hist/000001.XSHG_hist_index.pkl') #上证指数
# idx000016 = oss_client.read_oss_pickle_file('ad_hoc_prod/index_hist/000016.XSHG_hist_index.pkl') #上证 50
# idx000300 = oss_client.read_oss_pickle_file('ad_hoc_prod/index_hist/000300.XSHG_hist_index.pkl') #沪深 300
idx000852 = oss_client.read_oss_pickle_file('ad_hoc_prod/index_hist/000852.XSHG_hist_index.pkl') #中证 1000
idx000905 = oss_client.read_oss_pickle_file('ad_hoc_prod/index_hist/000905.XSHG_hist_index.pkl') #中证 500
idx000985 = oss_client.read_oss_pickle_file('ad_hoc_prod/index_hist/000985.XSHG_hist_index.pkl') #中证全指
# idx399001 = oss_client.read_oss_pickle_file('ad_hoc_prod/index_hist/399001.XSHG_hist_index.pkl') #深证成指
# idx399006 = oss_client.read_oss_pickle_file('ad_hoc_prod/index_hist/399006.XSHG_hist_index.pkl') #创业板指
# idx399303 = oss_client.read_oss_pickle_file('ad_hoc_prod/index_hist/399303.XSHG_hist_index.pkl') #国证 2000

adj_close = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/adj_close.pk')
adj_factor = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/adj_factor.pk')
adj_high = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/adj_high.pk')
adj_low = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/adj_low.pk')
adj_open = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/adj_open.pk')

circulation_a = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/circulation_a.pk') #流通股本
total_a = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/total_a.pk') #总股本
circulation_market_value = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/circulation_market_value.pk') #流通市值

volume = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/volume.pk') #成交量
num_trades = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/num_trades.pk') #交易次数
total_turnover = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/total_turnover.pk') #成交额
turnover_rate = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/turnover_rate.pk') #换手率

close = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/close.pk') 
open = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/open.pk') 
high = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/high.pk') 
low = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/low.pk') 

limit_down = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/limit_down.pk') #跌停
limit_up = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/limit_up.pk') #涨停

halt_status = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/halt_status.pk') #停牌 
st_status = oss_client.read_oss_pickle_file('ad_hoc_prod/fields_full/st_status.pk') #st

BS_pit = oss_client.read_oss_pickle_file('ad_hoc_prod/pit_fundemental/BS_pit.pkl')
CF_pit = oss_client.read_oss_pickle_file('ad_hoc_prod/pit_fundemental/CF_pit.pkl')
IS_pit = oss_client.read_oss_pickle_file('ad_hoc_prod/pit_fundemental/IS_pit.pkl')
all_pit = oss_client.read_oss_pickle_file('ad_hoc_prod/pit_fundemental/all_pit_fund_new_api.pkl')

##正常交易非st，halt的票打1.其他为np.nan
st_status = st_status.replace(True,np.nan).replace(False,True)
halt_status = halt_status.replace(True,np.nan).replace(False,True)

adj_return =  adj_close/adj_close.shift(1) - 1
adj_return = adj_return.replace(np.inf,np.nan).replace(-np.inf,np.nan)

##调仓周期
o1o2 = adj_open.shift(-2)/adj_open.shift(-1) - 1 #1日调仓
# o1o2 = adj_open.shift(-5) / adj_open.shift(-1) - 1 #5日调仓
# o1o2 = adj_open.shift(-10) / adj_open.shift(-1) - 1 #10日调仓
# o1o2 = adj_open.shift(-20) / adj_open.shift(-1) - 1 #20日调仓

mc = total_a * adj_close

csi500 = idx000905[['open','close','low']].set_index(idx000905['date'])
idx_500_ret = (csi500['close']/csi500['close'].shift(1) - 1)

mclose = csi500['close']
mopen = csi500['open']
mlow = csi500['low']

open['label'] = 1

# 中性化
logcmv = np.log(circulation_market_value)
logcmv = logcmv.loc['2017':]
logcmv = logcmv.stack().reset_index()
logcmv.columns = ['date','ticker','logcmv']
logcmv.to_pickle('D:/redata/risk_factors/logcmv.pkl')
logcmv.tail(1)

retsum120 = adj_return.rolling(window=120).sum()
retsum120 = retsum120.loc['2017':]
retsum120 = retsum120.stack().reset_index()
retsum120.columns = ['date','ticker','retsum120']
retsum120.to_pickle('D:/redata/risk_factors/retsum120.pkl')
retsum120.tail(1)

##行业市值中性化
##因子中性化 需要的数据
risk1 = pd.read_pickle('D:/redata/risk_factors/logcmv.pkl')
risk2 = pd.read_pickle('D:/redata/risk_factors/retsum120.pkl')
ind = pd.read_pickle('D:/redata/risk_factors/ind_label.pkl')
# ind = pd.read_pickle('D:/redata/risk_factor/ind_label.pk')
risk = risk1.merge(risk2,on=['date','ticker'],how='left')
risk = risk.merge(ind,on=['date','ticker'],how='left')
risk = risk[(risk['date']>'2017-01-01')]
risk = risk.drop_duplicates(keep='last')

##行业中性 2023.8.3 更新
def neturalize_ind(factor,risk=risk,risk_cols=['logcmv','retsum120']):
    XY = pd.merge(risk, factor, on=['date', 'ticker'], how='right')
    #print(XY.head(3))
    XY = XY.dropna()
    tradeday = XY.date.unique()
    daily_resid = []
    for date in tqdm(tradeday[:]):
        xy = XY[XY['date'] == date]
        #只保留有因子值的数据
        xy = xy.dropna()

        #数据标准化
        xy[risk_cols] = (xy[risk_cols] - xy[risk_cols].mean())/xy[risk_cols].std()                
        xy['factor'] = (xy['factor']-xy['factor'].mean())/xy['factor'].std() 

        #选出x y
        x = xy.iloc[:,2:-1]
        y = xy['factor']
        #回归
        multi_linear = LinearRegression(fit_intercept=False)  #无截距项
        multi_linear.fit(x, y)
        beta = multi_linear.coef_
        
        #取残差
        xy['beta'] = np.dot(x, beta)
        xy['resid'] = xy['factor'] - xy['beta']
        daily_resid.append(xy)
    data = pd.concat(daily_resid)
    factor = data[['date', 'ticker', 'factor', 'resid']]
    return factor

def factor_neutralize(df,factor_type,if_rank=True,risk=None,risk_cols=None): #factor_type=hist原值；ind行业及其他中性；单个风险因子文件名； if_rank：中性化之前时候对因子做rank处理
    #中性之前是否对因子做rank处理
    if if_rank == True:
        df = df.rank(axis=1, ascending=True)  # 原值越大rank值越大   
    #因子原值
    if factor_type == 'hist':
        factor_rank = df.copy()
    #行业市值中性
    elif factor_type == 'ind':  ###行业和风格中性
        def neu_process(df):
            df = df.stack().reset_index()
            df.columns = ['date','ticker','factor']
            df = neturalize_ind(df,risk,risk_cols)
            resid = df.set_index(['date','ticker'])['resid'].unstack()
            return resid        
        factor_rank = neu_process(df)
        factor_rank = factor_rank.rank(axis=1,ascending=True) 
        
    return factor_rank

def neu_process(df):
    df = df.stack().reset_index()
    df.columns = ['date','ticker','factor']
    df = neturalize_ind(df,risk=risk,risk_cols=['logcmv','retsum120'])
    resid = df.set_index(['date','ticker'])['resid'].unstack()
    return resid  

def select_num_stocks(n):
    select_stocks = (factor_rank <= n)
    select_stocks_last = select_stocks.iloc[-1:,]
    select_stocks_last = select_stocks_last.replace(True,1).replace(False,np.nan)
    select_stocks_last = select_stocks_last.dropna(how='all',axis=1)
    return select_stocks_last

##市值排uni
mv_rank = risk1.set_index(['date','ticker'])['logcmv'].unstack()
mv_rank = mv_rank.rank(axis=1, ascending=True, pct=True)

##小市值
small = (mv_rank <= 0.33)
small = small.replace(True,1).replace(False,np.nan)
small = small.loc['2017':]

##中市值
median = (mv_rank <= 0.66) & (mv_rank >= 0.33)
median = median.replace(True,1).replace(False,np.nan)
median = median.loc['2017':]

##小中市值
sm = (mv_rank <= 0.66)
sm = sm.replace(True,1).replace(False,np.nan)
sm = sm.loc['2017':]

##大市值
large = (mv_rank>=0.66)
large = large.replace(True,1).replace(False,np.nan)
large = large.loc['2017':]

def position(df, period):
    empty = df > 0
    empty = empty.replace(True,np.nan).replace(False,np.nan)
    
    position = df.iloc[::period]
    position = pd.concat([empty, position])
    position = position.sort_values(['date'])
    position = position.reset_index()
    position = position.drop_duplicates(subset=['date'], keep='last').ffill(limit = period)
    position = position.set_index(['date'])
    return position

##财务数据函数

def chg_format(df1):
    df1['ticker'] = df1['order_book_id'].str[:-5]
    df1['date'] = pd.to_datetime(df1['info_date'])
    df1['report_period'] = df1['quarter'].map(lambda x:x.replace('q1','0331').replace('q2','0630').replace('q3','0930').replace('q4','1231'))
    df1['report_period'] = df1['report_period'].astype('int')
    df1 = df1.sort_values(['date','ticker','report_period']).reset_index(drop=True)
    return df1

def func_save(factor,factor_name):
    ##只保留当前最新业绩的数据
    def func(x):
        x = x[x['report_period']>=x['report_period'].expanding(min_periods=1).max()]
        return x
    factor = factor.groupby('ticker').apply(func).reset_index(drop=True)
    #factor同一个date ticker 去重，保留当前最新report_period
    factor = factor.drop_duplicates(subset=['date','ticker'],keep='last')
    factor = factor.set_index(['date','ticker'])[factor_name].unstack()
    factor = pd.concat([factor,Open[['label']]],axis=1).drop(['label'],axis=1)
    factor = factor.fillna(method='ffill',limit=150).dropna(how='all',axis=0).dropna(how='all',axis=1)
    factor.to_pickle(os.path.join(path2,factor_name+'.pkl'))

    ##利润表流量单季度数据
dfaa = IS_pit
dfaa = dfaa.reset_index()
dfaa = chg_format(dfaa)
dfaa = dfaa.drop_duplicates(subset=['date','ticker','report_period'],keep='last')

#时间表
mls = pd.DataFrame(dfaa.report_period.unique())
mls.columns = ['report_period']
mls['pre_report_period'] = mls['report_period'].shift(1)
mls['pre2_report_period'] = mls['report_period'].shift(2)
mls['pre3_report_period'] = mls['report_period'].shift(3)
mls['next1_report_period'] = mls['report_period'].shift(-1)
mls['next2_report_period'] = mls['report_period'].shift(-2)
mls['next3_report_period'] = mls['report_period'].shift(-3)
mls = mls.replace(np.nan,0)
mls = mls.astype('int')