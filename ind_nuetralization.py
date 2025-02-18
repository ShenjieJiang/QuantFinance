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

ind = pd.read_pickle('D:/redata/risk_factors/ind_label.pkl')

#多天打标
starttime = '2024-12-26'
endtime = '2024-12-31'

ind_label_proc = pd.DataFrame(columns= ['date','ticker','first_industry_code'])

# 定义一个函数来生成工作日日期范围
def date_range(start, end):
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # 0-4代表周一到周五
            yield current_date.strftime('%Y-%m-%d')
        current_date += timedelta(days=1)

for date in date_range(starttime, endtime):
    ind_label_raw = oss_client.read_oss_pickle_file(f'ad_hoc_prod/sector/{date}.pkl')
    temp_df = pd.DataFrame(columns=['date', 'ticker', 'first_industry_code'])
    temp_df['ticker'] = ind_label_raw.index
    temp_df['ticker'] = temp_df['ticker'].str.extract(r'(\d+)')
    temp_df['date'] = ind_label_raw['date'].values
    temp_df['first_industry_code'] = ind_label_raw['first_industry_code'].values
    
    # 将临时 DataFrame 追加到最终的 DataFrame
    # ind_label_proc = ind_label_proc.append(temp_df, ignore_index=True)
    ind_label_proc = pd.concat([ind_label_proc, temp_df], ignore_index=True)

temp_df = ind_label_proc

df1 = pd.DataFrame(columns= ['date','ticker','10','11','12','20','21','22','23','24','25','26','27','28','30','31','32','33','34','35','36','37','40','41','42','43','50','60','61','62','63','70','999'])
df1['date'] = temp_df['date']
df1['ticker'] = temp_df['ticker']

for i, code in enumerate(temp_df['first_industry_code']):
    if code in df1.columns:
        df1.at[i, code] = 1
    else:
        df1.at[i, code] = 0
        
df1 = df1.fillna(0)

ind_label = pd.concat([ind, df1], ignore_index=True)
ind_label['date'] = pd.to_datetime(ind_label['date'])
ind_label = ind_label.reset_index(drop=True)
ind_label = pd.DataFrame(ind_label)
ind_label.to_pickle('D:/redata/risk_factors/ind_label.pkl')
ind_label

#单天打标
ind = pd.read_pickle('D:/redata/risk_factors/ind_label.pkl')
ind.tail(1)

date_data = '2024-12-27'

ind_label_raw = oss_client.read_oss_pickle_file(f'ad_hoc_prod/sector/{date_data}.pkl')

ind_label_proc = pd.DataFrame(columns= ['date','ticker','first_industry_code'])

temp_df = pd.DataFrame(columns=['date', 'ticker', 'first_industry_code'])
temp_df['ticker'] = ind_label_raw.index
temp_df['ticker'] = temp_df['ticker'].str.extract(r'(\d+)')
temp_df['date'] = ind_label_raw['date'].values
temp_df['first_industry_code'] = ind_label_raw['first_industry_code'].values
ind_label_proc = pd.concat([ind_label_proc, temp_df], ignore_index=True)

temp_df = ind_label_proc

df1 = pd.DataFrame(columns= ['date','ticker','10','11','12','20','21','22','23','24','25','26','27','28','30','31','32','33','34','35','36','37','40','41','42','43','50','60','61','62','63','70','999'])
df1['date'] = temp_df['date']
df1['ticker'] = temp_df['ticker']

for i, code in enumerate(temp_df['first_industry_code']):
    if code in df1.columns:
        df1.at[i, code] = 1
    else:
        df1.at[i, code] = 0
        
df1 = df1.fillna(0)

ind_label = pd.concat([ind, df1], ignore_index=True)
ind_label['date'] = pd.to_datetime(ind_label['date'])
ind_label = ind_label.reset_index(drop=True)
ind_label = pd.DataFrame(ind_label)
ind_label.to_pickle('D:/redata/risk_factors/ind_label.pkl')
ind_label