import numpy as np
import pandas as pd
import statsmodels.api as sm

def process_factors(df, factors, process_method='rank'):
    """
    使用示例：dfH = process_factors(dfH, factors=['HIGH250'], process_method='rank')
    
    """
    # 因子处理：排序或标准化
    for factor in factors:
        if process_method == 'rank':
            df[f'{factor}_process'] = df.groupby('date')[factor].rank(method='average', ascending=True)
            print('yes,it is rank')
        elif process_method == 'zscore':
            df[f'{factor}_process'] = df.groupby('date')[factor].transform(lambda x: (x - x.mean()) / x.std())
        else:
            df[f'{factor}_process'] = df[factor]
    return df


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def neutralize_by_date(group, factor, for_neutral):

    if not group.empty:
        X = group[for_neutral]
        y = group[factor]
        model = LinearRegression()
        model.fit(X,y)
        group[f'neu_{factor}'] = y-model.predict(X)

    else:
        group[f'neu_{factor}'] = group[factor]
    
    return group




def neutralize_factors(df, factors, process=True, market_factor=True, momentum_factor=True, industry_factor=True, df_industry = None,other=None):
    """
    使用示例：dfH = neutralize_factors(dfH, factors=['HIGH250'], market_factor=True, momentum_factor=True, industry_factor=False)

    """
    # 计算市值因子
    for_neutral = []
    if market_factor:
        df['market_factor'] = np.log(df['circulation_market_value'])
        for_neutral.append('market_factor')

    # 计算动量因子
    if momentum_factor:
        df['momentum_factor'] = df.groupby('order_book_id')['ret_daily'].transform(lambda x: x.rolling(window=120, min_periods=120).sum().shift(1))
        for_neutral.append('momentum_factor')

    if industry_factor:
        df = pd.merge(df, df_industry, on=['date', 'order_book_id'])
        dummy_columns = [col for col in df_industry.columns if col.startswith('industry_')]
        for_neutral = for_neutral + dummy_columns

    if other is not None:
        for_neutral.append(other)


    # 删除空值
    required_columns = factors + for_neutral
    df = df.dropna(subset=required_columns)


    for factor in factors:
        if process:
            df = df.groupby('date').apply(neutralize_by_date, factor=f'{factor}_process', for_neutral=for_neutral)
        else: 
            df = df.groupby('date').apply(neutralize_by_date, factor=f'{factor}', for_neutral=for_neutral)
        df = df.reset_index(drop=True)

    print('Finish neutralize')
    return df




import matplotlib.pyplot as plt

# 画图：累计rankIC或IC
def plot_IC(daily_ic, cumulative_ic, mean_ic, icir, ic_type='rankIC',log=False,name=None):
    """
    daily_ic 和 cumulative_ic 是包含日期索引的 Series
    ic_type - 'rankIC' 或 'IC'
    
    """
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # 左轴柱状图
    ax1.bar(daily_ic.index, daily_ic, color='b', width=0.8, label=f'Daily {ic_type}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'Daily {ic_type}', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title(f'{ic_type} Over Time (Mean: {mean_ic:.4f}, ICIR: {icir:.4f})')
   
    # mean_ic 的水平线
    ax1.axhline(y=mean_ic, color='g', linestyle='-', linewidth=2)
    
    # 右轴累计 IC
    ax2 = ax1.twinx()
    ax2.plot(cumulative_ic.index, cumulative_ic, color='r', linestyle='--', label=f'Cumulative {ic_type}')
    ax2.set_ylabel(f'Cumulative {ic_type}', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.grid(True)
    if log:
        plt.savefig(name)
    else:
        plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_and_plot_ic(df, factor, ret='ret_daily', ic_type='IC',log=False,namea=None,nameb=None):
    """
    计算并绘制IC或RankIC
    ic_type - 'rankIC' 或 'IC' 或 'both'
    """
   
    # results = {}
    
    if ic_type in ['rankIC', 'both']:
        # 计算每日RankIC
        rankic_by_day = df.groupby('date').apply(lambda group: group[factor].corr(group[ret], method='spearman'))
        # 累积RankIC
        cumulative_rankic = rankic_by_day.cumsum()
        # 计算Mean RankIC和RankICIR
        mean_rankic = rankic_by_day.mean()
        std_rankic = rankic_by_day.std()
        rankicir = mean_rankic / std_rankic
        
        # results['rankic_by_day'] = rankic_by_day
        # results['cumulative_rankic'] = cumulative_rankic
        # results['mean_rankic'] = mean_rankic
        # results['rankicir'] = rankicir
        
        plot_IC(rankic_by_day, cumulative_rankic, mean_rankic, rankicir, ic_type='rankIC',log=log,name=namea)
    
    if ic_type in ['IC', 'both']:
        # 计算每日IC
        ic_by_day = df.groupby('date').apply(lambda group: group[factor].corr(group[ret], method='pearson'))
        # 累积IC
        cumulative_ic = ic_by_day.cumsum()
        # 计算Mean IC和ICIR
        mean_ic = ic_by_day.mean()
        std_ic = ic_by_day.std()
        icir = mean_ic / std_ic
        
        
        plot_IC(ic_by_day, cumulative_ic, mean_ic, icir, ic_type='IC',log=log,name=nameb)
    


    import pandas as pd

def calculate_benchmark_return(df, factor,ret_overnight='ret_overnight',ret_day='ret_day',ret_daily='ret_daily'):
    """
    计算benchmark return及超额收益。
   
    """
    
    df = df.dropna(subset=[factor])
    
    # 计算benchmark return
    df['benchmark_overnight'] = df.groupby('date')[ret_overnight].transform('mean')
    df['benchmark_day'] = df.groupby('date')[ret_day].transform('mean')
    df['benchmark_daily'] = df['benchmark_overnight'] + df['benchmark_day']
    
    # 计算超额收益
    df['excess_overnight'] = df[ret_overnight] - df['benchmark_overnight']
    df['excess_day'] = df[ret_day] - df['benchmark_day']
    df['excess_daily'] = df[ret_daily] - df['benchmark_daily']
    
    return df


import pandas as pd
import numpy as np

def custom_quantile_grouping(df, date_col, factor_col, group_nums=10):
    """
    分组，和qcut没什么区别
    """
    df.sort_values([date_col, factor_col], inplace=True)

    df['quantile'] = np.nan  # 初始化组别列
    for date, group in df.groupby(date_col):
        n = len(group)
        base_size = n // group_nums
        remainder = n % group_nums
        
        sizes = [base_size + 1 if i < remainder else base_size for i in range(group_nums)]
        group_labels = np.hstack([np.repeat(i, size) for i, size in enumerate(sizes)])
        df.loc[group.index, 'quantile'] = group_labels
    
    return df



def calculate_daily_turnover(df, transaction_cost_rate=0.0012):
    """
    计算每个组合中每只股票的每日权重变化，求换手率，并根据换手率求transaction cost。
    得到longtable:每日每组的turnover_rate & transaction_cost
   
    """
    # 为每只股票计算权重（1/n)
    daily_counts = df.groupby(['date', 'quantile']).size().reset_index(name='count')
    df = df.merge(daily_counts, on=['date', 'quantile'], how='left')
    df['weight'] = 1 / df['count']

    # 权重矩阵
    df.sort_values(by=['order_book_id', 'date'], inplace=True)
    pivot_weights = df.pivot_table(index=['date', 'order_book_id'], columns='quantile', values='weight', fill_value=0)

    # 权重差，每日每组求和得到换手率（此处换手率：当天对比前一天的，可以理解为：在根据新的分组结果换仓之后的权重对比原来的变化）
    weight_changes = pivot_weights.groupby(level='order_book_id').diff().fillna(0)
    weight_changes = weight_changes.abs()
    turnover = weight_changes.groupby(level='date').sum()

    # 转下格式，计算下transaction cost
    turnover = turnover.reset_index().melt(id_vars=['date'], var_name='quantile', value_name='turnover_rate')
    turnover['transaction_cost'] = turnover['turnover_rate'] * transaction_cost_rate
    
    return turnover



import pandas as pd

def calculate_gross_excess_return(dft):
    """
    示例：pnl_result = calculate_gross_excess_return(dft)
    分别计算换仓前的隔夜收益和换仓后的当日收益，相加得到投资组合的每日超额收益（gross）

    """
    # 第一部分，换仓后的当日收益
    grouped = dft.groupby(['date', 'quantile'])['excess_day'].mean().reset_index()

    # 第二部分，换仓前的隔夜收益
    dft = dft.sort_values(['order_book_id', 'date'])
    dft['prev_quantile'] = dft.groupby('order_book_id')['quantile'].shift(1)
    dft = dft.dropna(subset=['prev_quantile'])

    grouped2 = dft.groupby(['date', 'prev_quantile'])['excess_overnight'].mean().reset_index()
    grouped2.rename(columns={'prev_quantile': 'quantile'}, inplace=True)
  

    # 合并结果并计算 gross excess return
    pnl_result = pd.merge(grouped, grouped2, on=['date', 'quantile'])
    pnl_result = pnl_result.dropna()
    pnl_result['gross_excess_return'] = pnl_result['excess_day'] + pnl_result['excess_overnight']

    return pnl_result



def get_pnl(df):
    """
    调取函数，计算gross excess return和turnover&transaction cost
    随后计算net excess return和cumulative return
    """
    # 调取函数，计算gross excess return和turnover&transaction cost
    pnl_result = calculate_gross_excess_return(df)
    turnover = calculate_daily_turnover(df)

    # net excess return
    pnl_result = pd.merge(pnl_result, turnover, on = ['date','quantile'], how = 'left')    
    pnl_result['net_excess_return'] = pnl_result['gross_excess_return'] - pnl_result['transaction_cost']

    # 累计收益
    pnl_result['cum_net_excess'] = pnl_result.groupby('quantile')['net_excess_return'].cumsum()
    pnl_result['cum_gross_excess'] = pnl_result.groupby('quantile')['gross_excess_return'].cumsum()

    return pnl_result



# 画图：分组累计超额收益
import matplotlib.pyplot as plt
import seaborn as sns
def plot_cumulative_returns(cumulative_returns,log=False,name=None):

    plt.figure(figsize=(10, 5))
    for quantile in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[quantile], label=f'Group {quantile + 1}')
        
    plt.xlabel('Date')
    plt.ylabel('Cumulative Excess Return')
    plt.title('Cumulative Excess Returns by Group')
    plt.legend()
    plt.grid(True)
    if log:
        plt.savefig(name)
    else:
        plt.show()


import pandas as pd
import numpy as np

def calculate_annual_metrics(pnl_result, annual_days=250):
    """
    得到包含：年化收益率、夏普比率和换手率的表格
    
    """
    # 年化净超额收益率
    annual_net_excess = pnl_result.groupby('quantile')['net_excess_return'].mean() * annual_days
    
    # 年化总超额收益率
    annual_gross_excess = pnl_result.groupby('quantile')['gross_excess_return'].mean() * annual_days
    
    # 计算净收益的夏普比率
    annual_net_std = pnl_result.groupby('quantile')['net_excess_return'].std() * np.sqrt(annual_days)
    sharpe_net = annual_net_excess / annual_net_std
    sharpe_net = sharpe_net.replace([np.inf, -np.inf], 0)
    
    # 计算总收益的夏普比率
    annual_gross_std = pnl_result.groupby('quantile')['gross_excess_return'].std() * np.sqrt(annual_days)
    sharpe_gross = annual_gross_excess / annual_gross_std
    sharpe_gross = sharpe_gross.replace([np.inf, -np.inf], 0)
    
    # 计算每组的平均换手率
    turnover_rate = pnl_result.groupby('quantile')['turnover_rate'].mean()
    
    results = pd.DataFrame({
        'net_excess_return': annual_net_excess,
        'gross_excess_return': annual_gross_excess,
        'sharpe_net': sharpe_net,
        'sharpe_gross': sharpe_gross,
        'turnover_rate': turnover_rate
    })
    
    return results