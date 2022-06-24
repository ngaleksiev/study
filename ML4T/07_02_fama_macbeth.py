import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from linearmodels.asset_pricing import LinearFactorModel

# average return on the factor portfolios ()
ff_factors = 'F-F_Research_Data_5_Factors_2x3'
ff_factors_data = web.DataReader(ff_factors,'famafrench',start='2010-01',end='2017-12')[0]


# average excess returns of the industry portfolios
ff_portfolios = '17_Industry_Portfolios'
ff_portfolios_data = web.DataReader(ff_portfolios,'famafrench',start='2010-01',end='2017-12')[0]
ff_portfolios_data = ff_portfolios_data.sub(ff_factors_data.loc[:,'RF'],axis=0)

with pd.HDFStore('X:/Nikolay/Personal/Study/ML4T/data/assets.h5') as store:
    prices = store['/quandl/wiki/prices'].loc[:,'adj_close'].unstack().loc['2010':'2017',:]              # unstack - 2nd index to cols; date x ticker df
    equities = store['/us_equities/stocks'].drop_duplicates()
    
sectors = equities.filter(prices.columns,axis=0).loc[:,'sector'].to_dict()                               # df.filter(list,axis); creaters tic[sec] dict
prices = prices.filter(sectors.keys(),axis=1).dropna(how='all',axis=1)                                   # prices for equities only

returns = prices.resample('M').last().pct_change().mul(100).to_period('M')                               # calc returns; .to_period('M') index date to month
returns = returns.dropna(how='all').dropna(axis=1)                                                       # drops any nas - only values left

ff_factors_data = ff_factors_data.loc[returns.index,:]
ff_portfolios_data = ff_portfolios_data.loc[returns.index,:]

excess_returns = returns.sub(ff_factors_data.loc[:,'RF'],axis=0)
excess_returns = excess_returns.clip(lower=np.percentile(excess_returns,1),upper=np.percentile(excess_returns,99)) # winzorize with clip at 1 and 99 percent

ff_factors_data = ff_factors_data.drop('RF',axis=1)

# 1. Factor exposures/loadings - N time-series regressions (for each industry) on all the factors
tbetas = []
for industry in ff_portfolios_data.columns:                                                                                  # for each industry
    step1 = sm.OLS(endog=ff_portfolios_data.loc[ff_factors_data.index,industry],exog=sm.add_constant(ff_factors_data)).fit() # y=ind rets, x=factors rets
    tbetas.append(step1.params.drop('const'))
betas = pd.DataFrame(tbetas,columns=ff_factors_data.columns,index=ff_portfolios_data.columns)

# 2. Risk Premia - T cross-sectional regression (one for each period)
tlambdas = []
for period in ff_portfolios_data.index:
    step2 = sm.OLS(endog=ff_portfolios_data.loc[period,betas.index],exog=betas).fit()
    tlambdas.append(step2.params)
lambdas = pd.DataFrame(tlambdas,index=ff_portfolios_data.index,columns=betas.columns.tolist())
lambdas.mean().sort_values().plot.barh(figsize=(10,6))
