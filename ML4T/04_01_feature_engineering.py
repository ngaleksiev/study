# https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/04_alpha_factor_research/01_feature_engineering.ipynb

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import pandas as pd

#from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

DATA_STORE = 'X:/Nikolay/Personal/Study/ML4T/data/assets.h5'
START = 2000
END = 2018

with pd.HDFStore(DATA_STORE) as store:
    prices = store['quandl/wiki/prices']                                            # load the .h5 file into a df
    prices = prices.loc[str(START):str(END),:]                                      # get only for date range using index slice 
    prices = prices.loc[:,'adj_close']
    prices = prices.unstack('ticker')                                               # leave date as only index
    
    stocks = store['us_equities/stocks']
    stocks = stocks.loc[:,['marketcap','ipoyear','sector']]
    stocks = stocks[~stocks.index.duplicated()]                                     # remove duplicates in the index
    
shared = prices.columns.intersection(stocks.index)                              # get the list of tickers with both prices and data

stocks = stocks.loc[shared,:]
prices = prices.loc[:,shared]

mprices = prices.resample('M').last()

#data = pd.DataFrame()
#for lag in [1,2,3]:
#    data[f'return_{lag}m'] = mprices.pct_change(lag)


data = pd.DataFrame()
lags = [1, 2, 3, 6, 9, 12]
for lag in lags:
    data[f'return_{lag}m'] = (mprices
                           .pct_change(lag)
                           .stack()                                                                    # stack tickers as second index
                           .pipe(lambda x: x.clip(lower=x.quantile(0.01),upper=x.quantile(1-0.01)))    # use pipe to winsorize with clip 
                           .add(1).pow(1/lag).sub(1)                                                   # normalize using geo mean for 1m returns
                           )
    
data = data.swaplevel().dropna()

data = data.loc[data.groupby(level='ticker').size()[data.groupby(level='ticker').size()>120].index,:] # keep only data with 10 years of monthly returns
data.index.get_level_values('ticker').nunique()                                                       # or len(.unique())


import pandas_datareader.data as web
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2000')[0].loc[:,factors]
#factor_data = factor_data.drop('RF',axis=1)                                                          # drop a column 'RF'
factor_data.index = factor_data.index.to_timestamp()                                                  # change month to timestamp
factor_data = factor_data.resample('M').last().div(100)                                               # or /100
factor_data.index.name = 'date'

#factor_data = factor_data.join(data['return_1m']).sort_index()
factor_data = pd.merge(factor_data,data['return_1m'],left_index=True,right_index=True)

from statsmodels.regression.rolling import RollingOLS

betas = (factor_data.groupby(level='ticker',group_keys=False)                                         # subdf for each ticker (x) and remove key
    .apply(                                                                                           # apply the OLS function
            lambda x: RollingOLS(
                    endog = x['return_1m'],                                                           # dependent var (1d endogenous) - returns
                    exog = sm.add_constant(x.drop('return_1m',axis=1)),                               # n x k array where n is observations and k is indeps.
                    window=min(24,x.shape[0]-1)                                                       # window min of 24m or shape[0] (months) - 1
            )                                                                                          
    .fit(params_only=True)                                                                            # skip all other calculations except parameter est.
    .params.drop('const',axis=1)                                                                      # drop the constant
    )
)

#cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.heatmap(betas.corr(),annot=True,center=0)

#betas = betas.groupby(level='ticker')
data = pd.merge(data,betas,left_index=True,right_index=True)
data.loc[:,factors] = data.groupby(level='ticker')[factors].apply(lambda x: x.fillna(x.mean()))       # fill NAs with mean of ticker's factors

for lag in [2,3,6,9,12]:
    data[f'momentum_{lag}'] = data.loc[:,f'return_{lag}m'] - data.loc[:,'return_1m']
data['momentum_3_12'] = data.loc[:,'momentum_12'] - data.loc[:,'momentum_3']

data['year']  = data.index.get_level_values('date').year
data['month'] = data.index.get_level_values('date').month

for t in range(1,7):
    data[f'return_1m_t-{t}'] = data.groupby(level='ticker')['return_1m'].shift(t)

for t in [1,2,3,6,12]:
    data[f'target_{t}m'] = data.groupby(level='ticker')[f'return_{t}m'].shift(-t)

data = pd.merge(data,pd.qcut(stocks.ipoyear,q=5,labels=range(1,6)).astype(float).fillna(0).astype(int).to_frame('age'),left_index=True,right_index=True)
data.age = data.age.fillna(-1)

monthly_prices = prices.resample('M').last()
size_factor = (monthly_prices.loc[data.index.get_level_values('date').unique(),data.index.get_level_values('ticker').unique()]   # get monthly prices
                    .sort_index(ascending=False)                                                                                 # reverse sort
                    .pct_change().fillna(0)                                                                                      # get % return
                    .add(1).cumprod()                                                                                            # calc cuml ret
                    )

msize = size_factor.mul(stocks.loc[size_factor.columns,'marketcap']).dropna(axis=1,how='all')
data['msize'] = (
        msize.apply(lambda x: pd.qcut(x,q=10,labels=list(range(1,11))).astype(int),axis=1)                     # size deciles for each month (axis1 is cols)
        .stack()                                                                                               # stack columns (tickers) as 2nd index
        .swaplevel()                                                                                           # swap level of indexes date <-> ticker
        )
data['msize'] = data['msize'].fillna(-1)

data = pd.merge(data,stocks['sector'],left_index=True,right_index=True).fillna('Unknown')

idx = pd.IndexSlice
with pd.HDFStore(DATA_STORE) as store:
    store.put('engineered_features',data.sort_index().loc[idx[:,:datetime(2018,3,1)],:])                       # idx=pd.IndexSlice for select mult index

dummy_data = pd.get_dummies(data,                                                                              # pd.get_dummies to creaty dummy columns
                            columns=['year','month','msize','age','sector'],                                   # which columns list
                            prefix=['year','month','msize','age',''],                                          # column prefix list
                            prefix_sep=['_','_','_','_',''])                                                   # separator 

#dummy_data = dummy_data.columns.replace('.0','')
dummy_data = dummy_data.rename(columns={c:c.replace('.0','') for c in dummy_data.columns})                     # df.rename(columns={old:new}) use a dict
