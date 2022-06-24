import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from talib import RSI, BBANDS, MACD

idx = pd.IndexSlice

DATA_STORE = 'X:/Nikolay/personal/study/ML4T/data/assets.h5'
with pd.HDFStore(DATA_STORE) as store:
    data = store['quandl/wiki/prices']
    data = data.loc[idx['2007':'2010','AAPL'],['adj_open','adj_high','adj_low','adj_close','adj_volume']]
    data = data.unstack('ticker').swaplevel(axis=1)
    data = data.loc[:,'AAPL'].rename(columns=lambda x: x.replace('adj_','')) 
    
    data = data.loc['AAPL'].rename(columns={c:c.replace('_adj','') for c in data.columns})

up, mid, low = BBANDS(data.loc[:,'close'],timeperiod=21,nbdevup=2,nbdevdn=2,matype=0)
rsi = RSI(data.loc[:,'close'],timeperiod=14)
macd,macdsignal,macdhist = MACD(data.loc[:,'close'],fastperiod=12,slowperiod=26,signalperiod=9)

macd_data = pd.DataFrame({'AAPL':data.loc[:,'close'],'MACD':macd,'MACD Signal':macdsignal,'MACD History':macdhist})

fig,axes = plt.subplot(figsize=(10,6))
macd_data.loc['AAPL',:].plot(ax=axes[0])
macd_data.drop('AAPL',axis=1).plot(ax=axes[1])
fig.tight_layout()
sns.despine()

