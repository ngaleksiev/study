# https://github.com/stefan-jansen/machine-learning-for-trading/blob/d938ca42b136ac93eb78e25c86cc27d330cc03d9/data/create_datasets.ipynb

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import requests
from io import BytesIO
from zipfile import ZipFile, BadZipFile

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.datasets import fetch_openml

pd.set_option('display.expand_frame_repr', False)

# set up a fast and scalable HDF5 storage format (hierarchical data format) available in pandas using the PyTables library
DATA_STORE = Path('X:/Nikolay/Personal/Study/ML4T/data/assets.h5')

## downloaded WIKI-PRICES from Nasdaq in a .csv format
## read the .csv into a dataframe
## store the dataframe into the DATA_STORE
#df = (pd.read_csv('X:/Nikolay/Personal/Study/ML4T/data/wiki_prices.csv',
#                 parse_dates=['date'],
#                 index_col=['date', 'ticker'],
#                 infer_datetime_format=True)
#     .sort_index())
#with pd.HDFStore(DATA_STORE) as store:
#    store.put('quandl/wiki/prices', df)

## downloaded wiki-stocks from Nasdaq in a .csv format
## read the .csv into a dataframe
## store the dataframe into the DATA_STORE
#df = pd.read_csv('X:/Nikolay/Personal/Study/ML4T/data/wiki_stocks.csv')
#with pd.HDFStore(DATA_STORE) as store:
#    store.put('quandl/wiki/stocks', df)


## use pandas_datareader.data as web
#df = web.DataReader(name='SP500', data_source='fred', start=2009).squeeze().to_frame('close')
#print(df.info())
#with pd.HDFStore(DATA_STORE) as store:
#    store.put('sp500/fred', df)

## use pd.read_html data table
#url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#df = pd.read_html(url, header=0)[0]
#df.columns = ['ticker', 'name', 'sec_filings', 'gics_sector', 'gics_sub_industry','location', 'first_added', 'cik', 'founded']
#df = df.drop('sec_filings', axis=1).set_index('ticker')
#with pd.HDFStore(DATA_STORE) as store:
#    store.put('sp500/stocks', df)

df = pd.read_csv('X:/Nikolay/Personal/Study/ML4T/data/us_equities_meta_data.csv')
with pd.HDFStore(DATA_STORE) as store:
    store.put('us_equities/stocks', df.set_index('ticker'))
    