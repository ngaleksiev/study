from datetime import datetime
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

idx = pd.IndexSlice

DATA_STORE = 'X:/Nikolay/personal/study/ML4T/data/assets.h5'
with pd.HDFStore(DATA_STORE) as store:
    sp500 = store['sp500/stooq'].loc['2009': '2010', 'close']
    