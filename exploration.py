# https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/misc/tesla-study.ipynb

# https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/17_deep_learning/05_backtesting_with_zipline.ipynb
# https://github.com/PacktPublishing/Machine-Learning-for-Finance-video

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.figure().clear()

tsla = pd.read_csv('X:/Nikolay/Personal/Study/TSLA.csv')
tsla = tsla[['Date','Open','High','Low','Close']]
tsla['Month'] = pd.DatetimeIndex(tsla['Date']).month
tsla['Year']  = pd.DatetimeIndex(tsla['Date']).year
tsla['Close_mean'] = np.mean(tsla['Close'])
tsla['Date'] = pd.DatetimeIndex(tsla['Date'])
tsla['TimeIndex'] = tsla['Date'] - tsla['Date'].min()
tsla['TimeIndex'] = (tsla['TimeIndex'] / np.timedelta64(1,'D')).round(0).astype(int)

## explore dataframe data with matplotlib
#plt.figure(1)
#tsla.plot(figsize=(10,5),kind='line',x='Date',y=['Open','High','Low','Close'])
#plt.figure(2)
#
#tslaPivot = pd.pivot_table(tsla,values='Close',columns='Year',index='Month')
#tslaPivot.plot(kind='line',figsize=(10,5),subplots=True)
#plt.figure(3)
#tsla.plot(figsize=(10,5),kind='hist',bins=20,y='Close')
#
#plt.figure(4)
#tsla.plot(kind='line',figsize=(10,5),x='Date',y=['Close','Close_mean'])


# linear regression with sklearn linear_model
from sklearn import linear_model

# x is 1..n dates, y is prices reshaped to (251,1)
x = np.arange(tsla.shape[0]).reshape((-1,1))
y = tsla['Close'].values.reshape((-1,1))
reg = linear_model.LinearRegression()
pred = reg.fit(x,y).predict(x)
tsla['linreg'] = pred

plt.figure(5)
tsla.plot(kind='line',figsize=(10,5),x='Date',y=['Close','Close_mean','linreg'])


import statsmodels as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller

model_linear = smf.ols('Close~TimeIndex',data=tsla).fit()
print(model_linear.summary())
print(model_linear.params)
model_linear_pred = model_linear.predict()
print(model_linear_pred.shape)
tsla['linear_stats'] = model_linear_pred
model_linear.resid.plot(kind='bar').get_xaxis().set_visible(False)
