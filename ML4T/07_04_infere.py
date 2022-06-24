import pandas as pd

from statsmodels.api import OLS, add_constant, graphics
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
idx = pd.IndexSlice

with pd.HDFStore('X:/Nikolay/Personal/Study/ML4T/data/data.h5') as store:
    data = (store['model_data'].dropna().drop(['open', 'close', 'low', 'high'], axis=1))
    data = data[data.loc[:,'dollar_vol_rank']<100]

y = data.filter(like='target',axis=1)
X = data.drop(y.columns,axis=1)
X = X.drop(['dollar_vol', 'dollar_vol_rank', 'volume', 'consumer_durables'], axis=1)

#sns.clustermap(y.corr(),center=0,annot=True,fmt='.1%')

sectors = X.iloc[:,-10:]
X = X.drop(sectors.columns,axis=1)
X = X.groupby(level='ticker').transform(lambda x: (x - x.mean()) / x.std())               # standardize values (ex sectors) on ticker level 
X = X.join(sectors).fillna(0)                                                             # add back sectors

model = OLS(endog=y['target_5d'],exog=add_constant(X)).fit()
print(model.summary())
preds = model.predict(add_constant(X))
resids = y['target_5d'] - preds

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(10,6))
sns.distplot(resids,fit=norm,ax=axes[0],axlabel='Residuals')
axes[0].set_title('Residuals Distribution')
axes[0].legend()

# check for autocorrelation of residuals
plot_acf(resids,lags=10,ax=axes[1],title='Residuals Autocorrelation')
axes[1].set_xlabel('Lags')
# or durbin watson test (if close to 2 no correlation)
from statsmodels.stats.stattools import durbin_watson
dwv = durbin_watson(resids)

sns.despine()
fig.tight_layout()
