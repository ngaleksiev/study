import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import statsmodels.api as sm

## create data
#x = np.linspace(-5, 50, 100)
#y = 50 + 2 * x + np.random.normal(0, 20, size=len(x))
#data = pd.DataFrame({'X': x, 'Y': y})
#
##ax = data.plot.scatter(x='X', y='Y', figsize=(10, 6))
##sns.despine()
##plt.tight_layout()
#
#y = data.loc[:,'Y']                                                             # set dependent variable Y
#X = sm.add_constant(data.loc[:,'X'])                                            # set independent varible X and add constant 
#model = sm.OLS(y,X).fit()                                                       # set up model (Y,X).fit()
#print(model.summary())
#
#data['y-hat'] = model.predict()
#data['residuals'] = model.resid
#
#ax1 = data.plot.scatter(x='X',y='Y',figsize=(10,6))                              # scatter of the Xs and the Ys
#data.plot.line(x='X',y='y-hat',ax=ax1)                                           # line of predicted values on same ax (plot)
#
#for i, row in data.iterrows():
#    plt.plot((row.X,row.X),(row.Y,row['y-hat']),c='grey')
#plt.tight_layout()                                                               # adjusts padding


# MULTIPLE REGRESSION
## Create data
size = 25
X_1, X_2 = np.meshgrid(np.linspace(-50, 50, size), np.linspace(-50, 50, size), indexing='ij')
data = pd.DataFrame({'X_1': X_1.ravel(), 'X_2': X_2.ravel()})
data['Y'] = 50 + data.X_1 + 3 * data.X_2 + np.random.normal(0, 50, size=size**2)

y = data.loc[:,'Y']
X = sm.add_constant(data.loc[:,['X_1','X_2']])

model = sm.OLS(y,X).fit()
print(model.summary())

fig = plt.figure(figsize=(10,6)).gca(projection='3d')                             # gca gets current axes instance of current fig
fig.scatter(data.loc[:,'X_1'],data.loc[:,'X_2'],data.loc[:,'Y'],c='green')
data.loc[:,'y-hat'] = model.predict()
to_plot = data.set_index(['X_1', 'X_2']).unstack().loc[:, 'y-hat']
fig.plot_surface(X_1, X_2, to_plot.values, color='black', alpha=0.2, linewidth=1, antialiased=True)
for _, row in data.iterrows():
    plt.plot((row.X_1, row.X_1), (row.X_2, row.X_2), (row.Y, row['y-hat']), 'k-');
fig.set_xlabel('$X_1$')
fig.set_ylabel('$X_2$')
fig.set_zlabel('$Y, \hat{Y}$')

sns.despine()
plt.tight_layout();


# Gradient descend
#   takes steps to calculate the derivatives of the loss function (until max number of steps reached or steps become too small) 
#   useful when not possible to solve for where the derivative = 0
# STOCHASTIC GRADIENT DESCEND REGRESSION
#   uses only a randomly selected subset of the data at every step rather than full dataset
#   reduces time when lots of data and parameters
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler                        # SGD is sensitive to scale so we use StandardScaler to standardize the data

scaler = StandardScaler()
Xft = scaler.fit_transform(X)                                            # fit calculates mean and stdv; transform does (X-mean)/stdv

sgd = SGDRegressor(                                                     # SGDRegressor with default values ex random_state
        loss='squared_loss',                                            # or cost function to be minimized in optimization
        fit_intercept=True,
        random_state=42,                                                # integer for reproducible output
        learning_rate='invscaling',                                     # starts large -> smaller
        eta0=0.01,
        power_t=0.25
        )

sgd.fit(X=Xft,y=y)
coeffs = (sgd.coef_ * scaler.scale_) + scaler.mean_
pd.Series(coeffs,index=X.columns)

resids = pd.DataFrame({
        'sgd':y-sgd.predict(Xft),
        'ols':y-model.predict(sm.add_constant(X))
        })
