import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train = pd.read_csv('Train.csv')
test = pd.read_csv('test.csv')

from sklearn.linear_model import LinearRegression

lreg = LinearRegression()

# splitting into training and cv for cross validation
X = train.loc[:,['Outlet_Establishment_Year','Item_MRP']]

x_train, x_cv, y_train, y_cv = train_test_split(X,train.Item_Outlet_Sales)
train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)

lreg.fit(x_train,y_train)
pred_cv = lreg.predict(x_cv)
mse = np.mean((pred_cv - y_cv)**2)

coeff = pd.DataFrame(x_train.columns)
coeff['Coefficient Estimate'] = pd.Series(lreg.coef_)
rsq = lreg.score(x_cv,y_cv)                           # what % of variation in y is explained by variation of X

x_plot = plt.scatter(pred_cv, (pred_cv - y_cv), c='b')
plt.hlines(y=0, xmin= -1000, xmax=5000)
plt.title('Residual plot')

# Ridge and Lasso regressions use regularization techniques to reduce magnitude of coefficients
# Ridge regression uses L2 regularization technique to prevent multicollinearity and reduce model complexity by shrinking coefs
# alpha is the hyperparameter of Rigde is manually set (higher alpha -> bigger penalty -> lower coefs)
from sklearn.linear_model import Ridge

ridgeReg = Ridge(alpha=0.05,normalize=True)           
ridgeReg.fit(x_train,y_train)
pred_cv = ridgeReg.predict(x_cv)
mse = np.mean((pred_cv - y_cv)**2)
rsq = ridgeReg.score(x_cv,y_cv)

# LASSO (Least Absolute Shrinkage Selector Operator) zeroes out some coefficients (feature selection)
# Lasso regression uses L1 regularization technique by shrinking coefs & feature selection (removing correlated variables)
from sklearn.linear_model import Lasso

lassoReg = Lasso(alpha=0.3,normalize=True)
lassoReg.fit(x_train,y_train)
pred_cv = lassoReg.predict(x_cv)
mse = np.mean((pred_cv - y_cv)**2)
rsq = lassoReg.score(x_cv,y_cv)

# Elastic net regression is a hybrid between Ridge and LASSO
# EN regression uses both L1 and L2 regularization technique by shrinking coefs & feature selection (removing correlated variables)
from sklearn.linear_model import ElasticNet

ENreg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
ENreg.fit(x_train,y_train)
pred_cv = ENreg.predict(x_cv)
mse = np.mean((pred_cv - y_cv)**2)
ENreg.score(x_cv,y_cv)


# https://www.kaggle.com/code/jnikhilsai/cross-validation-with-linear-regression/notebook

