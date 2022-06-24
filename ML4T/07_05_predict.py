import pandas as pd
import numpy as np

with pd.HDFStore('X:/Nikolay/Personal/Study/ML4T/data/data.h5') as store:
    data = (store['model_data']
            .dropna()
            .drop(['open', 'close', 'low', 'high'], axis=1))
data.index.names = ['symbol', 'date']
data = data.drop([c for c in data.columns if 'lag' in c], axis=1)
data = data[data.loc[:,'dollar_vol_rank']<100]

y = data.filter(like='target',axis=1)
X = data.drop(y.columns,axis=1)
X = X.drop(['dollar_vol', 'dollar_vol_rank', 'volume', 'consumer_durables'], axis=1)

# set up cross-validation
class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(self,
                 n_splits=3,
                 train_period_length=126,
                 test_period_length=21,
                 lookahead=None,
                 shuffle=False):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values('date').unique()
        days = sorted(unique_dates, reverse=True)

        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx,
                              test_start_idx, test_end_idx])

        dates = X.reset_index()[['date']]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(dates.date > days[train_start])
                              & (dates.date <= days[train_end])].index
            test_idx = dates[(dates.date > days[test_start])
                             & (dates.date <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

#YEAR = 252
#train_period_length = 63
#test_period_length = 10
#n_splits = int(3 * YEAR/test_period_length)
#lookahead = 1
#
#cv = MultipleTimeSeriesCV(n_splits=n_splits,
#                          test_period_length=test_period_length,
#                          lookahead=lookahead,
#                          train_period_length=train_period_length)

#i = 0
#for train_idx, test_idx in cv.split(X=data):
#    train = data.iloc[train_idx]
#    train_dates = train.index.get_level_values('date')
#    test = data.iloc[test_idx]
#    test_dates = test.index.get_level_values('date')
#    df = train.reset_index().append(test.reset_index())
#    n = len(df)
#    assert n== len(df.drop_duplicates())
#    print(train.groupby(level='symbol').size().value_counts().index[0],
#          train_dates.min().date(), train_dates.max().date(),
#          test.groupby(level='symbol').size().value_counts().index[0],
#          test_dates.min().date(), test_dates.max().date())
#    i += 1
#    if i == 10:
#        break

#from sklearn.linear_model import LinearRegression, Ridge, Lasso
#from sklearn.metrics import mean_squared_error
#from scipy.stats import spearmanr
#
## run cross-validation with linear regression
#target = f'target_{lookahead}d'
#lr_predictions, lr_scores = [], []
#lr = LinearRegression()
#for i, (train_idx, test_idx) in enumerate(cv.split(X), 1):
#    X_train, y_train = X.iloc[train_idx], y[target].iloc[train_idx]
#    X_test,  y_test  = X.iloc[test_idx],  y[target].iloc[test_idx]
#    lr.fit(X=X_train, y=y_train)
#    y_pred = lr.predict(X_test)
#
#    preds = y_test.to_frame('actuals').assign(predicted=y_pred)
#    preds_by_day = preds.groupby(level='date')
#    scores = pd.concat([preds_by_day.apply(lambda x: spearmanr(x.predicted,x.actuals)[0] * 100)
#                        .to_frame('ic'),
#                        preds_by_day.apply(lambda x: np.sqrt(mean_squared_error(y_pred=x.predicted,y_true=x.actuals)))
#                        .to_frame('rmse')], axis=1)
#
#    lr_scores.append(scores)
#    lr_predictions.append(preds)
#
#lr_scores = pd.concat(lr_scores)
#lr_predictions = pd.concat(lr_predictions)

# perform a Lasso regression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

train_period_length = 63
test_period_length  = 10
YEAR = 252
n_splits = int(3 * YEAR / test_period_length) # three years
lookahead = 1

cv = MultipleTimeSeriesCV(n_splits=n_splits, lookahead=lookahead, test_period_length=test_period_length, train_period_length=train_period_length)

target = f'target_{lookahead}d'
X = X.drop([c for c in X.columns if 'year' in c], axis=1)

lasso_alphas = np.logspace(-10, -3, 8)                                          # alphas for the regressions [0.0000000001, ..., 0.001]
lasso_coeffs, lasso_scores, lasso_predictions = {}, [], []
for alpha in lasso_alphas:
    print(alpha, end=' ', flush=True)
    model = Lasso(alpha=alpha,
                  fit_intercept=False,  # StandardScaler centers data
                  selection='random',   # random coef is updated in every iteration - leads to faster convergence
                  random_state=42,      # used when selection is 'random' and pass an int for a reproducible output
                  tol=1e-3,             # tolerance for the optimization 
                  max_iter=1000,        # maximum number of iterations
                  warm_start=True)      # True - reuses the solution of previous call to fit as initialization

    pipe = Pipeline([('scaler', StandardScaler()),      # pipeline allows for a sequence of data transformations followed by an estimator to simplify
                     ('model', model)])                 # used to avoid leaking the test set into the train set 
    coeffs = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        X_train, y_train, = X.iloc[train_idx], y[target].iloc[train_idx]
        X_test, y_test    = X.iloc[test_idx], y[target].iloc[test_idx]

        pipe.fit(X_train,y_train)                      # fit the model through the pipeline
        y_pred = pipe.predict(X_test)                  # predict the model through the pipeline

        preds = y_test.to_frame('actuals').assign(predicted=y_pred)
        preds_by_day = preds.groupby(level='date')
        # scipy.stats.spearmanr to calculate rho(IC) [0] and p-val [1] if needed between predicted (y_pred) and actual (y_test)
        # sklearn.metrics.mean_squared_error to get rmse between predicted (y_pred) and actual (y_test)
        scores = pd.concat([preds_by_day.apply(lambda x: spearmanr(x.predicted,x.actuals)[0] * 100).to_frame('ic'),     
                            preds_by_day.apply(lambda x: np.sqrt(mean_squared_error(x.predicted,x.actuals))).to_frame('rmse')], 
                           axis=1)

        lasso_scores.append(scores.assign(alpha=alpha))
        lasso_predictions.append(preds.assign(alpha=alpha))

        coeffs.append(pipe.named_steps['model'].coef_)

    lasso_coeffs[alpha] = np.mean(coeffs, axis=0)

lasso_scores = pd.concat(lasso_scores)
lasso_scores.to_hdf('data.h5', 'lasso/scores')

lasso_coeffs = pd.DataFrame(lasso_coeffs, index=X.columns).T
lasso_coeffs.to_hdf('data.h5', 'lasso/coeffs')

lasso_predictions = pd.concat(lasso_predictions)
lasso_predictions.to_hdf('data.h5', 'lasso/predictions')

# evaluate Lasso results to find best alpha and IC, p-val associated with it
best_alpha = lasso_scores.groupby('alpha').ic.mean().idxmax()                   # pick best alpha by best avg IC by alpha
preds = lasso_predictions[lasso_predictions.loc[:,'alpha']==best_alpha]

lasso_r, lasso_p = spearmanr(preds.actuals, preds.predicted)
print(f'Information Coefficient (Best Alpha overall): {lasso_r:.3%} (p-value: {lasso_p:.4%})')

# Lasso coefficient path
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(ncols=2, sharex=True, figsize=(15, 5))                                         # set up the plot with 2 cols

scores_by_alpha = lasso_scores.groupby('alpha').ic.agg(['mean', 'median'])                              # calculate mean and median IC per alpha score
best_alpha_mean = scores_by_alpha['mean'].idxmax()                                                      # best alpha by mean IC
best_alpha_median = scores_by_alpha['median'].idxmax()                                                  # best alpha by median IC

ax = sns.lineplot(x='alpha', y='ic', data=lasso_scores, estimator=np.mean, label='Mean', ax=axes[0])

scores_by_alpha['median'].plot(logx=True, ax=axes[0], label='Median')

axes[0].axvline(best_alpha_mean, ls='--', c='k', lw=1, label='Max. Mean')
axes[0].axvline(best_alpha_median, ls='-.', c='k', lw=1, label='Max. Median')
axes[0].legend()
axes[0].set_xscale('log')
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('Information Coefficient')
axes[0].set_title('Cross Validation Performance')

lasso_coeffs.plot(logx=True, legend=False, ax=axes[1], title='Lasso Coefficient Path')
axes[1].axvline(best_alpha_mean, ls='--', c='k', lw=1, label='Max. Mean')
axes[1].axvline(best_alpha_median, ls='-.', c='k', lw=1, label='Max. Median')
axes[1].set_xlabel('Alpha')
axes[1].set_ylabel('Coefficient Value')

fig.suptitle('Lasso Results', fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=.9)
sns.despine()

# lasso distribution and top 10 features
def plot_ic_distribution(df, ax=None):
    if ax is not None:
        sns.distplot(df.ic, ax=ax)
    else:
        ax = sns.distplot(df.ic)
    mean, median = df.ic.mean(), df.ic.median()
    ax.axvline(0, lw=1, ls='--', c='k')
    ax.text(x=.05, y=.9,
            s=f'Mean: {mean:8.2f}\nMedian: {median:5.2f}',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel('Information Coefficient')
    sns.despine()
    plt.tight_layout()

best_alpha = lasso_scores.groupby('alpha').ic.mean().idxmax()

fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
plot_ic_distribution(lasso_scores[lasso_scores.alpha==best_alpha], ax=axes[0])
axes[0].set_title('Daily Information Coefficients')

top_coeffs = lasso_coeffs.loc[best_alpha].abs().sort_values().head(10).index
top_coeffs.tolist()
lasso_coeffs.loc[best_alpha, top_coeffs].sort_values().plot.barh(ax=axes[1], title='Top 10 Coefficients')

sns.despine()
fig.tight_layout()
