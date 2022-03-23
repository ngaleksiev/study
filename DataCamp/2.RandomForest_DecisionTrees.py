# decision trees are the building blocks of random forests. 
# decision trees split the data into groups based on features. trees start at root node and end with leaf nodes
# regression trees use a reduction in variance (spread of the data) to decide on best split
# spread out target values in leaves is a bad tree; tightly clustered target values (low variance) is a good split
# to avoid overfitting limit the three depth (or height)

from sklearn.tree import DecisionTreeRegressor

for d in [3,5,10]:
    # Create a decision tree regression model with default arguments
    decision_tree = DecisionTreeRegressor(max_depth=d)
    
    # Fit the model to the training features and targets
    decision_tree.fit(train_features,train_targets)
    
    # Check the RSQ score on train and test
    print(decision_tree.score(train_features, train_targets))
    print(decision_tree.score(test_features, test_targets))
    
    # Predict values for train and test
    train_predictions = decision_tree.predict(train_features)
    test_predictions = decision_tree.predict(test_features)
    
    # Scatter the predictions vs actual values
    plt.scatter(train_predictions, train_targets, label='train')
    plt.scatter(test_predictions, test_targets, label='test')
    plt.show()
    

# random forests can be used for classification or regression
# random forests reduce the variance of decision trees (split on smaller sample from bootstrapping) and balance between high variance and high bias.
# high variance (well on training data but not on test data) vs high bias (captures general trends but misses small details).
# we sample with replacement from training set (bootstrapping) to get datasets for each tree we fit (bootstrap aggregating/bagging). we can repeat or ommit data points.    
# max_features: sqrt(total features); max depth: 5 to 20; n_estimators (# trees): 20 to 200 as performance flattens; random_state=42 to reproduce results

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
import numpy as np

# Create the random forest model
rfr = RandomForestRegressor(n_estimators=200)

# Create a dictionary of hyperparameters to search
grid = {'n_estimators': [200], 'max_depth': [3], 'max_features': [4,8], 'random_state': [42]}
test_scores = []

# Loop through the parameter grid, set the hyperparameters, and save the scores
for g in ParameterGrid(grid):
    rfr.set_params(**g)  # ** is "unpacking" the dictionary
    rfr.fit(train_features, train_targets)
    test_scores.append(rfr.score(test_features, test_targets))

# Find best hyperparameters from the test score and print
best_idx = np.argmax(test_scores)
print(test_scores[best_idx], ParameterGrid(grid)[best_idx])

# Use the best hyperparameters from before to fit a random forest model
rfr = RandomForestRegressor(n_estimators=200, max_depth=3, max_features=4, random_state=42)
rfr.fit(train_features, train_targets)

# Make predictions with our model
train_predictions = rfr.predict(train_features)
test_predictions = rfr.predict(test_features)

# Create a scatter plot with train and test actual vs predictions
plt.scatter(train_targets, train_predictions, label='train')
plt.scatter(test_targets, test_predictions, label='test')
plt.legend()
plt.show()

# once we fit a model like random forest we can extract feature importances
importances = rfr.feature_importances_

# Get the index of importances from greatest importance to least
sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))

# Create tick labels 
labels = np.array(feature_names)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)

# Rotate tick labels to vertical
plt.xticks(rotation=90)
plt.show()

# linear models are simple to use while boosted models (general class of ML models) work better but difficult to use and interpret
# Gradient boosting - iteratively fitting models (like decision trees) to data first and then to residual errors of prev tree
# Adaboost - alternative to gradient boosting also fitting data iteratively

from sklearn.ensemble import GradientBoostingRegressor

# Create GB model -- hyperparameters have already been searched for you
gbr = GradientBoostingRegressor(max_features=4, learning_rate=0.01, n_estimators=200, subsample=0.6, random_state=42)
gbr.fit(train_features,train_targets)

print(gbr.score(train_features, train_targets))
print(gbr.score(test_features, test_targets))
