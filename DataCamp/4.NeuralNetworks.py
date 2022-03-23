# Neural networks are similar to other models - we supply features and targets to get predictions
# NN can outperform other models because they have nonlinearity , capture variable interactions and are highly customizable
# each row is a layer of neurons (linear algebra math). we send input data, multiply by weights and add a bias. 
# after data goes through layers we apply an activation function (ReLU - rectified linear units) to add non-linearity. ReLU is 0 for negative numbers, linear for positive
# once we have predictions (from single output node) we use loss function to compare predictions and targets. for regression we use MSE
# we use error from loss function and pass it back to update weights and biases (backpropagation)
# in keras library with tensorflow back end we can use sequential or functional API
# when training data fits well but not test we are overfitting 1) decrease nodes, 2) use L1/L2 regularization, 3) dropout some neurons, 4) autoencoder architecture
#                                                              5) early stopping, 6) adding noise to data, 7) max norm constraints, 8) ensembling

from keras.models import Sequential
from keras.layers import Dense

# Create the model
model_1 = Sequential()

# define number of nodes in each layer; number of features input_dim (# cols) and activation for each node
model_1.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(20, activation='relu'))
model_1.add(Dense(1, activation='linear'))

# Fit the model with optimized (how fast the model learns) and loss function. number of training cycles (epochs)
model_1.compile(optimizer='adam', loss='mse')
history = model_1.fit(scaled_train_features, train_targets, epochs=25)

# Plot the losses from the fit vs epochs to make sure curve flattened
plt.plot(history.history['loss'])
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()

from sklearn.metrics import r2_score

# Calculate R^2 score
train_preds = model_1.predict(scaled_train_features)
test_preds = model_1.predict(scaled_test_features)
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Plot predictions vs actual
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label='test')
plt.legend()
plt.show()

# define a custom loss function to use instead of MSE with penalty for wrong direction
import keras.losses
import tensorflow as tf

def sign_penalty(y_true, y_pred):
    penalty = 100.
    #                boolean matrix (true/false)            if true * penalty                 if false don't
    loss = tf.where(tf.less(y_true * y_pred, 0), penalty * tf.square(y_true - y_pred), tf.square(y_true - y_pred))
    return tf.reduce_mean(loss, axis=-1)                          # take average of the loss (square errors) across the last axis -1

keras.losses.sign_penalty = sign_penalty  # enable use of loss with keras
print(keras.losses.sign_penalty)


# we can ensamble (average predictions) severale models (mse, custom loss funct, with dropout) to improve performance (predictions) and to avoid overfitting
train_pred1 = model_1.predict(scaled_train_features)
test_pred1  = model_1.predict(scaled_test_features)

train_pred2 = model_2.predict(scaled_train_features)
test_pred2  = model_2.predict(scaled_test_features)

train_pred3 = model_3.predict(scaled_train_features)
test_pred3  = model_3.predict(scaled_test_features)

# Horizontally stack predictions and take the average across rows
train_preds = np.mean(np.hstack((train_pred1, train_pred2, train_pred3)), axis=1)
test_preds  = np.mean(np.hstack((test_pred1, test_pred2, test_pred3)), axis=1)
print(test_preds[-5:])


























