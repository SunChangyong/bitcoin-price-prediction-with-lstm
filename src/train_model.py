import json
import requests

from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from pandas.plotting import register_matplotlib_converters

import read_data as etl
import plot_data

# print(hist.tail(5))

sns.set_palette('Set2')
register_matplotlib_converters()

target_col = 'close'
train, test = etl.train_test_split(hist, test_size=0.1)
plot_data.line_plot(train[target_col], test[target_col], 'training', 'test', title='BTC')
plt.show()

def build_lstm_model(input_data, output_size, neurons=20, activ_func='linear',
                     dropout=0.25, loss='mae', optimizer='adam'):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

np.random.seed(42)

# data params
window_len = 7
test_size = 0.1
zero_base = True

# model params
lstm_neurons = 20.
epochs = 50
batch_size = 4
loss = 'mae'
dropout = 0.25
optimizer = 'adam'

train, test, X_train, X_test, y_train, y_test = etl.prepare_data(
    hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)
history = model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

model.save('./model/model.h5')

# Plot predictions
targets = test[target_col][window_len:]
preds = model.predict(X_test).squeeze()

#print(mean_absolute_error(preds, y_test))
# 0.045261384613638447 with mae

preds = test[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
print (preds.tail(1))


plot_data.line_plot(targets, preds, 'actual', 'prediction', lw=3)
plt.show()

n_points = 30
plot_data.line_plot(targets[-n_points:], preds[-n_points:], 'actual', 'prediction', lw=3)
plt.show()

plot_data.line_plot(targets[-n_points:][:-1], preds[-n_points:].shift(-1), 'actual', 'prediction', lw=3)
plt.show()

# Compare returns
actual_returns = targets.pct_change()[1:]
predicted_returns = preds.pct_change()[1:]

plot_data.dual_line_plot(actual_returns[-n_points:],
               predicted_returns[-n_points:],
               actual_returns[-n_points:][:-1],
               predicted_returns[-n_points:].shift(-1),
               'actual returns', 'predicted returns', lw=3)

plot_data.line_plot(actual_returns[-n_points:][:-1], predicted_returns[-n_points:].shift(-1),
          'actual returns', 'predicted returns', lw=3)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

# actual correlation
corr = np.corrcoef(actual_returns, predicted_returns)[0][1]
ax1.scatter(actual_returns, predicted_returns, color='k', marker='o', alpha=0.5, s=100)
ax1.set_title('r = {:.2f}'.format(corr), fontsize=18)

# shifted correlation
shifted_actual = actual_returns[:-1]
shifted_predicted = predicted_returns.shift(-1).dropna()
corr = np.corrcoef(shifted_actual, shifted_predicted)[0][1]
ax2.scatter(shifted_actual, shifted_predicted, color='k', marker='o', alpha=0.5, s=100)
ax2.set_title('r = {:.2f}'.format(corr), fontsize=18);
plt.show()
