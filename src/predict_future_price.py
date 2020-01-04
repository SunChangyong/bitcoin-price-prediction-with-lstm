import json
import requests
import pandas as pd
from keras.models import load_model

import read_data as etl

# endpoint = 'https://min-api.cryptocompare.com/data/histoday'
# res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=1')
# predict_data = pd.DataFrame(json.loads(res.content)['Data'])
# predict_data = predict_data.set_index('time')
# predict_data.index = pd.to_datetime(predict_data.index, unit='s')
# print(predict_data)
# predict_data = predict_data.tail(1)
predict_data = pd.read_csv('./data/dataset.csv')
predict_data = predict_data.tail(4)
print(predict_data)

target_col = 'close'
# data params
window_len = 7
test_size = 1
zero_base = True

# X_predict = etl.extract_window_data(predict_data, window_len, zero_base)

predict_data = etl.normalise_zero_base(predict_data)
print(predict_data)
print(type(predict_data))
X_predict = np.array()
print(X_predict)
print(type(X_predict))
y_predict = predict_data[target_col][window_len:].values
# zero base
y_test = y_predict / predict_data[target_col][:-window_len].values - 1

model = load_model('./model/model.h5')

targets = predict_data[target_col][window_len:]
preds = model.predict(X_predict).squeeze()

preds = predict_data[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
print (preds)
