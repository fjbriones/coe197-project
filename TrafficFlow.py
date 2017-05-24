from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def read_dataset(filename, columns, scaler):
    dataframe = pandas.read_csv(filename, usecols=columns, engine='python')
    dataset = dataframe.values
    dataset = scaler.fit_transform(dataset)
    return dataset

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)

def save_model(filename, model):
    fn_json = filename + '.json'
    fn_h5 = filename + '.h5'
    model_json = model.to_json()
    with open(fn_json, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(fn_h5)
    print('Model is saved')

def load_model(filename):
    fn_json = filename + '.json'
    fn_h5 = filename + '.h5'
    json_file = open(fn_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(fn_h5)
    print('Model is loaded')
    return loaded_model

def hard_tanh (x):
    return tf.minimum(tf.maximum(x, -1.), 1.)

def noisy_input(x, c=0.05):
    noise = np.random.normal(loc=0.0, scale=1.0, size=np.shape(x))
    output = x + c*noise
    return output

def model_lstm_train(units, trainX, trainY, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(LSTM(units, input_shape=(time_step, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/LSTM_model', model)

    return model

def model_gru_train(units, trainX, trainY, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(GRU(units, input_shape=(time_step, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/GRU_model',model)

    return model

def model_nan_lstm_train(units, trainX, trainY, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(LSTM(units, input_shape=(time_step, 1), activation=hard_tanh))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/NAN_LSTM_model', model)

    return model

def model_nan_gru_train(units, trainX, trainY, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(GRU(units, input_shape=(time_step, 1), activation=hard_tanh))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/NAN_GRU_model',model)

    return model

def model_nani_lstm_train(units, trainX, trainY, epochs, batch_size, validation_split):
    trainX = noisy_input(trainX)
    model = Sequential()
    model.add(LSTM(units, input_shape=(time_step, 1), activation=hard_tanh))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/NANI_LSTM_model', model)

    return model

def model_nani_gru_train(units, trainX, trainY, epochs, batch_size, validation_split):
    trainX=noisy_input(trainX)
    model = Sequential()
    model.add(GRU(units, input_shape=(time_step, 1), activation=hard_tanh))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/NANI_GRU_model',model)

    return model

def flow_predict(x_data, x_true, model, model_type):
    dataPredict = model.predict(x_data)

    dataPredict = scaler.inverse_transform(dataPredict)
    x_true = scaler.inverse_transform([x_true])

    scoreMSE = mean_squared_error(x_true[0], dataPredict[:,0])
    scoreMAE = mean_absolute_error(x_true[0], dataPredict[:,0])

    print(model_type + 'Test score: %.2lf MSE %.2lf MAE' %(scoreMSE, scoreMAE))

    return dataPredict, x_true

def flow_plot(x_pred, x_true, len):
    truePlot = np.transpose(x_true)
    truePlot = truePlot[0:len]
    predPlot = x_pred[0:len]

    plt.plot(truePlot, label='True Plot')
    plt.plot(predPlot, label='Predicted Plot')
    plt.legend()
    plt.show()

np.random.seed(7)

scaler = MinMaxScaler(feature_range=(-1, 1))

train = read_dataset('data/train_dataset.csv', [1], scaler)
test = read_dataset('data/test_dataset.csv', [1], scaler)

time_step = 6
len_day = 288 #len_day in terms of 5 minutes
epochs = 50
batch_size = 32
units = 32
validation_split = 0.2

#Create datasets
trainX, trainY = create_dataset(train, time_step)
testX, testY = create_dataset(test, time_step)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

modelLstm = model_lstm_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
modelGru = model_gru_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

modelNanLstm= model_nan_lstm_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
modelNanGru = model_nan_gru_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

modelNaniLstm= model_nani_lstm_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
modelNaniGru = model_nani_gru_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

predictedTrainLstm, trainLstm = flow_predict(trainX, trainY, modelLstm, 'LSTM Training ')
predictedTestLstm, testLstm = flow_predict(testX, testY, modelLstm, 'LSTM Testing ')
predictedTrainGru, trainGru = flow_predict(trainX, trainY, modelGru, 'GRU Training ')
predictedTestGru, testGru = flow_predict(testX, testY, modelGru, 'GRU Testing ')

predictedTrainNanLstm, trainNanLstm = flow_predict(trainX, trainY, modelNanLstm, 'NAN LSTM Training ')
predictedTestNanLstm, testNanLstm= flow_predict(testX, testY, modelNanLstm, 'NAN LSTM Testing ')
predictedTrainNanGru, trainNanGru = flow_predict(trainX, trainY, modelNanGru, 'NAN GRU Training ')
predictedTestNanGru, testNanGru = flow_predict(testX, testY, modelNanGru, 'NAN GRU Testing ')

predictedTrainNaniLstm, trainNaniLstm = flow_predict(trainX, trainY, modelNaniLstm, 'NANI LSTM Training ')
predictedTestNaniLstm, testNaniLstm= flow_predict(testX, testY, modelNaniLstm, 'NANI LSTM Testing ')
predictedTrainNaniGru, trainNaniGru = flow_predict(trainX, trainY, modelNaniGru, 'NANI GRU Training ')
predictedTestNaniGru, testNaniGru = flow_predict(testX, testY, modelNaniGru, 'NANI GRU Testing ')