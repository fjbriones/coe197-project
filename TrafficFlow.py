from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.dates import DateFormatter
import pandas
import keras
import numpy as np
import matplotlib.pyplot as plt
import mock
import datetime
import matplotlib.dates as mdates
import tensorflow as tf

#Function for reading data
def read_dataset(filename, columns, scaler):
    dataframe = pandas.read_csv(filename, usecols=columns, engine='python')
    dataset = dataframe.values
    #dataset = scaler.fit_transform(dataset)
    return dataset

#Function for formatting the data into proper arrays
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)

#Function for saving the model
def save_model(filename, model):
    fn_json = filename + '.json'
    fn_h5 = filename + '.h5'
    model_json = model.to_json()
    with open(fn_json, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(fn_h5)
    # print('Model is saved')

#hard tan function
def hard_tanh (x):
    return tf.minimum(tf.maximum(x, -1.), 1.)

#Function for loading the model
old_get = keras.activations.get
def patch_get(x):
    return hard_tanh if x == 'hard_tanh' else old_get(x)
@mock.patch('keras.activations.get', patch_get)
def load_model(filename):
    fn_json = filename + '.json'
    fn_h5 = filename + '.h5'
    json_file = open(fn_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={'hard_tanh': hard_tanh})
    loaded_model.load_weights(fn_h5)
    print('Model is loaded')
    return loaded_model

#Function for adding noise to input
def noisy_input(x, c=0.05):
    noise = np.random.normal(loc=0.0, scale=1.0, size=np.shape(x))
    output = x + c*noise
    return output

#Function for training LSTM model
def model_lstm_train(units, trainX, trainY, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(LSTM(units, input_shape=(time_step, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/LSTM_model', model)

    return model

#Function for training GRU model
def model_gru_train(units, trainX, trainY, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(GRU(units, input_shape=(time_step, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/GRU_model',model)

    return model

#Function for training NAN LSTM model
def model_nan_lstm_train(units, trainX, trainY, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(LSTM(units, input_shape=(time_step, 1), activation=hard_tanh))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/NAN_LSTM_model', model)

    return model

#Function for training NAN GRU model
def model_nan_gru_train(units, trainX, trainY, epochs, batch_size, validation_split):
    model = Sequential()
    model.add(GRU(units, input_shape=(time_step, 1), activation=hard_tanh))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/NAN_GRU_model',model)

    return model

#Function for training NANI LSTM model
def model_nani_lstm_train(units, trainX, trainY, epochs, batch_size, validation_split):
    trainX = noisy_input(trainX)
    model = Sequential()
    model.add(LSTM(units, input_shape=(time_step, 1), activation=hard_tanh))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/NANI_LSTM_model', model)

    return model

#Function for training NANI GRU model
def model_nani_gru_train(units, trainX, trainY, epochs, batch_size, validation_split):
    trainX=noisy_input(trainX)
    model = Sequential()
    model.add(GRU(units, input_shape=(time_step, 1), activation=hard_tanh))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=validation_split)

    save_model('models/NANI_GRU_model',model)

    return model

#Function for testing data
def flow_predict(x_data, x_true, model, model_type):
    dataPredict = model.predict(x_data)

    #dataPredict = scaler.inverse_transform(dataPredict)
    #x_true = scaler.inverse_transform([x_true])

    scoreMSE = mean_squared_error(x_true, dataPredict[:,0])
    scoreMAE = mean_absolute_error(x_true, dataPredict[:,0])

    print(model_type + 'Test score: %.2lf MSE %.2lf MAE' %(scoreMSE, scoreMAE))

    return dataPredict, x_true, scoreMSE, scoreMAE

#Function for plotting data
def flow_plot(x_pred, x_true, start, len):
    truePlot = np.transpose(x_true)
    truePlot = truePlot[start:len]
    predPlot = x_pred[start:len]

    plt.plot(truePlot, label='True Plot')
    plt.plot(predPlot, label='Predicted Plot')
    plt.legend()
    plt.show()

#Set so results don't change every time it is run
np.random.seed(7)



#Scaler for data
scaler = MinMaxScaler(feature_range=(-1, 1))

#Read dataset from file. Dataset already seperated into two files for ease
train = read_dataset('data/train_dataset.csv', [1], scaler)
test = read_dataset('data/test_dataset.csv', [1], scaler)

#General variables
time_step = 6
len_day = 288 #len_day in terms of 5 minutes
# epochs = 500
batch_size = 32
units = 32
validation_split = 0.2

#Create datasets
trainX, trainY = create_dataset(train, time_step)
testX, testY = create_dataset(test, time_step)
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

#For optimal training, search for the right number of iterations.
start = 20
end = 320
step = 20

n = (int)((end - start)/step)

data_train_mse = np.zeros((6, n))
data_test_mse = np.zeros((6, n))
data_train_mae = np.zeros((6, n))
data_test_mae = np.zeros((6, n))

#Uncomment the code below to train data
# for epochs in range(start, end, step):
#
#     i = (int)((epochs - start)/step)
#
#     print('Number of epochs: %d' %(epochs))
#
#     modelLstm = model_lstm_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
#     modelGru = model_gru_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
#
#     modelNanLstm= model_nan_lstm_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
#     modelNanGru = model_nan_gru_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
#
#     modelNaniLstm= model_nani_lstm_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
#     modelNaniGru = model_nani_gru_train(units, trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
#
#     predictedTrainLstm, trainLstm, data_train_mse[0,i], data_train_mae[0,i] = flow_predict(trainX, trainY, modelLstm, 'LSTM Training ')
#     predictedTestLstm, testLstm, data_test_mse[0,i], data_test_mae[0,i] = flow_predict(testX, testY, modelLstm, 'LSTM Testing ')
#     predictedTrainGru, trainGru, data_train_mse[1,i], data_train_mae[1,i] = flow_predict(trainX, trainY, modelGru, 'GRU Training ')
#     predictedTestGru, testGru, data_test_mse[1,i], data_test_mae[1,i] = flow_predict(testX, testY, modelGru, 'GRU Testing ')
#
#     predictedTrainNanLstm, trainNanLstm, data_train_mse[2,i], data_train_mae[2,i] = flow_predict(trainX, trainY, modelNanLstm, 'NAN LSTM Training ')
#     predictedTestNanLstm, testNanLstm, data_test_mse[2,i], data_test_mae[2,i] = flow_predict(testX, testY, modelNanLstm, 'NAN LSTM Testing ')
#     predictedTrainNanGru, trainNanGru, data_train_mse[3,i], data_train_mae[3,i] = flow_predict(trainX, trainY, modelNanGru, 'NAN GRU Training ')
#     predictedTestNanGru, testNanGru, data_test_mse[3,i], data_test_mae[3,i] = flow_predict(testX, testY, modelNanGru, 'NAN GRU Testing ')
#
#     predictedTrainNaniLstm, trainNaniLstm, data_train_mse[4,i], data_train_mae[4,i] = flow_predict(trainX, trainY, modelNaniLstm, 'NANI LSTM Training ')
#     predictedTestNaniLstm, testNaniLstm, data_test_mse[4,i], data_test_mae[4,i] = flow_predict(testX, testY, modelNaniLstm, 'NANI LSTM Testing ')
#     predictedTrainNaniGru, trainNaniGru, data_train_mse[5,i], data_train_mae[5,i] = flow_predict(trainX, trainY, modelNaniGru, 'NANI GRU Training ')
#     predictedTestNaniGru, testNaniGru, data_test_mse[5,i], data_test_mae[5,i] = flow_predict(testX, testY, modelNaniGru, 'NANI GRU Testing ')
# else:
#     print('Done')
#     df_mse = pandas.DataFrame(data_test_mse)
#     df_mae = pandas.DataFrame(data_test_mae)
#     df_mse.to_csv('models/mse.csv')
#     df_mae.to_csv('models/mae.csv')

#Load saved data. Comment out if training
modelLstm = load_model('models/LSTM_model')
modelGru = load_model('models/GRU_model')
modelNanLstm = load_model('models/NAN_LSTM_model')
modelNanGru = load_model('models/NAN_GRU_model')
modelNaniLstm = load_model('models/NANI_LSTM_model')
modelNaniGru = load_model('models/NANI_GRU_model')

i=1
predictedTestLstm, testLstm, data_test_mse[0,i], data_test_mae[0,i] = flow_predict(testX, testY, modelLstm, 'LSTM Testing ')
predictedTestGru, testGru, data_test_mse[1,i], data_test_mae[1,i] = flow_predict(testX, testY, modelGru, 'GRU Testing ')
predictedTestNanLstm, testNanLstm, data_test_mse[2,i], data_test_mae[2,i] = flow_predict(testX, testY, modelNanLstm, 'NAN LSTM Testing ')
predictedTestNanGru, testNanGru, data_test_mse[3,i], data_test_mae[3,i] = flow_predict(testX, testY, modelNanGru, 'NAN GRU Testing ')
predictedTestNaniLstm, testNaniLstm, data_test_mse[4,i], data_test_mae[4,i] = flow_predict(testX, testY, modelNaniLstm, 'NANI LSTM Testing ')
predictedTestNaniGru, testNaniGru, data_test_mse[5,i], data_test_mae[5,i] = flow_predict(testX, testY, modelNaniGru, 'NANI GRU Testing ')

#For graphing purposes
start = (int)(len_day - 6)
end = (int)(2*len_day)
timeAxis = [datetime.datetime(year=2016, month=3, day=2, hour=0, minute=0) + datetime.timedelta(minutes=5*i) for i in range(0,end)]
true = np.transpose(testLstm)
true = true[start:start+end]
pred1 = predictedTestLstm[start:start+end]
pred2 = predictedTestGru[start:start+end]
pred3 = predictedTestNanLstm[start:start+end]
pred4 = predictedTestNanGru[start:start+end]
pred5 = predictedTestNaniLstm[start:start+end]
pred6 = predictedTestNaniGru[start:start+end]

fig, ax = plt.subplots(1)
ax.plot(timeAxis, true, label='True Plot')
# ax.plot(timeAxis, pred1, label='LSTM')
# ax.plot(timeAxis, pred2, label='GRU')
# ax.plot(timeAxis, pred3, label='NAN LSTM')
# ax.plot(timeAxis, pred4, label='NAN GRU')
# ax.plot(timeAxis, pred5, label='NANI LSTM')
ax.plot(timeAxis, pred6, label='NANI GRU')
fig.autofmt_xdate()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('Time')
plt.ylabel('Veh/5min')
plt.legend()
plt.show()