import numpy as np
# https://www.tensorflow.org/guide/keras/rnn
from keras.layers import LSTM, Dense, Conv1D
from keras.models import Sequential
from Evaluation import evaluation


def Model_RNN(train_data, train_target, test_data, test_target):
    pred = np.zeros(test_target.shape)
    for i in range(train_target.shape[1]):
        out, model = RNN_train(train_data, train_target[:, i], test_data)  # RNN
        out = np.round(out)
        pred[:] = out

    Eval = evaluation(pred, test_target)
    return Eval, pred


def RNN_train(trainX, trainY, testX):
    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, dilation_rate=1, activation='relu'))  # Adding Dilated Layer
    model.add(Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu'))
    model.add(LSTM(10, input_shape=trainX.shape)) # (1, trainX.shape[0])
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)
    testPredict = model.predict(testX)
    return testPredict, model
