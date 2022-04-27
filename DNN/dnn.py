import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, Sequential
from collections import deque
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import talib


class DNNModel(object):
    def __init__(self, id, od):
        self.inputDim = id
        self.outputDim = od
        self.replay_size = 8000
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model()
        self.samples = None
        self.init = True
    def appendSamples(self, data, label):
        if self.init:
            self.samples = [data, label]
            self.init = False
        else:
            self.samples[0] = np.concatenate((self.samples[0], data), axis=0)
            self.samples[1] = np.concatenate((self.samples[1], label), axis=0)

    def create_model(self):
        model = Sequential([
            layers.Dense(100,input_dim=self.inputDim, activation=tf.nn.relu),
            layers.Dense(100, activation=tf.nn.relu),
            layers.Dense(100, activation=tf.nn.relu),
            layers.Dense(100, activation=tf.nn.relu),
            layers.Dense(100, activation=tf.nn.relu),
            layers.Dense(self.outputDim)
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.0001),
                      metrics=['accuracy']
                      )
        return model
    def save_model(self, file_path = 'saved_model'):
        print('model saved')
        #tf.saved_model.save(self.model, file_path)
        self.model.save_weights('my_model_weights.h5')
        self.model.save(file_path, overwrite=True, include_optimizer=True)
    def trainModel(self):
        self.model = keras.models.load_model('saved_model', compile=False)
        self.model.load_weights('my_model_weights.h5', by_name=True)
        self.model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.0001))
        print('training model: ', self.model.layers)
        #plot_model(self.model,to_file='model.png',show_shapes=True)
        #self.model.summary()
        #fig, ax = plt.subplots()
        #ax.plot(self.samples[0], self.samples[1])
        #sma = talib.SMA(self.samples[0][:, 6], timeperiod=100)
        #ax.plot(sma, label='AMF-1')
        #sma = talib.SMA(self.samples[0][:, 7], timeperiod=100)
        #ax.plot(sma, label='AMF-2')
        #sma = talib.SMA(self.samples[0][:, 8], timeperiod=100)
        #ax.plot(sma, label='AMF-3')
        #ax.plot(self.samples[0][:,6])
        #ax.plot(self.samples[0][:,7])
        #ax.plot(self.samples[0][:,8])
        #plt.show()
        data = tf.convert_to_tensor(self.samples[0])
        label = tf.convert_to_tensor(self.samples[1])
        print(data.shape, label.shape)

        history = self.model.fit(data, label, epochs=10, steps_per_epoch=100)
        self.save_model()

        #pred = self.model.predict(self.samples[0])
        #sma = talib.SMA(pred[:, 0].astype(float), timeperiod=100)
        #ax.plot(sma, label='AMF-1-predicted')
        #sma = talib.SMA(pred[:, 1].astype(float), timeperiod=100)
        #ax.plot(sma, label='AMF-2-predicted')
        #sma = talib.SMA(pred[:, 2].astype(float), timeperiod=100)
        #ax.plot(sma, label='AMF-3-predicted')
        #ax.plot(pred[:, 0])
        #ax.plot(pred[:, 1])
        #ax.plot(pred[:, 2])
        #plt.legend()
        #plt.show()

        #loss = history.history['loss']
        #plt.plot(loss)
        #plt.show()
        return self.samples[0]
    def updateModel(self, file_path):
        self.model = keras.models.load_model(file_path, compile=False)
        self.model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.001), metrics=['accuracy'])
        data = tf.convert_to_tensor(self.samples[0])
        label = tf.convert_to_tensor(self.samples[1])
        history = self.model.fit(data, label, epochs=100, steps_per_epoch=10)
        self.model.save("saved_model_v2")

    #fig, ax = plt.subplots()
    #color = ['red', 'blue', 'black', 'green', 'pink', 'orange', 'violet', 'lawngreen', 'dodgerblue', 'magenta']
    #for i in range(len(amfList)):
    #    sma = talib.SMA(np.array(amfList[i].n_msgs_record).astype(float), timeperiod=10)
    #    ax.plot(np.array(amfList[i].time_point), sma)
    #plt.savefig(str(num)+'.jpg')
    #plt.show()
