from contextlib import suppress
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tkinter as tk

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error
import math
from pmdarima import auto_arima
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings("ignore")
from keras import callbacks

class MyProgressBarCallback(Callback):
  def setProgressBar(self,queue,nEpochs):
      self.queue = queue
      self.jump = 200 / nEpochs
  def on_train_batch_end(self,batch, logs=None):
      print(self.jump)
      print(batch)
      


class StockPrediction:

    def __init__(self,tickerSymbol, prediction_days, start_date,end_date,data_source='yahoo'):
        self.tickerSymbol = tickerSymbol
        self.data = web.DataReader(tickerSymbol, data_source , start_date, end_date)
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.prediction_days = prediction_days
        self.data_source=data_source
    
    def PrepareData(self, column = 'Close'):
        
        x_train = []    
        y_train = []
        self.column=column
        scaled_data = self.scaler.fit_transform(self.data[column].values.reshape(-1,1))

        for x in range(self.prediction_days,len(scaled_data)):
            x_train.append(scaled_data[x-self.prediction_days:x,0])
            y_train.append(scaled_data[x,0])

        x_train, self.y_train = np.array(x_train), np.array(y_train)
        self.x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    def BuildModel(self, _optimizer='adam', _loss='mean_squared_error',_epochs=25, _batch_size=32, modelInputs = {'units':[50,50,50,1],'return_sequences':[True,True,False],'dropout':[0.2,0.2,0.2]},progressBar=None):
        self.model = Sequential()

        self.model.add(LSTM(units = modelInputs['units'][0],return_sequences = modelInputs['return_sequences'][0], input_shape = (self.x_train.shape[1],1)))
        self.model.add(Dropout(modelInputs['dropout'][0]))

        for i in range(1,len(modelInputs['dropout'])):
            self.model.add(LSTM(units = modelInputs['units'][i], return_sequences = modelInputs['return_sequences'][i]))
            self.model.add(Dropout(modelInputs['dropout'][i]))
        
        self.model.add(Dense(units=modelInputs['units'][-1]))

        self.model.compile(optimizer=_optimizer, loss=_loss)

        self.model.fit(self.x_train, self.y_train, epochs=_epochs, batch_size=_batch_size)
        self.model.summary()
    
    def TestAccurancyAndPlot(self, test_start_date,test_end_date):
        #Load data
        test_data = web.DataReader(self.tickerSymbol, self.data_source, test_start_date,test_end_date)
        actual_prices = test_data[self.column].values

        total_dataset = pd.concat((self.data[self.column],test_data[self.column]), axis=0)
        model_input= total_dataset[len(total_dataset) - len(test_data) - self.prediction_days:].values

        model_input = model_input.reshape(-1,1)
        model_input = self.scaler.transform(model_input)

        #Make predictions on Test Data

        x_test = []

        for x in range(self.prediction_days, len(model_input)+1):
            x_test.append(model_input[x-self.prediction_days:x,0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

        predicted_prices = self.model.predict(x_test)
        predicted_prices = self.scaler.inverse_transform(predicted_prices)
        print(predicted_prices[-1])
        y_train = self.scaler.inverse_transform(self.y_train.reshape(-1,1))

        y_train_test = self.model.predict(self.x_train)
        y_train_test = self.scaler.inverse_transform(y_train_test)

        naiveMethod = total_dataset[len(self.data)-1:len(total_dataset)-1]

        # self.stepwise_fit = auto_arima(self.data["Close"],trace=True,suppress_warnings =True)
        # self.stepwise_fit.summary()
            
        # self.arima_model = ARIMA(self.data[self.column],order =(0,1,0))
        # self.arima_model = self.stepwise_fit.fit(y=self.data["Close"])
        # self.arima_model = self.arima_model(n_periods=5)
        # self.arima_model.summary()
        # arima_predicted_prices =self.stepwise_fit.predict(n_periods=len(actual_prices),typ="levels")
        # print(arima_predicted_prices)
        # AR_model = AutoReg(self.x_train.tolist(), lags=1)

        # AR_model.fit()
        # y=AR_model.predic(start = len(self.x_train), end =len(actual_prices))
        

        return {"arimaMethod":[],"naiveMethod":naiveMethod,"trained_prices":y_train,"predicted_trained_prices":y_train_test,"actual_prices":actual_prices,"predicted_prices":predicted_prices}


# test = StockPrediction('AAPL',60,dt.datetime(2012,1,1),dt.datetime(2020,1,1))
# test.PrepareData()
# test.BuildModel()
# test.TestAccurancyAndPlot(dt.datetime(2020,1,1))