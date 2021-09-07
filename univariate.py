# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Importing Lib for Ploting
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from datetime import datetime 

# Import Data Set
dataset = pd.read_csv('household_power_consumption.txt', 
                      sep=';', 
                      parse_dates={'DateTime' : ['Date', 'Time']},
                      infer_datetime_format=True, 
                      low_memory=False, 
                      na_values=['nan','?'])


#Checking Missing values
dataset.isnull().sum()

# fill missing values with a value at the same time one week ago
dataset.index = pd.to_datetime(dataset['DateTime']) 
cols = ["Global_active_power", "Global_reactive_power", "Voltage","Global_intensity",
        "Sub_metering_1","Sub_metering_2","Sub_metering_3"]
dataset[cols] = dataset[cols].fillna(dataset[cols].shift(7*24*60))
dataset = dataset.reset_index(drop=True)
print (dataset.head())
dataset.isnull().sum()

# fill missing values with a value at the same time one day ago for remaining filling values
dataset.index = pd.to_datetime(dataset['DateTime']) 
cols = ["Global_active_power", "Global_reactive_power", "Voltage","Global_intensity",
        "Sub_metering_1","Sub_metering_2","Sub_metering_3"]
dataset[cols] = dataset[cols].fillna(dataset[cols].shift(1440))
dataset = dataset.reset_index(drop=True)
print (dataset.head())

#searching null 
dataset.isnull().sum()

# Sorting data
data = [dataset["DateTime"], 
        dataset["Global_active_power"],
        dataset["Sub_metering_1"],
        dataset["Sub_metering_2"],
        dataset["Sub_metering_3"]]
header= ['DateTime',
         'Units','App-1', 'App-2','App-3']
#converting strings into internal datetime 
dataset = pd.concat(data, axis = 1, keys = header)

"""---------------------------------------Joining  Datasets------------------------------------------"""

#Resampling the data Hourly
dataset = dataset.resample('H', on ='DateTime').mean()
dataset = dataset.reset_index()


#Adding weather data
weather = pd.read_csv("weathernational_data.csv",infer_datetime_format=True, 
                     low_memory=False,parse_dates={'datetime':[0]},index_col=['datetime'])

#Resampling weather in Hour on date time and preparing for concatenation 
weather = weather.resample('H').mean()
weather = weather.reset_index()


#Requried Data
starting_date = datetime.strptime("2007-06-06 14:00:00", '%Y-%m-%d %H:%M:%S')
ending_date =  datetime.strptime("2010-11-26 21:00:00", '%Y-%m-%d %H:%M:%S')

dataset = dataset[dataset['DateTime'] >= starting_date]
dataset = dataset.reset_index(drop=True)

weather = weather[weather['datetime'] <= ending_date] 
weather = weather.reset_index(drop=True)


#Concatenating the weather and dataset
total_df = [dataset, weather]
total_df = pd.concat(total_df, axis=1)
total_df = total_df.drop(['datetime'],axis = 1)


#Adding weather data
holiday = pd.read_csv("holiday_france.csv",
                     low_memory=False)
holiday = holiday.to_numpy()
holiday = np.append(holiday, np.ones([holiday.shape[0], 1], dtype=np.int32), axis=1)
holiday = pd.DataFrame(holiday)
holiday['dat'] = pd.to_datetime(holiday[0])
holiday = holiday.drop([0],axis = 1)

#Concating dataset with holiday(1 for Holiday)
holiday.index = pd.to_datetime(holiday.index)
total_df['Holiday'] = (total_df['DateTime'].dt.date.map(
                        holiday.set_index('dat')[1].to_dict()).fillna(0))

#convert these strings into internal datetimes
#Break apart the date and get the year, month, week of year, day of month, hour
total_df['Year']  = total_df['DateTime'].dt.year
total_df['Month'] = total_df['DateTime'].dt.month
total_df['Hour']  = total_df['DateTime'].dt.hour
total_df['DayofWeek'] = total_df['DateTime'].dt.dayofweek
#Setting Indexing
total_df = total_df.set_index('DateTime')


#Saving Dataset Before feature engineering
#total_df.to_csv("paris_dataset_before_feature_engineering.csv")

"""_____________________________________Feature Engineering____________________________"""

#Cyclic Variables (Month, Hour, DayofWeek)
#Month
total_df['mnth_sin'] = np.sin((total_df.Month-1)*(2.*np.pi/12))
total_df['mnth_cos'] = np.cos((total_df.Month-1)*(2.*np.pi/12))
#Hour
total_df['hr_sin']   = np.sin(total_df.Hour*(2.*np.pi/24))
total_df['hr_cos']   = np.cos(total_df.Hour*(2.*np.pi/24))
#DayofWeek
total_df['dy_sin']   = np.sin(total_df.DayofWeek*(2.*np.pi/7))
total_df['dy_cos']   = np.cos(total_df.DayofWeek*(2.*np.pi/7))

# Sorting data and droping unwanted columns
important_columns = ['outdoor_humidity', 'outdoor_temperature', 'wind_speed', 'Year',
                     "mnth_sin", 'mnth_cos', "hr_sin", 'hr_cos', "dy_sin", 'dy_cos',
                     "App-1","App-2","App-3","Units"]
total_df = total_df[important_columns]


#Categorey Variable (year)  #Day(Date) Garbage 
#-----Month Date Needs to be used as Categorical variable But its not f any signigicance---- 
from sklearn.preprocessing import LabelEncoder
Label_encoder = LabelEncoder()
total_df['Year'] = Label_encoder.fit_transform(total_df['Year'])

#Feature Scaling of independent Variables
from sklearn.preprocessing import MinMaxScaler
iv_columns = ['outdoor_humidity', 'outdoor_temperature', 'wind_speed', 'Year',
                     "mnth_sin", 'mnth_cos', "hr_sin", 'hr_cos', "dy_sin", 'dy_cos',]
iv_transformer = MinMaxScaler(feature_range= (0,1))
iv_transformer = iv_transformer.fit(total_df[iv_columns].to_numpy())
total_df[iv_columns] = iv_transformer.transform(total_df[iv_columns].to_numpy())

#Feature Scaling of Dependent Variables
dv_columns = ['Units', 'App-1', 'App-2', 'App-3']
dv_transformer = MinMaxScaler(feature_range= (0,1))
dv_transformer = dv_transformer.fit(total_df[dv_columns])
total_df[dv_columns] = dv_transformer.transform(total_df[dv_columns])

#Saving dataset after featuring engineering
#total_df.to_csv("paris_dataset_after_feature_engineering.csv")

# from pickle import dump
# dump(Label_encoder , open('labelencoder.pkl'  , 'wb'))
# dump(iv_transformer, open('iv_transformer.pkl', 'wb'))
# dump(dv_transformer, open('dv_transformer.pkl', 'wb'))



"""_____________________________________Test/Train Dataset split ____________________________"""

#Function to Split dataset into train and test set
def split_dataset(dataset, split_factor):
    train_size = int(len(dataset) * split_factor)
    test_size = len(dataset) - train_size
    train, test = dataset.iloc[0:train_size], dataset.iloc[train_size:len(dataset)]
    print(len(train), len(test))
    return train , test

#Split Dataset
dataset = total_df
split_factor = 0.95
train , test = split_dataset(dataset ,split_factor)

"""_____________________________________Look back Function for Single Output______________________"""

# #Look Back function
# def create_dataset(X, y, time_steps=1):
#     Xs, ys = [], []
#     for i in range(len(X) - time_steps):
#         v = X.iloc[i:(i + time_steps)].values
#         Xs.append(v)
#         ys.append(y.iloc[i + time_steps])
#     return np.array(Xs), np.array(ys)

# time_steps = 72
# # reshape to [samples, time_steps, n_features]
# X_train, y_train = create_dataset(train, train.Global_active_power, time_steps )
# X_test , y_test  = create_dataset(test , test.Global_active_power , time_steps )
# print(X_train.shape, y_train.shape)





"""_____________________________________Look back Function for Multiple Output____________________"""
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


time_steps = 24
# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(train, train.loc[:,dv_columns], time_steps )
X_test,   y_test = create_dataset(test , test.loc[:, dv_columns] , time_steps )
print(X_train.shape, y_train.shape)


"""___________________________________LSTM Framework____________________________________"""

import tensorflow as tf
from tensorflow import keras

#Importing the keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout



#Defining Parameters
input_shape = (X_train.shape[1], X_train.shape[2])
dropout = 0.2
dense_untis = 4                                      #Output layer



#Initializing the RNN
model_lstm = Sequential()

#Adding the LSTM layer and some Dropout Regularization
model_lstm.add(LSTM(units = 50, return_sequences= True , input_shape = input_shape))
model_lstm.add(Dropout(dropout))
#Adding the second layer and some dropout regularization
model_lstm.add(LSTM(units = 50, return_sequences= True))
model_lstm.add(Dropout(dropout))
#Adding a 3rd layer and some dropout regulariztion
model_lstm.add(LSTM(units = 50, return_sequences= True))
model_lstm.add(Dropout(dropout))
#Adding a 4th(last) layers and some dropout regularization
model_lstm.add(LSTM(units = 50, return_sequences = False))
model_lstm.add(Dropout(dropout))
#Adding the output layer
model_lstm.add(Dense(units = dense_untis, activation='linear'))



#Function for Compiling the RNN


def compile_model(model, optimizer, loss):
    model.compile(optimizer = optimizer, 
                  loss = loss 
                  )    



#Function for Fitting the RNN to the training set
def fit_model(model, batch_size, epochs, validation_split):
    history = model.fit(X_train , y_train ,
                        batch_size = batch_size ,
                        epochs = epochs, 
                        validation_split = validation_split ,
                        shuffle = False
                        )
    return history




"""________________________________Model Analysis________________________________"""
#Compiling the RNN
model = model_lstm
optimizer = 'adam'
loss = 'mse'
compile_model(model, optimizer, loss)


#Fitting the RNN to the training set
#Defining Parameters
batch_size = 72
epochs = 50
validation_split=0.1
history = fit_model(model, batch_size, epochs, validation_split)

#Visualizing loss and val_loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

#Prediction Plotting
y_pred = model_lstm.predict(X_test)



"""________________________________polting_______________________________________"""