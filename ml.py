""" Y_pred , Y_test, X_train , X_test , Feature_importance , Plots ,Conclusions """ 
# Multi step prediction in LSTM 
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from pickle import load
import datetime
import streamlit as st
import os 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
#-------------------------Importing the Model and the weights--------------------------------------

def import_model():
    json_file = open('models/paris_model.json', 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/paris_weights.h5")
    return loaded_model
dv_transformer= load(open('models/dv_transformer.pkl', 'rb'))
iv_transformer= load(open('models/iv_transformer.pkl', 'rb'))
    # return dv_transformer, iv_transformer

def engineer_dataset():
    global iv_columns
    global dv_columns
    global dates
    iv_columns = ['outdoor_humidity', 'outdoor_temperature', 'wind_speed']
    dv_columns = ['Units','App-1','App-2','App-3']
    test = pd.read_csv("data/test_dataset.csv",index_col = 0)
    test['DateTime'] =  pd.to_datetime(test['DateTime'], infer_datetime_format=True)
    test.loc[:, dv_columns] = dv_transformer.transform(test[dv_columns])
    test.loc[:, iv_columns] = iv_transformer.transform(test[iv_columns].to_numpy())
    test = test.reset_index(drop = True)
    return test 

def required_dataframe():
    important_columns = ["DateTime","Units","App-1","App-2","App-3"]
    test = engineer_dataset()
    actual_data = test[important_columns]
    actual_data.columns = ["DateTime","Units_real","App-1_real","App-2_real","App-3_real"]
    return actual_data


def dates():
    test = engineer_dataset()
    date = test["DateTime"]
    date = pd.DataFrame(date)
    return date

#Feature Scaling from original scaled functions
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
def create_tensor():
    time_steps = 24*7
    test = engineer_dataset()
    test.set_index("DateTime",inplace = True)
    X_test,  y_test = create_dataset(test , test.loc[:, dv_columns] , time_steps)
    return X_test , y_test

def prediction():
    X_test , y_test = create_tensor()
    model = import_model()
    y_pred = model.predict(X_test)
    return y_pred

def inverse_transform():
    date = dates()
    y_pred = prediction()
    y_pred_inv = dv_transformer.inverse_transform(y_pred)
    y_pred_inv = pd.DataFrame(y_pred_inv)
    y_pred_inv.columns = ["Units_pred","App-1_pred","App-2_pred","App-3_pred"]
    Pred = pd.concat((date,y_pred_inv),axis = 1)
    return Pred

def search_dates(filtered_date):
    global filtered_dates 
    start_date = filtered_date
    prediction = inverse_transform()
    start_date =datetime.combine(start_date , datetime.min.time())
    next_date = start_date + timedelta(days=1)
    filtered_dates = prediction[(prediction['DateTime'] >= start_date) & (prediction['DateTime']< next_date)]
    filtered_dates = filtered_dates.reset_index(drop = True)
    return filtered_dates


def actual_data(filtered_date):
    start_date = filtered_date
    actual = required_dataframe()
    start_date =datetime.combine(start_date , datetime.min.time())
    next_date = start_date + timedelta(days=1)
    filtered_dates = actual[(actual['DateTime'] >= start_date) & (actual['DateTime']< next_date)]
    filtered_dates = filtered_dates.reset_index(drop = True)
    return filtered_dates

#filtered_date = '2010-11-01'
# actual_data_1  = actual_data(filtered_date)  
# potato = search_dates(filtered_date)

def total_dataframe(filtered_date):
    actual_dataset = actual_data(filtered_date)
    predicted_values = search_dates(filtered_date)
    total_df = pd.concat([actual_dataset,predicted_values],axis=1)
    return total_df






def evaluate_model(filtered_date):
    #calculation of the accuracy of the model using root mean sqaured error method
    actual_values = actual_data(filtered_date)
    predicted_values = search_dates(filtered_date)
    columns = ["Unit","App-1","App-2","App-3"]
    actual_values.columns = [["DateTime"] + columns]
    predicted_values.columns = [["DateTime"] + columns]
    eval_actual = actual_values[columns]
    eval_predicted = predicted_values[columns]
    rmse = [["columns","rmse"]]
    for col in range(len(columns)):
        errors = math.sqrt(mean_squared_error(eval_actual.loc[col], eval_predicted.loc[col]))
        rmse.append([columns[col],errors])
    return rmse


















