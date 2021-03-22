# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:13:27 2021

@author: klein
"""
import os
import sys
path_parent = os.path.dirname(os.getcwd())
os.chdir(path_parent)
basepath = os.path.abspath('')

#project_path = os.path.dirname(basepath)
sys.path.append(f"{basepath}/NN")
model_paths = f"{basepath}/Trained_models"
import loss
import tools_kk
import numpy as np
import pandas as pd
from keras import layers 
from keras import optimizers
from keras import callbacks
from keras import models

# In[Load data]
Data_obj = np.load("Data_tf_ready_QT.npy", allow_pickle=True)
Data_dict = Data_obj[()]

X_train = Data_dict['X_train'].astype('float32')
Y_train = Data_dict['Y_train'].astype('float32')
X_val = Data_dict['X_val'].astype('float32')
Y_val = Data_dict['Y_val'].astype('float32')
Y_train = np.expand_dims(Y_train, axis = 1)
Y_val = np.expand_dims(Y_val, axis = 1)

Data = [X_train, Y_train, X_val, Y_val]

# In[Train functions]
def train_new_model(Data, opts):
    """Takes Data with [X_train, Y_train, X_val, Y_val] and options and trains
    the network"""
    X_train, Y_train, X_val, Y_val = Data
    #choose a directory to store a model
    model_directory = f"{model_paths}/{opts['Model_name']}"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    #save description
    f = open(f"{model_directory}/description.log","w+")
    f.write(opts['Description'])
    f.close()
    pat = opts['Patience']

    callback_list = []
    #create early stopping callback to prevent model from overfitting
    es = callbacks.EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=pat)
    callback_list.append(es)

    filepath_best = f"{model_directory}/best_weights.hdf5"
    mccb = callbacks.ModelCheckpoint(
        filepath_best,monitor = 'val_loss', save_best_only=True)
    callback_list.append(mccb)

    #measures time of training
    time_callback = tools_kk.TimeHistory()
    callback_list.append(time_callback)
    
    model = models.Sequential()
    Hidden_units = opts['Hidden_units']
    model.add(layers.LSTM(units = Hidden_units[0], return_sequences = True,
               input_shape = (X_train.shape[1], X_train.shape[2])))
    #Perform some dropout regularization
    model.add(layers.Dropout(0.2))
    #Adding three more LSTM layers with dropout regularization
    for i in range(len(Hidden_units)-1):
        ret_seq = True
        if i==int(len(Hidden_units)-2):
            ret_seq = False
        model.add(
            layers.LSTM(units = Hidden_units[i-1], return_sequences = ret_seq))
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(2))
    model.compile(loss=loss.gaussian_nll, 
                  optimizer=optimizers.Adam(amsgrad = True))
     
    batch_size = opts['Batch_size']
    epochs = opts['Epochs']
    
    model.summary()
    history = model.fit(
        X_train, Y_train, validation_data=(X_val,Y_val),
        batch_size =batch_size, epochs = epochs, shuffle = True,
        callbacks = callback_list)
    #save model after training
    #save weights, params and architecture to HDF5
    model.save(f"{model_directory}/model.hdf5")

    #save training history
    hist_df = pd.DataFrame(history.history) 
    times = time_callback.times

    hist_df.insert(2,"time",  np.array(times))
    hist_csv_file = f"{model_directory}/history.csv"
    with open(hist_csv_file, mode='a') as f:
        hist_df.to_csv(f, header = True)
  
    print("Training finished .\
          New weights saved to {}.".format(model_directory))
    return 0

def train_existing_model(Data,opts):

  """Takes Data and opts and trains an existing model."""

  model_directory = opts['Model_directory']  
  callback_list = []
  #create early stopping callback to prevent model from overfitting
  es = callbacks.EarlyStopping(
      monitor='val_loss', mode='min', verbose=1, patience=opts['Patience'])
  callback_list.append(es)

  filepath_best = f"{model_directory}/best_weights.hdf5"
  mccb = callbacks.ModelCheckpoint(
      filepath_best,monitor = 'val_loss', save_best_only=True)
  callback_list.append(mccb)

  #measures time of training
  time_callback = tools_kk.TimeHistory()
  callback_list.append(time_callback)

  model = models.load_model(filepath_best, compile = False)  #load model
  model.load_weights(filepath_best)#load weights

  model.compile(loss=loss.gaussian_nll, optimizer=optimizers.Adam(amsgrad = True))
     
  X_train, Y_train, X_val, Y_val = Data
  batch_size = opts['Batch_size']
  epochs = opts['Epochs']

  print("Training model in {}".format(model_directory))
  history = model.fit(X_train, Y_train, validation_data=(X_val,Y_val),
                    batch_size=batch_size, epochs = epochs, shuffle = True,
                    callbacks = callback_list)

  hist_df = pd.DataFrame(history.history) 
  times = time_callback.times

  hist_df.insert(2,"time",  np.array(times))
  hist_csv_file = f"{model_directory}/history.csv"
  with open(hist_csv_file, mode='a') as f:
      hist_df.to_csv(f, header = False)

  print("Training finished .\
  New weights saved to {}.".format(model_directory))
  return 0

# In[Train]
model_name = 'Model0'
description = 'Test description'
patience = 200#if val loss doesn't go down, stop after patience steps and save model
opts = {'Model_name':model_name, 'Description':description,
        'Patience':patience, 'Batch_size':200, 'Epochs':1000,
        'Hidden_units':[50,50,50,50,45]}
train_new_model(Data, opts)

