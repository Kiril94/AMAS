# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:50:32 2021

@author: klein
"""
import numpy as np
import os
import sys
path_parent = os.path.dirname(os.getcwd())
os.chdir(path_parent)
basepath = os.path.abspath('')
sys.path.append(f"{basepath}")
sys.path.append(f"{basepath}/NN")
model_paths = f"{basepath}/Trained_models"
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

# In[Function for predictions]
def predict_prob(model, x, batch_size=2048):
    """Make predictions given model and 2d data
    """
    ypred = model.predict(x, batch_size=batch_size, verbose=1)
    n_outs = int(ypred.shape[1] / 2)
    mean = ypred[:, 0:n_outs]
    sigma = np.exp(ypred[:, n_outs:])
    return mean, sigma

# In[Make predictions]
model_name = 'Test'
model_path = f"{model_paths}/{model_name}"
filepath_best = f"{model_path}/best_weights.hdf5"
model = models.load_model(filepath_best, compile = False)

Weights_and_biases = model.layers[0].get_weights()
Weights = Weights_and_biases[0]
    
for i in range(len(model.layers)):
    layer = model.layers[i].get_weights()#[0]
    shape = np.shape(layer)
    print(shape)
    
Y_val_pred, sigma_val_pred = predict_prob(
    model, X_val)

# In[Compare]
print(Y_val_pred.shape)
print(Y_val.shape)
print(Y_val_pred[200], sigma_val_pred[200])
print(Y_val[200])