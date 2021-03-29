# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:50:32 2021

@author: klein
"""
import numpy as np
import os
import sys
#path_parent = os.path.dirname(os.getcwd())
#os.chdir(path_parent)
basepath = os.path.abspath('')
sys.path.append(f"{basepath}")
sys.path.append(f"{basepath}/NN")
model_paths = f"{basepath}/Trained_models"
from keras import models
import matplotlib.pyplot as plt
from scipy import stats
# In[Load data]

Data_dict = np.load("Data_tf_ready_QT.npy", allow_pickle=True)[()]
Data_dict_uns = np.load("Test_data_unscaled.npy", allow_pickle=True)[()]

X_train = Data_dict['X_train'].astype('float32')
Y_train = Data_dict['Y_train'].astype('float32')
X_val = Data_dict['X_val'].astype('float32')
Y_val = Data_dict['Y_val'].astype('float32')

X_test = Data_dict_uns['X_test'].astype('float32')
Y_test = Data_dict_uns['Y_test'].astype('float32')
X_test_sc = Data_dict['X_test'].astype('float32')

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
model_name = 'Model2'
model_path = f"{model_paths}/{model_name}"
filepath_best = f"{model_path}/model.hdf5"
model = models.load_model(filepath_best, compile = False)
probabilistic = False

if probabilistic:
    Y_val_pred, sigma_val_pred = predict_prob(
        model, X_val)
else:
    Y_val_pred = model.predict(
        X_val).reshape([len(Y_val)])

# In[Compare]
print(Y_val_pred.shape)
print(Y_val.shape)
print(Y_val_pred[2])#, sigma_val_pred[1])
print(Y_val[2])
# In[]
#plt.hist(Y_val_pred,40)
#plt.hist(Y_val_pred[:,0]-Y_val, 40)
#plt.hist(X_val[:,:,-2].flatten(), 40)
print(Y_val_pred.max(), Y_val_pred.min())
plt.scatter(np.arange(len(Y_val)), Y_val)
plt.scatter(np.arange(len(Y_val)), Y_val_pred)
#plt.yscale('log')
# In[]
plt.scatter(np.arange(len(Y_val)), (Y_val_pred-Y_val)/Y_val)