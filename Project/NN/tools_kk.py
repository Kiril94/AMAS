# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:55:46 2020

@author: klein
"""


import keras
import time

################################################################

class TimeHistory(keras.callbacks.Callback):
    
    """Measures training time."""
    
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


