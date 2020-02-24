#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:01:18 2020

@author: labuser

This script will load the models and calculate the accuracy for each model
"""

import os
import numpy as np
import DeepCluster
from sklearn import preprocessing

signal_path = '/media/labuser/Data/nanopore/pUC19_nanopolish/numpy/'
model_path = '/media/labuser/Data/nanopore/DESPERADO/models/'
models = os.listdir(model_path)

file_out = '/media/labuser/Data/nanopore/DESPERADO/results/nanopolish_results'
f = open(file_out, "w")

for model_ in models:
    # load numpy files
    mod_signal = np.load(signal_path+model_.split('.')[0]+'.npy')
    no_mod_signal = np.load(signal_path+'no_'+model_.split('.')[0]+'.npy')
    
    min_max_scaler = preprocessing.MinMaxScaler()
    mod_signal = min_max_scaler.fit_transform(mod_signal)
    no_mod_signal = min_max_scaler.fit_transform(no_mod_signal)
    
    # to load a saved model
    loaded_model = DeepCluster.DeepCluster(signal_shape=(60,1))
    trained_model = loaded_model.fit()
    trained_model.load_weights(model_path+model_)
    
    f.write('Model '+model_+'\n')

    predictions_mod, _ = trained_model.predict(mod_signal[900:].reshape(100,60,1))
    acc = loaded_model.accuracy(np.zeros(100), predictions_mod.argmax(1))
    f.write('Accuracy test 100 modified reads : '+str(acc))

    f.write('\n')
    predictions_mod, _ = trained_model.predict(no_mod_signal[900:].reshape(100,60,1))
    acc = loaded_model.accuracy(np.ones(100), predictions_mod.argmax(1))
    f.write('Accuracy test 100 non-modified reads :'+ str(acc))

    f.write('\n')

f.close()

