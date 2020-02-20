#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:41:38 2020

@author: labuser

This script will take the numpy arrays save from either nanopolish or Tombo and run DESPERADO
"""

import numpy as np
import DeepCluster
from sklearn import preprocessing

#path to the numpy files
path ='/media/labuser/Data/nanopore/pUC19_nanopolish/numpy/'

motifs = [
          'mod354_CCAGG_np.npy',
          'mod545_CCTGG_np.npy',
          'mod833_CCAGG_np.npy',
          'mod954_CCAGG_np.npy',
          'mod967_CCTGG_np.npy'
         ]

for motif in motifs:
    # load numpy files
    mod_signal = np.load(path+motif)
    no_mod_signal = np.load(path+'no_'+motif)
    
    #make train 
    train = np.concatenate((no_mod_signal[:900], mod_signal[:900]))
    
    min_max_scaler = preprocessing.MinMaxScaler()
    train = train.reshape((len(train),60))
    train = min_max_scaler.fit_transform(train)
    
    train_x = train.reshape((train.shape[0], train.shape[1], 1))
    train_y = np.concatenate((np.repeat(1, 900), np.repeat(0, 900)))
    
    # define the model and fit DC
    seed_value = 42 # random seed
    model = DeepCluster.DeepCluster(signal_shape=(60,1))
    DC = model.fit(train_x, train_y, seed_value=seed_value, shuffle_=True)
    
    predictions_train, _ = DC.predict(train_x, verbose=0)
    labels = predictions_train.argmax(1)
    print(model.accuracy(train_y, labels))
    
    mod_test = mod_signal[900:]
    nomod_test = no_mod_signal[900:]
    
    mod_test = min_max_scaler.transform(mod_test)
    nomod_test = min_max_scaler.transform(nomod_test)
    
    mod_test = mod_test.reshape((len(mod_test), 60, 1))
    nomod_test = nomod_test.reshape((len(nomod_test), 60, 1))
    
    predictions_mod, _ = DC.predict(mod_test, verbose=0)
    labels = predictions_mod.argmax(1)
    print(model.accuracy(np.ones(len(labels)), labels))
    
    predictions_nomod, _ = DC.predict(nomod_test, verbose=0)
    labels = predictions_nomod.argmax(1)
    print(model.accuracy(np.ones(len(labels)), labels))
    
    DC.save_weights(filepath='/media/labuser/Data/nanopore/DESPERADO/models/'+motif+'_weight.h5')

'''
# to load a saved model
new_model=DeepCluster.DeepCluster(signal_shape=(60,1))
new_model = new_model.fit()
new_model.load_weights('/media/labuser/Data/nanopore/DESPERADO/models/weight.h5')
'''
