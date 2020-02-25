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
#path ='/media/labuser/Data/nanopore/pUC19_nanopolish/numpy/'

motifs_nano = [
          'mod354_CCAGG_np.npy',
          'mod545_CCTGG_np.npy',
          'mod833_CCAGG_np.npy',
          'mod954_CCAGG_np.npy',
          'mod967_CCTGG_np.npy'
         ]

motif_tombo = [
                'motif6_CCAGGAACC_1000.npy',
                'motif6_CCAGGCGTTT_1000.npy',
                'motif6_CCAGGGTTT_1000.npy',
                'motif6_CCTGGAAGC_1000.npy',
                'motif6_CCTGGGGTG_1000.npy'
               ]

motif_tombo5 = [
                'motif_CCAGGAACC_1000.npy',
                'motif_CCAGGCGTTT_1000.npy',
                'motif_CCAGGGTTT_1000.npy',
                'motif_CCTGGAAGC_1000.npy',
                'motif_CCTGGGGTG_1000.npy'
               ]

path = '/media/labuser/Data/nanopore/pUC19/processed/numpy/tombo/n_prepro/5-mers/1000/'


file_out = '/media/labuser/Data/nanopore/DESPERADO/results/tombo_results_5_kmenasCOP+semi'
f = open(file_out, "w")

for motif in motif_tombo5:
    
    f.write('Model '+motif+'\n')

    # load numpy files
    mod_signal = np.load(path+motif)
    no_mod_signal = np.load(path+'motif_mod_'+motif.split('_')[1]+'_'+motif.split('_')[2])
    
    #mod_signal = np.load('/media/labuser/Data/nanopore/pUC19/processed/numpy/tombo/motif_CCTGGTATCT_500.npy')
    #no_mod_signal = np.load('/media/labuser/Data/nanopore/pUC19/processed/numpy/tombo/motif_mod_CCTGGTATCT_500.npy')
    
    #make train 
    train = np.concatenate((no_mod_signal[:900], mod_signal[:900]))
    
    #min_max_scaler = preprocessing.MinMaxScaler()
    #train = train.reshape((len(train), 50))
    #train = min_max_scaler.fit_transform(train)
    
    train_x = train.reshape((train.shape[0], train.shape[1], 1))
    train_y = np.concatenate((np.repeat(1, 900), np.repeat(0, 900)))
    
    #train = np.concatenate((no_mod_signal, mod_signal))
    #train_x = train.reshape((train.shape[0], train.shape[1], 1))
    #train_y = np.concatenate((np.repeat(1, 500), np.repeat(0, 500)))
    
    # define the model and fit DC
    seed_value = 42 # random seed
    model = DeepCluster.DeepCluster(signal_shape=(50,1))
    DC = model.fit(train_x, 
                   train_y,
                   seed_value=seed_value, 
                   shuffle_=False, 
                   verbose=True,
                   file_out=f,
                   N_no_mod=900)
    
    predictions_train, _ = DC.predict(train_x, verbose=0)
    labels = predictions_train.argmax(1)
    f.write('training accuracy :'+ str(model.accuracy(train_y, labels))+'\n')
    
    mod_test = mod_signal[900:]
    nomod_test = no_mod_signal[900:]
    
    #mod_test = min_max_scaler.transform(mod_test)
    #nomod_test = min_max_scaler.transform(nomod_test)
    
    mod_test = mod_test.reshape((len(mod_test), 50, 1))
    nomod_test = nomod_test.reshape((len(nomod_test), 50, 1))
    
    predictions_mod, _ = DC.predict(mod_test, verbose=0)
    labels = predictions_mod.argmax(1)
    f.write('Accuracy for modified signals :'+str(model.accuracy(np.ones(len(labels)), labels))+'\n')
    
    predictions_nomod, _ = DC.predict(nomod_test, verbose=0)
    labels = predictions_nomod.argmax(1)
    f.write(' Accuracy for unmodified signals :'+ str(model.accuracy(np.ones(len(labels)), labels))+'\n')
    f.write('\n')
    DC.save_weights(filepath='/media/labuser/Data/nanopore/DESPERADO/models/'+motif+'_tombo5_weight.h5')

f.close()
'''
# to load a saved model
new_model=DeepCluster.DeepCluster(signal_shape=(60,1))
new_model = new_model.fit()
new_model.load_weights('/media/labuser/Data/nanopore/DESPERADO/models/weight.h5')
'''
