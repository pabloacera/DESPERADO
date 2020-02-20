#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:41:38 2020

@author: labuser

This script will take the numpy arrays save from either nanopolish or Tombo and run DESPERADO
"""

import numpy as np
import DeepCluster

path ='/media/labuser/Data/nanopore/pUC19_nanopolish/numpy/'

mod_signal = np.load(path+'mod545_CCTGG_np.npy')
no_mod_signal = np.load(path+'no_mod545_CCTGG_np.npy')

print(mod_signal.shape, no_mod_signal.shape)

train = np.concatenate((no_mod_signal[:530], mod_signal[:530]))

train_x = train.reshape((train.shape[0], train.shape[1], 1))
train_y = np.concatenate((np.repeat(1, 530), np.repeat(0, 530)))

seed_value = 42 # random seed

model = DeepCluster.DeepCluster()

model.fit(train_x, train_y, seed_value=seed_value)

