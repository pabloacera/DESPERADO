#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:24:43 2019

Deep Clustering

@author: Pablo

This sript will make the preprocess of looking for motifs in puC19 and copy the numpy objects to 
a folder
"""

import numpy as np
from sklearn import preprocessing
from math import floor
import os 
import pickle
from scipy import stats


def load_object(filename):
    with open(filename, 'rb') as input:
        name = pickle.load(input)
        return name

def MAD(raw_signal):
    return stats.median_absolute_deviation(raw_signal)

def de_normalize_signal(norm_signal, shift, scale):
    de_normalized = (norm_signal*scale)+shift
    return de_normalized

def normalize_out_outliers(raw_signal):
    '''
    This function normalize the data using min_max_scaler and put outliers at 2.5 MAD 
    '''
    # get median absolute deviation
    MAD_signal = MAD(raw_signal)
    # get meadia
    median_signal = np.median(raw_signal)
    
    # convert using MAD
    raw_signal_normMAD = (raw_signal - median_signal) / MAD_signal
    
    raw_signal_normMAD = np.array([2.5 if i > 2.5 else -2.5 if i < -2.5 else i for i in raw_signal_normMAD])
    
    # Recover original value signal using smae MAD and median parameters as before because I do not 
    # want the new values to affect my MAD
    raw_signal_ = (raw_signal_normMAD * MAD_signal) + median_signal
    
    # [0,1] scale the data
    min_max_scaler = preprocessing.MinMaxScaler()
    raw_signal_ =  raw_signal.reshape((len(raw_signal_),1))
    raw_signal_ = min_max_scaler.fit_transform(raw_signal_)
    
    # Convert array to list 
    raw_signal_ = raw_signal_.reshape((len(raw_signal_)))
    
    return raw_signal_



def top_median(array, lenght):
    '''
    This function top an array until some specific lenght
    '''
    extra_measure = [np.median(array)]*(lenght-len(array))
    array += extra_measure
    return array



def smooth_event(raw_signal, file_obj, lenght_events):
    '''
    Make list of list containing the events
    Reduce the size of the signal by subsampling events using PDF of the events 
    '''
    raw_signal_events = []
    for i in range(0, len(file_obj.segs)-1):
        event = sorted(list(raw_signal[file_obj.segs[i]:file_obj.segs[i+1]]))
        if len(event) < lenght_events:
            event = top_median(event, lenght_events)
            raw_signal_events = raw_signal_events + [event]
        else:
            division = floor(len(event)/lenght_events)
            new_event = []
            for i in range(0, len(event), division):
                new_event.append(np.median(event[i:i+1]))
                if len(new_event) == 10:
                    break         
               
            if len(new_event) < lenght_events:
                new_event = top_median(new_event, lenght_events)
            raw_signal_events = raw_signal_events + [new_event]

    return raw_signal_events


def find_motif_measurements(files, path, number_reads, motif_letter):
    '''
    Find the measurements for specific motifs and take list of lists containing ind signal values for
    these motifs
    '''
    
    motifs = []
    # read object files, extract signal
    for file in files:
       
        file_obj = load_object(path+"/"+file)
        #if file_obj.genome_loc[1] == "-":
        #    continue
        
        
        if file_obj.genome_seq.find(motif_letter) == -1:
            continue
            
        raw_signal = de_normalize_signal(file_obj.raw_signal, 
                                         file_obj.scale_values[0],
                                         file_obj.scale_values[1])
        
        raw_signal_ = normalize_out_outliers(raw_signal)
        
        raw_signal_events = smooth_event(raw_signal_, file_obj, 10)

        try:
            ### I am going to use the motif '' and is going to take measurements for the first 3 nc
            list_inception = raw_signal_events[file_obj.genome_seq.find(motif_letter):file_obj.genome_seq.find(motif_letter)+5]
            smooth_list =  [item for sublist in list_inception for item in sublist]  
            
            motifs.append(smooth_list)
           
        except FloatingPointError:
            pass
        
        if len(motifs) >= number_reads:
            break
        
    return motifs


if __name__ == '__main__':
    
    NB07_files = os.listdir('/media/labuser/Data/nanopore/pUC19/processed/NB07_obj')
    path = '/media/labuser/Data/nanopore/pUC19/processed/NB07_obj'
     
    NB08_files = os.listdir('/media/labuser/Data/nanopore/pUC19/processed/NB08_obj')
    files_path_mod = '/media/labuser/Data/nanopore/pUC19/processed/NB08_obj'
    
    motif_letter = ['CCTGGGGTG',
                    'CCTGGAAGC',
                    'CCAGGAACC',
                    'CCAGGCGTTT',
                    'CCAGGGTTT'
                   ]
    
    for sequence in motif_letter:
        
        motif = find_motif_measurements(NB07_files, path, 1000, sequence)    
        # Select a motif for a number of reads in the test 
        motif_mod = find_motif_measurements(NB08_files, files_path_mod, 1000, sequence)
        
        save_np = '/media/labuser/Data/nanopore/pUC19/processed/numpy/tombo/n_prepro/5-mers'
        np.save(save_np+'/motif_'+sequence+'_1000', motif)
        np.save(save_np+'/motif_mod_'+sequence+'_1000', motif_mod)
