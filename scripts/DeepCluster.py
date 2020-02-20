#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:24:43 2019

Deep Clustering

@author: labuser

This sccript has the DeepCluster class
"""
from keras.models import Model
from keras import backend as K
from keras import layers
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape
from keras.models import Model
import numpy as np
from keras.datasets import mnist
from keras.engine.topology import Layer, InputSpec
from sklearn import preprocessing
from math import ceil
import os 
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import pickle
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy import stats
import random
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import tensorflow as tf
from keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import sys
from numpy.random import seed
from tensorflow import set_random_seed    
from sklearn.metrics import davies_bouldin_score # use this to measure how good clusters are
from sklearn.utils.linear_assignment_ import linear_assignment
from keras.models import load_model


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepCluster(ClusteringLayer):
    
    def __init__(self,  signal_shape=(50, 1)):
        self.signal_shape = signal_shape
        print('importing DeepCluster class')
        
        
    def autoencoderConv1D(self, signal_shape):
        """
        Conv2D auto-encoder model.
        Arguments:
            img_shape: e.g. (28, 28, 1) for MNIST
        return:
            (autoencoder, encoder), Model of autoencoder and model of encoder
        """
        input_img = Input(shape=signal_shape)
        # Encoder
        x = Conv1D(16, 3, activation='relu', padding='same')(input_img)
        x = Conv1D(8, 3, activation='relu', padding='same')(x)
        x = Conv1D(8, 3, activation='relu', padding='same')(x)
        shape_before_flattening = K.int_shape(x)
        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        x = Flatten()(x)
        encoded = Dense(5, activation='relu', name='encoded')(x)
    
        # Decoder
        x = Dense(np.prod(shape_before_flattening[1:]),
                    activation='relu')(encoded)
        # Reshape into an image of the same shape as before our last `Flatten` layer
        x = Reshape(shape_before_flattening[1:])(x)
    
        x = Conv1D(8, 3, activation='relu', padding='same')(x)
        #x = UpSampling1D((2, 2))(x)
        x = Conv1D(8, 3, activation='relu', padding='same')(x)
        #x = UpSampling1D((2, 2))(x)
        x = Conv1D(16, 3, activation='relu', padding='same')(x)
        #x = UpSampling1D((2, 2))(x)
        decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
    
        return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')
    
    def accuracy(self, y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    
    def fit(self, x=None, y=None, shuffle_=None, 
            pretrain_epochs=100, 
            batch_size_au=256,
            maxiter_DC=7000,
            update_interval=140,
            n_clusters=2,
            seed_value=42):
        '''
        Fit the model 
        '''
        if not x:
            autoencoder, encoder = self.autoencoderConv1D(self.signal_shape)
            autoencoder.compile(optimizer='adadelta', loss='mse')
            clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
            model = Model(inputs=encoder.input,
                      outputs=[clustering_layer, autoencoder.output])
            model.compile(loss=['kld', 'mse'], loss_weights=[0.3, 1], optimizer='adam')
            return model

        # 2. Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed_value)# 3. Set `numpy` pseudo-random generator at a fixed value
        np.random.seed(seed_value)# 4. Set `tensorflow` pseudo-random generator at a fixed value
        tf.set_random_seed(seed_value)# 5. For layers that introduce randomness like dropout, make sure to set seed values 
           
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        nmi_f = normalized_mutual_info_score
        ari_f = adjusted_rand_score
        
        if shuffle_:
            x, y = shuffle(x, y, random_state=seed_value)
    
        # Baseline1 raw data
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=10, random_state = seed_value)
        y_pred_kmeans = kmeans.fit_predict(x.reshape((x.shape[0], x.shape[1])))
        
        print('Acc. k-means', self.accuracy(y, y_pred_kmeans))
        
        batch_size = batch_size_au
        
        autoencoder, encoder = self.autoencoderConv1D(self.signal_shape)
        
        autoencoder.summary()
        
        autoencoder.compile(optimizer='adadelta', loss='mse')
        autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)
        
        # Baseline 2
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=10, random_state=42)
        y_pred_kmeans = kmeans.fit_predict(encoder.predict(x))
        print('Acc. Autoencoder', self.accuracy(y, y_pred_kmeans))
       
        # build the clustering layer
        clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
        
        model = Model(inputs=encoder.input,
                      outputs=[clustering_layer, autoencoder.output])
        
        model.compile(loss=['kld', 'mse'], loss_weights=[0.3, 1], optimizer='adam')
        model.summary()

        #Initialize cluster centers k-means
        model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        
        loss = 0
        index = 0
        maxiter = maxiter_DC
        update_interval = update_interval
        index_array = np.arange(x.shape[0])
        # change the batch size to the number of samples
        #batch_size = x.shape[0]
        
        # computing an auxiliary target distribution
        def target_distribution(q):
            weight = q ** 2 / q.sum(0)
            return (weight.T / weight.sum(1)).T
        
    
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _  = model.predict(x, verbose=0)
                p = target_distribution(q)  # update the auxiliary target distribution p
        
                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = self.accuracy(y, y_pred)
                    ari = ari_f(y, y_pred)
                    nmi = nmi_f(y, y_pred)
                    loss = loss
                    #print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
        
        return model
    
    def save_model(self, model, path):
        '''
        Save the weight of the model
        '''
        model.save(path)
        return True
    
    def load_model(self, path):
        '''
        load a model
        '''
        model = load_model(path)
        return model
    

        
        
        
    
    
    
    

    
    
