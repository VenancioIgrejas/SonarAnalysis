
# coding: utf-8

# In[1]:


import os
import sys
import time
import sqlite3
from Functions.database import createTable
import tensorboard as tf

from envConfig import CONFIG
CONFIG["PACKAGE_NAME"] = os.path.join(CONFIG["OUTPUTDATAPATH"], "PCDTeste")


import numpy as np
import pandas as pd


#from lps_toolbox.metrics.classification import sp_index


analysis_name = 'PCDTeste'

data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']


#db = sqlite3.connect(results_path+'classification.db')
#time = time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime())




# In[2]:


from Functions.dataset.shipClasses import LoadData

dt_24 = LoadData(dev=False)
#dt_24.infoData()
data,trgt= dt_24.getData()



# # Binary classification with one neuron

# In[26]:


#Metrics with binary classes

import sklearn
import numpy as np

import keras.callbacks as callbacks

from keras import backend as K
from keras.utils import to_categorical, get_custom_objects
from sklearn.metrics import recall_score


class Metrics(callbacks.Callback):
    def __init__(self, monitor, verbose=0,verbose_train=1):
        super(callbacks.Callback,self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.verbose_train = verbose_train
        self.monitor_train = []
    
    def on_epoch_end(self, batch, logs={}):
        
        if not self.monitor in ['sp']:
            return
        
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(self.model.predict(X_val))
        
        threshold = 0
        
        if self.monitor == 'sp':
            
            
            pred = np.sum(np.array(map(lambda x: 1 if x >= threshold else 0,y_predict)))
            true = np.sum(np.clip(a=y_val, a_max=1, a_min=0))
            
            sp = float(pred)/true
            
            logs['sp'] = sp
            
            self.monitor_train.append(sp)
            
            monitor_value = logs.get('sp')
            
            if self.verbose > 0:
                print("\n Monitor - epoch: %d - val score(%s): %.6f" % (batch+1,self.monitor, monitor_value))
        
        return
    
    def on_train_end(self, logs=None):
        best_epoch = np.argmax(self.monitor_train)
        best_value = self.monitor_train[best_epoch]
        if self.verbose_train > 0:
            print("\n Monitor - best epoch: %d - best val score(%s): %.6f" % (best_epoch+1,self.monitor, best_value))
        
        return


# In[27]:


import sklearn.model_selection
from sklearn import preprocessing


from keras.models import Sequential, Model
from keras.layers import Input,Dense, Dropout, Activation, merge
from keras.optimizers import SGD,Adam
import keras.callbacks as callbacks
from keras.utils import np_utils

from keras.layers import concatenate

def pcdc_extractor(inputdata, targetdata, trn_params=None):
    ''' 
        This function extracts the Cooperative Principal Components of Discrimination of a Dataset
        
        Parameters:
            inputdata: dataset with inputs
            
            targetdata: each class -> an integer
            
            trn_params: train parameters
            
            trn_params['n_folds'] = number of cross validation folds
            trn_params['n_inits'] = number of initializations
            trn_params['n_pcds'] = number of PCDs to be extracted
            trn_params['norm'] = normalization
            trn_params['learning_rate'] = learning rate
            trn_params['learning_decay'] = learning rate decay
            trn_params['momentum'] = momentum
            trn_params['nesterov'] = nesterov momentum
            trn_params['train_verbose'] = train verbose
            trn_params['n_epochs'] = number of epochs
            trn_params['batch_size'] = batch size
        
    '''
    
    if trn_params == None:
        trn_params = {}
        trn_params['n_folds'] = 2
        trn_params['n_inits'] = 2
        trn_params['n_pcds'] = 2
        trn_params['norm'] = 'none'
        trn_params['learning_rate'] = 0.01
        trn_params['learning_decay'] = 1e-6
        trn_params['momentum'] = 0.3
        trn_params['nesterov'] = True
        trn_params['train_verbose'] = True
        trn_params['n_epochs'] = 300
        trn_params['batch_size'] = 8

    print 'PCD Cooperative Extractor'
    print 'trn_params: ',trn_params
    
    # trained classifiers
    classifiers = {}
    trn_desc = {}
    pcds = {}
    history = {}
    
    CVO = sklearn.model_selection.StratifiedKFold(n_splits=trn_params['n_folds'])
    CVO = list(CVO.split(inputdata, targetdata))
    
    targetdata_sparse = targetdata
    print targetdata_sparse
    for ifold in range(trn_params['n_folds']):
        train_id, test_id = CVO[ifold]

        # normalize data based in train set
        if trn_params['norm'] == 'mapstd':
            scaler = preprocessing.StandardScaler().fit(inputdata[train_id,:])
        elif trn_params['norm'] == 'mapstd_rob':
            scaler = preprocessing.RobustScaler().fit(inputdata[train_id,:])
        elif trn_params['norm'] == 'mapminmax':
            scaler = preprocessing.MinMaxScaler().fit(inputdata[train_id,:])
        
        if trn_params['norm'] != "none":
            norm_inputdata = scaler.transform(inputdata)
        else:
            norm_inputdata = inputdata
         
        
        classifiers[ifold] = {}
        trn_desc[ifold] = {}
        pcds[ifold] = {}
        history[ifold] = {}
        for ipcd in range(trn_params['n_pcds']):
            best_init = 0
            best_loss = 999
            if ipcd == 0:
                # first pcd - random init
                for i_init in range(trn_params['n_inits']):
                    # create the model (Keras - API functional)
                    # This returns a input tensor
                    inputs = Input(shape=(norm_inputdata.shape[1],))
                    
                    # a layer instance is callable on a tensor, and returns a tensor
                    hidden_layer = Dense(1, activation='tanh')(inputs)
                    outputs = Dense(1, activation='tanh')(hidden_layer)
                    
                    model = Model(inputs=inputs, outputs=outputs)
                    
                    # creating a optimization function using steepest gradient
                    sgd = SGD(lr=trn_params['learning_rate'],
                              decay=trn_params['learning_decay'],
                              momentum=trn_params['momentum'],
                              nesterov=trn_params['nesterov'])
                    
                    adam = Adam(lr=trn_params['learning_rate'], 
                                beta_1=0.9, 
                                beta_2=0.999, 
                                epsilon=None, 
                                decay=trn_params['learning_decay'], 
                                amsgrad=False)
                    
                    # compile the model
                    model.compile(loss='mean_squared_error', 
                              optimizer=adam,
                              metrics=['accuracy','mean_squared_error'])
                    
                    # early stopping to avoid overtraining
                    earlyStopping = callbacks.EarlyStopping(
                                            monitor='sp', patience=10,
                                            verbose=trn_params['train_verbose'], mode='max',restore_best_weights=True)
            
                    #save model
                    save_model = callbacks.ModelCheckpoint(filepath='./models/model_%d_fold_%d_pcd-{epoch:02d}-{sp:.2f}.hdf5'%(ifold,ipcd),
                                                   monitor='sp',
                                                   verbose=trn_params['train_verbose'],
                                                   save_best_only=True, 
                                                   mode='max', 
                                                   period=1)
            
            
                    tensorboard = callbacks.TensorBoard(log_dir='./logs_{0}_fold'.format(ifold), histogram_freq=0, batch_size=trn_params['batch_size'],
                                  write_graph=True, write_grads=False,
                                  write_images=False, embeddings_freq=0, 
                                  embeddings_layer_names=None, embeddings_metadata=None, 
                                  embeddings_data=None, update_freq='epoch')
                    # metrics monitorate
            
                    mt = Metrics(monitor='sp',verbose=trn_params['train_verbose'])
            
                    #CSV logger
            
                    log = callbacks.CSVLogger(filename='./models/log_%d.hdf5'%ifold, separator=',', append=False)
                    
                    # Train model
                    init_trn_desc = model.fit(norm_inputdata[train_id], targetdata_sparse[train_id],
                                              epochs=trn_params['n_epochs'], 
                                              batch_size=trn_params['batch_size'],
                                              callbacks=[mt, earlyStopping, save_model], 
                                              verbose=trn_params['train_verbose'],
                                              validation_data=(norm_inputdata[test_id],
                                                               targetdata_sparse[test_id]),
                                              shuffle=True)
                    
                    history[ifold][ipcd] = init_trn_desc.history
                    if np.min(init_trn_desc.history['val_loss']) < best_loss:
                        best_init = i_init
                        best_loss = np.min(init_trn_desc.history['val_loss'])
                        classifiers[ifold][ipcd] = model
                        trn_desc[ifold][ipcd] = init_trn_desc
                        pcds[ifold][ipcd] = model.layers[2].get_weights()[0][0,:][:,np.newaxis]
                    
                    print ('Fold: %i of %i - PCD: %i of %i - Init: %i of %i - finished with val cost: %1.3f'%
                           (ifold+1,trn_params['n_folds'],
                            ipcd+1,trn_params['n_pcds'],
                            i_init+1,trn_params['n_inits'],
                            best_loss
                           ))
                
                    
            else: # ipcd != 0
                # from second pcd to the end - freeze previous neurons and create a new neuron
                for i_init in range(trn_params['n_inits']):
                    # create the model (Keras - API functional)
                    # This returns a input tensor
                    inputs = Input(shape=(norm_inputdata.shape[1],))
                    
                    # add a non-linear freeze previous extracted pcd 
                    freeze_layer = Dense(ipcd, activation='tanh',trainable=False)(inputs)
                    
                    # add a non-linear no-freeze new neuron
                    non_freeze_layer = Dense(1, activation='tanh',trainable=True)(inputs)
                    
                    #concatenate both, non-freeze and freeze layers
                    hidden_layer = concatenate([freeze_layer, non_freeze_layer])
                    outputs = Dense(1, activation='tanh')(hidden_layer)
                    
                    model = Model(inputs=inputs, outputs=outputs)
                    freeze_weights = model.layers[1].get_weights()
                    
                    
                    #return freeze_weights, pcds,ipcd, ifold, model
                    
                    #for i_old_pcd in range(ipcd):
                    #    for idim in range(norm_inputdata.shape[1]):
                    #        freeze_weights[0][idim,i_old_pcd] = pcds[ifold][i_old_pcd][idim]
                    
                    model.layers[1].set_weights(freeze_weights)
                    
                    
                    # creating a optimization function using steepest gradient
                    sgd = SGD(lr=trn_params['learning_rate'],
                              decay=trn_params['learning_decay'],
                              momentum=trn_params['momentum'],
                              nesterov=trn_params['nesterov'])
                    
                    # compile the model
                    model.compile(loss='mean_squared_error', 
                              optimizer=adam,
                              metrics=['accuracy','mean_squared_error'])
                    
                    # early stopping to avoid overtraining
                    earlyStopping = callbacks.EarlyStopping(
                                            monitor='sp', patience=10,
                                            verbose=trn_params['train_verbose'], mode='max',restore_best_weights=True)
            
                    #save model
                    save_model = callbacks.ModelCheckpoint(filepath='./models/model_%d_fold_%d_pcd-{epoch:02d}-{sp:.2f}.hdf5'%(ifold,ipcd),
                                                   monitor='sp',
                                                   verbose=trn_params['train_verbose'],
                                                   save_best_only=True, 
                                                   mode='max', 
                                                   period=1)
            
            
                    tensorboard = callbacks.TensorBoard(log_dir='./logs_{0}_fold'.format(ifold), histogram_freq=0, batch_size=trn_params['batch_size'],
                                  write_graph=True, write_grads=False,
                                  write_images=False, embeddings_freq=0, 
                                  embeddings_layer_names=None, embeddings_metadata=None, 
                                  embeddings_data=None, update_freq='epoch')
                    # metrics monitorate
            
                    mt = Metrics(monitor='sp',verbose=trn_params['train_verbose'])
            
                    #CSV logger
            
                    log = callbacks.CSVLogger(filename='./models/log_%d.hdf5'%ifold, separator=',', append=False)
                    
                    # Train model
                    init_trn_desc = model.fit(norm_inputdata[train_id], 
                                              targetdata_sparse[train_id],
                                              epochs=trn_params['n_epochs'], 
                                              batch_size=trn_params['batch_size'],
                                              callbacks=[mt, earlyStopping, save_model], 
                                              verbose=trn_params['train_verbose'],
                                              validation_data=(norm_inputdata[test_id],
                                                               targetdata_sparse[test_id]),
                                              shuffle=True)
                    
                    history[ifold][ipcd] = init_trn_desc.history
                    if np.min(init_trn_desc.history['val_loss']) < best_loss:
                        best_init = i_init
                        best_loss = np.min(init_trn_desc.history['val_loss'])
                        classifiers[ifold][ipcd] = model
                        trn_desc[ifold][ipcd] = init_trn_desc
                        pcds[ifold][ipcd] = model.layers[2].get_weights()[0]
                        
                    print ('Fold: %i of %i - PCD: %i of %i - Init: %i of %i - finished with val cost: %1.3f'%
                           (ifold+1,trn_params['n_folds'],
                            ipcd+1,trn_params['n_pcds'],
                            i_init+1,trn_params['n_inits'],
                            best_loss
                           ))
               
                    
    # add cross-validation information in train desc.
    trn_desc['CVO'] = CVO
    trn_desc['history'] = history
                    
    return [pcds,classifiers,trn_desc]


# In[28]:


K.clear_session()

# Extract PCD Independent
trn_params = {}
trn_params['n_folds'] = 10
trn_params['n_inits'] = 1
trn_params['n_pcds'] = 5
trn_params['norm'] = 'mapstd'
trn_params['learning_rate'] = 0.01
trn_params['learning_decay'] = 1e-4
trn_params['momentum'] = 0.9
trn_params['nesterov'] = True
trn_params['train_verbose'] = True
trn_params['n_epochs'] = 1000
trn_params['batch_size'] = 1200

#class, not class (class 23 because have 45.589 samples)

c23_trgt = -np.ones(trgt.shape,dtype=int)
#c23_trgt = np.zeros(trgt.shape,dtype=int)

c23_trgt[trgt==22] = 1

print "number of samples class %d"%sum(c23_trgt==1)
print "number of samples not class %d\n"%sum(c23_trgt==-1)


#[pcds,pcd_classifiers,pcd_trn_desc] = pcdc_extractor(data,trgt,trn_params)
[pcds,classifiers,trn_desc] = pcdc_extractor(data,c23_trgt,trn_params)

