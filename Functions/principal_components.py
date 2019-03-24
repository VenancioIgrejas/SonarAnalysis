"""
  This file contents some processing functions

Autor: Natanael Nunes de Moura Junior
gitHub: https://github.com/natmourajr

Some parts of the code was changed by Venancio Igrejas
"""

import os
import numpy as np
from os.path import exists
from pathlib2 import Path 

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, concatenate
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.models import load_model

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

from keras import backend as K

from Functions.callbackKeras import metricsAdd, StopTraining
from Functions.preprocessing import class_weight_Keras

from sklearn.base import BaseEstimator
from sklearn.externals import joblib

import pandas as pd

class TrnParams(object):
    def __init__(self, learning_rate=0.001,
    learning_decay=1e-6, momentum=0.3,
    nesterov=True, train_verbose=False, verbose= False,
    n_epochs=500,batch_size=8, n_inits=1, 
    monitor = 'val_loss', mode='auto',patience=25):
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.train_verbose = train_verbose
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_inits = n_inits
        self.monitor = monitor
        self.mode = mode
        self.patience = patience

    def __str__(self):
        m_str = 'Class TrnParams\n'
        for i in self._generate_str():
            m_str = m_str + i + '\n'

        return m_str

    def _generate_str(self):
        for key,values in self.__dict__.iteritems():
            yield ' {0}: {1}'.format(key,values)

class PCDCooperativeBase(object):

    """
    PCD Cooperative class
        This class implement the Principal Component of Discrimination Analysis in Cooperative Approach
    """
    def __init__(self, n_components=2, activation='tanh', dir=None):
        """
        PCD Cooperative constructor
            n_components: number of components to be extracted
        """
        self.n_components = n_components
        self.models = {}
        self.trn_descs = {}
        self.dir = dir
        self.pcds = None
        self.activation = activation
        self._y = None

    def __str__(self):
        m_str = 'Class PCDCooperation'
        m_str = '%s\n %s'%(m_str,'Components: %i'%(self.n_components))
        m_str = '%s\n %s'%(m_str,'Activation: %s'%(self.activation))
        if self.pcds is None:
            m_str = '%s\n %s'%(m_str,'Obj: Not fitted')
        else:
            m_str = '%s\n %s'%(m_str,'Obj: fitted')
        return m_str

    def fit(self, inputdata, targetdata, trn_params=None, trn_tst_idx=None, sample_weight=None):
        '''
            This function extracts the Cooperative Principal Components of Discrimination of a Dataset

            Parameters:
                inputdata: normalized inputs

                targetdata: each class -> an integer

                trn_params: train parameters

                trn_tst_idx: [trn, tst], indexes for fit

        '''

        if trn_params == None:
            print('Creating TrnParams')
            trn_params = TrnParams()
        if trn_params.verbose: print ('PCD Cooperative Extractor')
        if trn_params.verbose: print (trn_params)

        if trn_tst_idx is None:
            if trn_params.verbose: print ('Warning: train_id was passed as None')
            CVO = StratifiedKFold(n_splits=10)
            CVO = list(CVO.split(inputdata, targetdata))
            trn_tst_idx = CVO[0]

        self.pcds = np.zeros([self.n_components,inputdata.shape[1]])

        trainId, testId = trn_tst_idx

        # from each class an integer -> target max sparse

        if -1 in targetdata:
            #ATENTION: it's happend only if you use PCDCooperation as estimator of HierarqClassification
            #this code change the None class -1 in class 0 for use 'class_weight_Keras' without problem
            #so we will considere None class as class 0 but the train_id and test_id will not recognize
            # the 'new' class None.
            #
            #in other worlds: 
            #shape of targetdata != shape of targetdata[train_id] + shape of targetdata[test_id]
            #because targetdata will contain None class target
            targetdata[targetdata==-1] = 0
            
        targetdata_sparse = np_utils.to_categorical(targetdata)
        
        if self.activation is 'tanh':
            targetdata_sparse = 2*targetdata_sparse -1

        self._y = targetdata_sparse

        if trn_params.verbose: print ('[PCD Cooperative Extractor]: PCD Loop')
        for ipcd in range(self.n_components):
            if trn_params.verbose: print (' Extracting %i PCD'%(ipcd))
            best_init = 0
            best_loss = 999
            best_sp = -np.inf

            if ipcd == 0:# first pcd - random init


                #load ipcd models if file exists
                if isinstance(self.dir,str):
                    if self._load(ipcd, path=self.dir,filename="pcd_coop_obj"):
                        for idx in range(inputdata.shape[1]):
                            self.pcds[ipcd, idx] = self.models[ipcd].layers[1].get_weights()[0][idx]

                        continue

                for i_init in range(trn_params.n_inits):
                    if trn_params.verbose: print ('[+] {0} of {1} inicialization'.format(i_init+1,trn_params.n_inits))
                    callback_list = []
                    # create the model (Keras - API functional)
                    # https://keras.io/getting-started/functional-api-guide/
                    # This returns a input tensor
                    inputs = Input(shape=(inputdata.shape[1],))
                    # a layer instance is callable on a tensor, and returns a tensor
                    hidden_layer = Dense(1, activation=self.activation)(inputs)
                    outputs = Dense(targetdata_sparse.shape[1], activation='tanh')(hidden_layer)

                    model = Model(inputs=inputs, outputs=outputs)

                    # creating a optimization function using steepest gradient
                    sgd = SGD(lr=trn_params.learning_rate,
                              decay=trn_params.learning_decay,
                              momentum=trn_params.momentum,
                              nesterov=trn_params.nesterov)

                    adam = Adam(lr=trn_params.learning_rate, 
                                beta_1=0.9,  
                                beta_2=0.999, 
                                epsilon=None, 
                                decay=0.0, 
                                amsgrad=False)

                    # compile the model
                    model.compile(loss='mean_squared_error',
                                  optimizer=adam,
                                  metrics=['accuracy','mean_squared_error'])

                    if trn_params.monitor is 'sp':
                        trn_params.mode = 'max'
                    mt_sp = metricsAdd(monitor='sp', verbose=trn_params.train_verbose,verbose_train=trn_params.train_verbose)
                    callback_list.append(mt_sp)


                    # early stopping to avoid overtraining
                    earlyStopping = callbacks.EarlyStopping(
                        monitor=trn_params.monitor, patience=trn_params.patience,
                        verbose=trn_params.train_verbose, 
                        mode=trn_params.mode, 
                        restore_best_weights=True)


                    st = StopTraining(monitor=trn_params.monitor,restore_best_weights=True,
                                      verbose=trn_params.train_verbose,patience=trn_params.patience,
                                      min_delta=0.03)

                    callback_list.append(st)



                    csv_logger = callbacks.CSVLogger('%s/training.csv'%(self.dir))

                    callback_list.append(csv_logger)

                    if sample_weight is None:
                        class_weight = class_weight_Keras(targetdata[trainId])
                        sample_weight_fit = sample_weight
                    else:
                        class_weight = None
                        sample_weight_fit = sample_weight[trainId]
                    # Train model
                    init_trn_desc = model.fit(inputdata[trainId], targetdata_sparse[trainId],
                                              epochs=trn_params.n_epochs,
                                              batch_size=trn_params.batch_size,
                                              callbacks=callback_list,
                                              verbose=trn_params.train_verbose,
                                              validation_data=(inputdata[testId],
                                                               targetdata_sparse[testId]),
                                              class_weight=class_weight,
                                              sample_weight=sample_weight_fit,
                                              shuffle=True)

                    if trn_params.monitor is 'sp':
                        bool_trn = (np.max(init_trn_desc.history['sp']) > best_sp)
                    else:
                        bool_trn = (np.min(init_trn_desc.history['val_loss']) < best_loss)

                    if bool_trn:
                        best_init = i_init
                        best_loss = np.min(init_trn_desc.history['val_loss'])
                        if trn_params.monitor is 'sp':
                            best_sp = np.max(init_trn_desc.history['sp'])
                        self.models[ipcd] = model
                        df = pd.read_csv('%s/training.csv'%(self.dir))
                        self.trn_descs[ipcd] = df
                        os.remove('%s/training.csv'%(self.dir))
                        for idx in range(inputdata.shape[1]):
                            self.pcds[ipcd, idx] = model.layers[1].get_weights()[0][idx]
            else:

                #load ipcd models if file exists
                if isinstance(self.dir,str):
                    if self._load(ipcd, path=self.dir,filename="pcd_coop_obj"):
                        for idx in range(inputdata.shape[1]):
                            self.pcds[ipcd, idx] = self.models[ipcd].layers[2].get_weights()[0][idx]
                        continue


                # from second pcd to the end - freeze previous neurons and create a new neuron
                for i_init in range(trn_params.n_inits):
                    callback_list = []
                    # create the model (Keras - API functional)
                    # https://keras.io/getting-started/functional-api-guide/
                    # This returns a input tensor
                    inputs = Input(shape=(inputdata.shape[1],))

                    # add a non-linear freeze previous extracted pcd
                    freeze_layer = Dense(ipcd, activation=self.activation,trainable=False)(inputs)

                    # add a non-linear no-freeze new neuron
                    non_freeze_layer = Dense(1, activation=self.activation,trainable=True)(inputs)

                    #concatenate both, non-freeze and freeze layers
                    hidden_layer = concatenate([freeze_layer, non_freeze_layer])
                    outputs = Dense(targetdata_sparse.shape[1], activation='tanh')(hidden_layer)

                    model = Model(inputs=inputs, outputs=outputs)
                    freeze_weights = model.layers[1].get_weights()

                    for i_old_pcd in range(ipcd):
                        for idim in range(inputdata.shape[1]):
                            freeze_weights[0][idim,i_old_pcd] = self.pcds[i_old_pcd,idim]

                    model.layers[1].set_weights(freeze_weights)

                    # creating a optimization function using steepest gradient
                    sgd = SGD(lr=trn_params.learning_rate,
                              decay=trn_params.learning_decay,
                              momentum=trn_params.momentum,
                              nesterov=trn_params.nesterov)

                    adam = Adam(lr=trn_params.learning_rate, 
                                beta_1=0.9,  
                                beta_2=0.999, 
                                epsilon=None, 
                                decay=0.0, 
                                amsgrad=False)

                    # compile the model
                    model.compile(loss='mean_squared_error',
                                  optimizer=adam,
                                  metrics=['accuracy','mean_squared_error'])

                    if trn_params.monitor is 'sp':
                        trn_params.mode = 'max'
                    mt_sp = metricsAdd(monitor='sp', verbose=trn_params.train_verbose,verbose_train=trn_params.train_verbose)
                    callback_list.append(mt_sp)


                    # early stopping to avoid overtraining
                    earlyStopping = callbacks.EarlyStopping(
                        monitor=trn_params.monitor, patience=trn_params.patience,
                        verbose=trn_params.train_verbose, 
                        mode=trn_params.mode, 
                        restore_best_weights=True)

                    st = StopTraining(monitor=trn_params.monitor,restore_best_weights=True,
                                      verbose=trn_params.train_verbose,patience=trn_params.patience,
                                      min_delta=0)

                    callback_list.append(st)

                    csv_logger = callbacks.CSVLogger('%s/training.csv'%(self.dir))

                    callback_list.append(csv_logger)

                    if sample_weight is None:
                        class_weight = class_weight_Keras(targetdata[trainId])
                        sample_weight_fit = sample_weight
                    else:
                        class_weight = None
                        sample_weight_fit = sample_weight[trainId]
                    # Train model
                    init_trn_desc = model.fit(inputdata[trainId], targetdata_sparse[trainId],
                                              epochs=trn_params.n_epochs,
                                              batch_size=trn_params.batch_size,
                                              callbacks=callback_list,
                                              verbose=trn_params.train_verbose,
                                              validation_data=(inputdata[testId],
                                                               targetdata_sparse[testId]),
                                              class_weight=class_weight,
                                              sample_weight=sample_weight_fit,
                                              shuffle=True)

                    if trn_params.monitor is 'sp':
                        bool_trn = (np.max(init_trn_desc.history['sp']) > best_sp)
                    else:
                        bool_trn = (np.min(init_trn_desc.history['val_loss']) < best_loss)

                    if bool_trn:
                        #print('\n\nval_loss:',np.min(init_trn_desc.history['val_loss']), '- best_loss:', best_loss,'\n\n')
                        best_init = i_init
                        best_loss = np.min(init_trn_desc.history['val_loss'])
                        if trn_params.monitor is 'sp':
                            best_sp = np.max(init_trn_desc.history['sp'])
                        self.models[ipcd] = model

                        df = pd.read_csv('%s/training.csv'%(self.dir))
                        self.trn_descs[ipcd] = df
                        os.remove('%s/training.csv'%(self.dir))
                        for idx in range(inputdata.shape[1]):
                            self.pcds[ipcd, idx] = model.layers[2].get_weights()[0][idx]

        #K.clear_session()       

    def save(self, path=".",filename="pcd_coop_obj"):
        # save models
        for i in self.models:
            self.models[i].save('%s/%s_pcd_%i.h5'%(path,filename,i))
        # save trn_descs and comp
        for i in self.trn_descs:
            self.trn_descs[i].to_csv('%s/%s_trn_desc_pcd_%i.csv'%(path,filename,i),index=False)
            #joblib.dump([self.trn_descs[i].history],'%s/%s_trn_desc_pcd_%i.jbl'%(path,filename,i),compress=9)
        # save number of comp
        joblib.dump([self.n_components],'%s/%s_n_components.jbl'%(path,filename))
        return 0

    def load(self, path=".",filename="pcd_coop_obj"):
        # load number of comp
        self.n_components = joblib.load('%s/%s_n_components.jbl'%(path,filename))[0]
        # load pcds
        for i in range(self.n_components):
            self.models[i] = load_model('%s/%s_pcd_%i.h5'%(path,filename,i))
            self.trn_descs[i] = pd.read_csv('%s/%s_trn_desc_pcd_%i.csv'%(path,filename,i))
        return 0

    def _load(self, icomponent, path=".",filename="pcd_coop_obj"):

        i = icomponent
        file_pcd = '%s/%s_pcd_%i.h5'%(path,filename,i)

        if not os.path.exists(file_pcd):
            return False

        self.models[i] = load_model('%s/%s_pcd_%i.h5'%(path,filename,i))
        self.trn_descs[i] = pd.read_csv('%s/%s_trn_desc_pcd_%i.csv'%(path,filename,i))
        return True


class PCDCooperative(PCDCooperativeBase,BaseEstimator):
    """docstring for PCDCooperative"""
    def __init__(self, n_components=10, 
                activation='tanh', 
                dir=None,
                trn_params = None, 
                validation_id=(None,None),
                is_save=True):

        super(PCDCooperative, self).__init__(n_components=n_components, activation=activation, dir=dir)
        self.scaler_ = None
        self.validation_id = validation_id
        self.trn_params = trn_params
        self.is_save = is_save

    def fit(self, inputdata, targetdata, sample_weight=None):#, trn_tst_idx=None):

        self.scaler_ = StandardScaler()

        # for AdaBoostClassifier of Sklearn (without that, this classifier does not work)
        self.le_ = LabelEncoder().fit(targetdata)
        
        self.classes_ = self.le_.classes_

        if not self.validation_id[0] is None:
            trn_tst_idx = self.validation_id
            train_id, _ = self.validation_id
            self.scaler_.fit(inputdata[train_id])
        else:
            trn_tst_idx = None
            self.scaler_.fit(inputdata)

        fit_inputdata = self.scaler_.transform(inputdata)

        fit_base =  super(PCDCooperative, self).fit(inputdata=fit_inputdata, 
                                        targetdata=targetdata, 
                                        trn_params=self.trn_params, 
                                        trn_tst_idx=trn_tst_idx,
                                        sample_weight=sample_weight)
        if self.is_save:
            self.save(path=self.dir)

        return fit_base

    def predict(self, X ,y=None, predict='classes'):

        preproc_X = self.scaler_.transform(X)

        df_log = pd.concat(self.trn_descs)

        sp_pcd = df_log.groupby(level=0).agg(['max'])['sp'].values

        self.best_pcd = np.argmax(sp_pcd)
        self.best_sp = sp_pcd[np.argmax(sp_pcd)]

        if predict is 'sparce':
            pred = self.models[self.best_pcd].predict(preproc_X)
        else:
            pred = np.argmax(self.models[self.best_pcd].predict(preproc_X),axis=1)
            pd.Series(pred).to_csv(self.dir + '/predict.csv',index=False)

        return pred
        

class NLPCA(object):
	"""
	Non-Linear Principal Component Analysis class
		This class implement the Non-Linear Principal Component Analysis
	"""
	def __init__(self, n_components=2,n_neurons_encoder=2):
		"""
		NLPCA constructor
			n_components: number of components to be extracted
            n_neurons_encoder: number of neurons in encoder and decoder layers
		"""
		self.n_components = n_components
		self.n_neurons_encoder = n_neurons_encoder
		self.models = {}
		self.trn_descs = {}
		self.nlpcas = {}

	def fit(self, data, train_ids, test_ids, trn_params=None):
		"""
		NLPCA fit function
			data: data to be fitted (events x features)
			train_ids:  train indexes - user generated
			test_ids: test indexes - user generated
			trn_params: class TrnParams (optional)
		"""
		print 'NLPCA fit function'

		if trn_params is None:
			trn_params = TrnParams()

		#print 'Train Parameters'
		#trn_params.Print()

		if trn_params.verbose:
			print 'NLPCA Model Struct: %i - %i - %i - %i - %i'%(data.shape[1],self.n_neurons_encoder,
                                                      self.n_components,n_neurons_encoder,
                                                      data.shape[1])
		return self

	def save(self, path=".",filename="nlpca_obj"):
		# save models
		self.models.save('%s/%s.h5'%(path,filename))
		# save trn_descs and comp
		joblib.dump([self.trn_descs[i].history],'%s/%s_trn_desc_nlpca.jbl'%(path,filename),compress=9)
		# save number of comp
		joblib.dump([self.n_components, self.n_neurons_encoder],'%s/%s_n_components.jbl'%(path,filename))
		return 0

	def load(self, path=".",filename="pcd_indep_obj"):
		# load number of comp
		[self.n_components, self.n_neurons_encoder] = joblib.load('%s/%s_n_components.jbl'%(path,filename))
		# load models
		self.models = load_model('%s/%s.h5'%(path,filename))
		return 0