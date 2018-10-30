import inspect
import os
import re
import time
import warnings
import pandas as pd
#import matplotlib.pyplot as plt
from abc import abstractmethod
from itertools import product, cycle
from warnings import warn
#import seaborn as sns

import keras
import numpy as np
from keras import Sequential, Model
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from tensorflow import set_random_seed

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection._search import BaseSearchCV, ParameterGrid, _check_param_grid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder

from Functions import TrainParameters
from Functions.util import check_mount_dict,get_objects

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.base import MetaEstimatorMixin

from hep_ml.nnet import MLPMultiClassifier,_prepare_scaler




class MLPKeras(BaseEstimator, ClassifierMixin):
    """docstring for MLPKeras."""
    def __init__(self,
                  hidden_layer_sizes=(100,),
                  activation=('tanh','softmax'),
                  solver='adam',
                  op_adam_kwargs=None,
                  loss='mean_squared_error',
                  max_init=1,
                  batch_size=None,
                  max_iter=200,
                  shuffle=True,
                  random_state=None,
                  verbose=False,
                  early_stopping = False,
                  es_kwargs = None,
                  save_best_model = False,
                  mc_kwargs = None,
                  metrics=['acc'],
                  validation_fraction=0.0,
                  dir='/.'):

        if len(hidden_layer_sizes) != 1:
            raise ValueError('only one hidden layer implemented')

        if not solver in ['adam']:
            raise ValueError('choose \'adam\' as optimizer ')

        self.model = None

        if batch_size is 'auto':
            batch_size = None

        nn_params = {
                    'hidden_layer_sizes':hidden_layer_sizes,
                    'activation':activation,
                    'optimize':solver,
                    'batch_size':batch_size,
                    'epoch':max_iter,
                    'verbose':verbose,
                    'n_init':max_init,
                    'loss':loss,
                    'metrics':metrics
        }

        callback_params = {
                           'early_stopping':early_stopping,
                           'save_best_model':save_best_model
        }

        def_op_adam_kwargs = {'lr':0.001,
                          'beta_1':0.9,
                          'beta_2':0.999,
                          'epsilon':None,
                          'decay':0.0,
                          'amsgrad':False}


        def_es_kwargs = {'monitor':'val_loss',
                     'min_delta':0,
                     'patience':0,
                     'verbose':0,
                     'mode':'auto',
                     'baseline':None,
                     'restore_best_weights':False}

        def_mc_kwargs = {'filepath':dir,
                     'monitor':'val_loss',
                     'verbose':0,
                     'save_best_only':True,
                     'save_weights_only':False,
                     'mode':'auto',
                     'period':1}

        add_params = {
                     'shuffle':shuffle,
                     'random_state':random_state,
                     'validation_fraction':validation_fraction,
                     'dir':dir
        }


        all_params = {}
        all_params.update(nn_params)
        all_params.update(add_params)
        all_params.update(callback_params)

        for  keys,values in all_params.items():
            self.__dict__[keys] = values

        self.op_adam_kwargs = check_mount_dict(def_op_adam_kwargs,op_adam_kwargs)
        self.es_kwargs = check_mount_dict(def_es_kwargs,es_kwargs)
        self.mc_kwargs = check_mount_dict(def_mc_kwargs,mc_kwargs)



    def _transform(self, X, y=None, fit=False,train_id=None):
        """Apply selected scaler or transformer to dataset
        (also this method adds a column filled with ones).

        :param numpy.array X: of shape [n_samples, n_features], data
        :param numpy.array y: of shape [n_samples], labels
        :param bool fit: if True, fits transformer
        :return: transformed data, numpy.array of shape [n_samples, n_output_features]
        """
        # Fighting copy-bug of sklearn's transformers
        X = np.array(X, dtype=float)

        if fit:
            self.scaler_ = _prepare_scaler('standard')
            if not train_id is None:
                self.scaler_.fit(X[train_id], y)
            else:
                self.scaler_.fit(X, y)

        result = self.scaler_.transform(X)
        #result = numpy.hstack([result, numpy.ones([len(X), 1])]).astype('float32')

        return result


    def _classes(self,y):
        le = LabelEncoder()
        le.fit(y)
        return le.classes_

    def _preprocess(self,X,y=None,fit=False,train_id=None):
        """Prepare data for training and test

        :param numpy.array X: of shape [n_samples, n_feactures]
        :param numpy.array y: of shape [n_samples]
        :return: list of []
        """

        if not y is None:
            ohe = OneHotEncoder(sparse=False)
            sparce_y = ohe.fit_transform(pd.DataFrame(y,columns=['target']))
        else:
            return self._transform(X,y,fit,train_id)
        return self._transform(X,y,fit,train_id),sparce_y

    def fit(self,X,y,train_id=None,test_id=None,sample_weight=None):

            self.classes_ = self._classes(y)

            preproc_X,sparce_y = self._preprocess(X,y,fit=True,train_id=train_id)

            for init in range(1):

                    callbacks_list = []

                    print "[+] {0} of {1} inits".format(init+1,1)
                    model = Sequential()

                    model.add(Dense(self.hidden_layer_sizes[0], input_dim=X.shape[1], activation=self.activation[0]))

                    model.add(Dense(len(self.classes_), activation=self.activation[1]))

                    opt = getattr(keras.optimizers,'Adam')

                    model.compile(loss=self.loss, optimizer=opt(**self.op_adam_kwargs), metrics=self.metrics)

                    if self.early_stopping:
                        callbacks_list.append(get_objects(keras.callbacks,
                                                          'EarlyStopping',
                                                          self.es_kwargs))

                    if self.save_best_model:
                        callbacks_list.append(get_objects(keras.callbacks,
                                                         'ModelCheckpoint',
                                                         self.mc_kwargs))


                    if train_id is None:
                        validation_data = None
                        x_data = preproc_X
                        y_sparse = sparce_y
                    else:
                        validation_data = (preproc_X[test_id],sparce_y[test_id])
                        x_data = preproc_X[train_id]
                        y_sparse = sparce_y[train_id]

                    model.fit(x_data,y_sparse,
                              epochs=self.epoch,
                              batch_size=self.batch_size,
                              callbacks=callbacks_list,
                              validation_split=self.validation_fraction,
                              validation_data=validation_data,
                              sample_weight=sample_weight,
                              verbose=self.verbose,
                              shuffle=self.shuffle)

                    self.model = model

            return self

    def predict(self,X,y=None):

        if self.model is None:
            raise Exception('use \'fit\' function first')

        preproc_X = self._preprocess(X)
        return self.model.predict_classes(X)

    def predict_proba(self,X):

        if self.model is None:
            raise Exception('use \'fit\' function first')

        preproc_X = self._preprocess(X)
        return self.model.predict_proba(preproc_X)

    def score(self,X,y=None):

        if self.model is None:
            raise Exception('use \'fit\' function first')

        preproc_X = self._preprocess(X)
        return self.model.evaluate(preproc_X,y)


class MLPSKlearn(MLPKeras):
    def __init__(self,
                  hidden_layer_sizes=(100,),
                  activation=('tanh','softmax'),
                  solver='adam',
                  op_adam_kwargs=None,
                  loss='mean_squared_error',
                  max_init=1,
                  batch_size=None,
                  max_iter=200,
                  shuffle=True,
                  random_state=None,
                  verbose=False,
                  early_stopping = False,
                  es_kwargs = None,
                  save_best_model = False,
                  mc_kwargs = None,
                  metrics=['acc'],
                  validation_fraction=0.0,
                  dir='/.'):

        sup = super(MLPSKlearn,self)
        sup.__init__(hidden_layer_sizes=hidden_layer_sizes,
                     activation=activation,solver=solver,op_adam_kwargs=op_adam_kwargs,
                     loss=loss,
                     batch_size=batch_size,max_iter=max_iter,
                     shuffle=shuffle,
                     random_state=random_state,verbose=verbose,
                     early_stopping=early_stopping,es_kwargs=es_kwargs,
                     save_best_model=save_best_model,mc_kwargs=mc_kwargs,
                     validation_fraction=validation_fraction,metrics=metrics,
                     dir=dir)
