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

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.base import MetaEstimatorMixin

from hep_ml.nnet import MLPMultiClassifier,_prepare_scaler


## MLP of https://arogozhnikov.github.io/hep_ml/_modules/hep_ml/nnet.html#AbstractNeuralNetworkClassifier
# class MLPhep(MLPMultiClassifier):
#     def __init__(self,layers=(10,), scaler='standard', trainer='irprop-', epochs=100,
#                  trainer_parameters=None, random_state=None):
#                  MLPMultiClassifier.__init__(self,layers=(10,), scaler='standard', trainer='irprop-', epochs=100,
#                               trainer_parameters=None, random_state=None)
#
#
#     def fit(self,X,y,sample_weight=None):
#
#             X, y,_ = self._prepare_inputs(X, y, sample_weight=sample_weight)
#
#             for init in range(1):
#                     print "[+] {0} of {1} inits".format(init+1,1)
#                     model = Sequential()
#
#                     model.add(Dense(8, input_dim=4, activation='relu'))
#
#                     model.add(Dense(3, activation='softmax'))
#
#                     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#
#                     model.fit(X,y,epochs=10,sample_weight=sample_weight,verbose=1)
#
#                     self.model = model
#
#             return self
#
#     def predict(self,X,y=None):
#         preproc_X = self._transform(X)
#         return self.model.predict_classes(preproc_X)


class MLPKeras(BaseEstimator, ClassifierMixin):
    """docstring for MLPKeras."""
    def __init__(self):
        self.model = None
        #set_random_seed(random_state)

            #super(MLPKeras, self).__init__()
            #self.trnparams = TrnParams
            #information = {}


    def _transform(self, X, y=None, fit=False):
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
            self.scaler_.fit(X, y)

        result = self.scaler_.transform(X)
        #result = numpy.hstack([result, numpy.ones([len(X), 1])]).astype('float32')

        return result


    def _classes(self,y):
        le = LabelEncoder()
        le.fit(y)
        return le.classes_

    def _preprocess(self,X,y=None,fit=False):
        """Prepare data for training and test

        :param numpy.array X: of shape [n_samples, n_feactures]
        :param numpy.array y: of shape [n_samples]
        :return: list of []
        """

        if not y is None:
            ohe = OneHotEncoder(sparse=False)
            sparce_y = ohe.fit_transform(pd.DataFrame(y,columns=['target']))
        else:
            return self._transform(X,y,fit)
        return self._transform(X,y,fit),sparce_y

    def fit(self,X,y,sample_weight=None):

            self.classes_ = self._classes(y)

            preproc_X,sparce_y = self._preprocess(X,y,fit=True)

            for init in range(1):
                    print "[+] {0} of {1} inits".format(init+1,1)
                    model = Sequential()

                    model.add(Dense(40, input_dim=X.shape[1], activation='tanh'))

                    model.add(Dense(len(self.classes_), activation='softmax'))

                    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

                    model.fit(preproc_X,sparce_y,
                              epochs=20,
                              sample_weight=sample_weight,
                              verbose=1,
                              shuffle=True)

                    self.model = model

            return self

    def predict(self,X,y=None):
        preproc_X = self._preprocess(X)
        return self.model.predict_classes(X)

    def predict_proba(self,X):
        preproc_X = self._preprocess(X)
        return self.model.predict_proba(preproc_X)

    def score(self,X,y=None):
        preproc_X = self._preprocess(X)
        return self.model.evaluate(preproc_X,y)


class MLPSKlearn(MLPKeras):
    def __init__(self,
                  hidden_layer_sizes=(100,),
                  activation="relu",
                  solver='adam',
                  alpha=0.0001,
                  batch_size='auto',
                  learning_rate="constant",
                  learning_rate_init=0.001,
                  power_t=0.5,
                  max_iter=200,
                  shuffle=True,
#                  random_state=None,
                  tol=1e-4,
                  verbose=False,
                  warm_start=False,
                  momentum=0.9,
                  nesterovs_momentum=True,
                  early_stopping=False,
                  validation_fraction=0.1,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-8,
                  n_iter_no_change=10,
                  dir='/.'):
                  super(MLPSKlearn,self).__init__()


#     def __init__(self,
#                  hidden_layer_sizes=(10,),
#                  activations=("relu",),
#                  solver="adam",
#                  batch_size=32,
#                  epochs=200,
#                  loss="categorical_crossentropy",
#                  metrics=["acc"],
#                  input_shape=(None,),
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  momentum=0.9,
#                  nesterov=True,
#                  decay=0.0,
#                  beta_1=0.9,
#                  beta_2=0.999,
#                  epsilon=1e-08,
#                  learning_rate=0.001,
#                  amsgrad=False,
#                  early_stopping=False,
#                  es_kwargs=None,
#                  model_checkpoint=True,
#                  save_best=True,
#                  mc_kwargs=None,
#                  log_history=True,
#                  cachedir='./'):
#
#         self.cachedir = cachedir
#         args, _, _, values = inspect.getargvalues(inspect.currentframe())
#         values.pop("self")
#
#         for arg, val in values.items():
#             setattr(self, arg, val)
#
#
#     def _buildParameters(self):
#
#         if self.es_kwargs is None:
#             es_kwargs = {"monitor": 'val_loss',
#                          "min_delta": 0,
#                          "patience": 10,
#                          "verbose": 0,
#                          "mode": 'auto',
#                          "baseline": None,
#                          "restore_best_weights": False}
#         else:
#             tmp_kwargs = {"monitor": 'val_loss',
#                           "min_delta": 0,
#                           "patience": 10,
#                           "verbose": 0,
#                           "mode": 'auto',
#                           "baseline": None,
#                           "restore_best_weights": False}
#
#             for key in self.es_kwargs.keys():
#                 tmp_kwargs[key] = self.es_kwargs[key]
#             es_kwargs = tmp_kwargs
#
#         if self.mc_kwargs is None:
#             mc_kwargs = {"monitor": 'val_loss',
#                          "verbose": 0,
#                          "save_weights_only": False,
#                          "mode": 'auto',
#                          "period": 1,
#                          "save_best": self.save_best}
#         else:
#             tmp_kwargs = {"monitor": 'val_loss',
#                          "verbose": 0,
#                          "save_weights_only": False,
#                          "mode": 'auto',
#                          "period": 1,
#                          "save_best": self.save_best}
#             for key in self.mc_kwargs.keys():
#                 tmp_kwargs[key] = self.mc_kwargs[key]
#             mc_kwargs = tmp_kwargs
#
#         layers = [self.build_layer(units,
#                                    activation,
#                                    self.kernel_initializer,
#                                    self.bias_initializer,
#                                    self.kernel_regularizer,
#                                    self.bias_regularizer,
#                                    self.activity_regularizer,
#                                    self.kernel_constraint,
#                                    self.bias_constraint)
#                   for units, activation in zip(self.hidden_layer_sizes, self.activations)]
#         layers[0] = self.build_layer(self.hidden_layer_sizes[0],
#                                      self.activations[0],
#                                      self.kernel_initializer,
#                                      self.bias_initializer,
#                                      self.kernel_regularizer,
#                                      self.bias_regularizer,
#                                      self.activity_regularizer,
#                                      self.kernel_constraint,
#                                      self.bias_constraint,
#                                      input_shape=self.input_shape)
#
#
#         optimizer = self.build_optimizer(self.solver,
#                                          self.momentum,
#                                          self.nesterov,
#                                          self.decay,
#                                          self.learning_rate,
#                                          self.amsgrad,
#                                          self.beta_1,
#                                          self.beta_2,
#                                          self.epsilon)
#
#         trnParams = CNNParams(prefix='mlp',
#                               optimizer=optimizer,
#                               layers=layers,
#                               loss=self.loss,
#                               metrics=self.metrics,
#                               epochs=self.epochs,
#                               batch_size=self.batch_size,
#                               callbacks=None,
#                               input_shape=self.input_shape)
#
#         filepath = os.path.join(self.cachedir, trnParams.getParamPath())
#
#         mc_kwargs["filepath"] = filepath
#         callbacks = []
#         if self.early_stopping:
#             callbacks.append(self.build_early_stopping(**es_kwargs))
#         if self.model_checkpoint:
#             m_check, best_model = self.build_model_checkpoint(**mc_kwargs)
#             callbacks.append(m_check)
#             if best_model is not None:
#                 callbacks.append(best_model)
#         if self.log_history:
#             csvlog = {"type": "CSVLogger", "filename": os.path.join(filepath, 'history.csv')}
#             callbacks.append(csvlog)
#
#         callbacks_list = TrainParameters.Callbacks()
#         callbacks_list.add(callbacks)
#         return trnParams, callbacks_list, filepath
#
#     def plotTraining(self, ax=None,
#                      train_scores='all',
#                      val_scores='all',
#                      savepath=None):
#         if ax is None:
#             sns.set_style("whitegrid")
#             fig, ax = plt.subplots(1,1)
#             loss_ax = ax
#             score_ax = plt.twinx(ax)
#         history = self.history
#
#         if val_scores is 'all':
#             val_re = re.compile('val_')
#             val_scores = map(lambda x: x.string,
#                              filter(lambda x: x is not None,
#                                     map(val_re.search, history.columns.values)))
#
#         if train_scores is 'all':
#             train_scores = history.columns.values[1:] # Remove 'epoch' column
#             train_scores = train_scores[~np.isin(train_scores, val_scores)]
#
#         x = history['epoch']
#         linestyles = ['-', '--', '-.', ':']
#         ls_train = cycle(linestyles)
#         ls_val = cycle(linestyles)
#
#         loss_finder = re.compile('loss')
#         for train_score, val_score in zip(train_scores, val_scores):
#             if loss_finder.search(train_score) is None:
#                 score_ax.plot(x, history[train_score], color="blue", linestyle=ls_train.next(), label=train_score)
#             else:
#                 loss_ax.plot(x, history[train_score], color="blue", linestyle=ls_train.next(), label=train_score)
#
#             if loss_finder.search(val_score) is None:
#                 score_ax.plot(x, history[val_score], color="red", linestyle=ls_val.next(), label=val_score)
#             else:
#                 loss_ax.plot(x, history[val_score], color="red", linestyle=ls_val.next(), label=val_score)
#
#         ax.legend()
#         plt.show()
#
#     def fit(self,
#             X,
#             y,
#             n_inits = 1,
#             validation_split=0.0,
#             validation__data=None,
#             shuffle=True,
#             verbose=0,
#             class_weight=True,
#             sample_weight=None,
#             steps_per_epoch=None,
#             validation_steps=None):
#
#         if class_weight:
#             class_weights = self._getGradientWeights(y)
#         else:
#             class_weights = None
#
#         if n_inits < 1:
#             warnings.warn("Number of initializations must be at least one."
#                           "Falling back to one")
#             n_inits = 1
#
#         print self.input_shape
#
#         trnParams, callbacks, filepath = self._buildParameters()
#         keras_callbacks = #callbacks.toKerasFn()
#
#         model = SequentialModelWrapper(trnParams,
#                                        results_path=filepath)
#         best_weigths_path = os.path.join(filepath, 'best_weights')
#
#         if exists(best_weigths_path):
#             print "Model trained, loading best weights"
#             model.build_model()
#         else:
#             for init in range(n_inits):
#                 # for callback in keras_callbacks:
#                 #     if isinstance(callback, ModelCheckpoint):
#                 #         print callback.best
#
#                 model.build_model()
#
#                 # print model.model.optimizer.weights
#                 before = model.model.get_weights()
#                 model.fit(x=X,
#                           y=y,
#                           batch_size=self.batch_size,
#                           epochs=self.epochs,
#                           verbose=verbose,
#                           callbacks=keras_callbacks,
#                           validation_split=validation_split,
#                           validation_data=validation__data,
#                           shuffle=shuffle,
#                           class_weight=class_weights,
#                           sample_weight=sample_weight,
#                           initial_epoch=0,
#                           steps_per_epoch=steps_per_epoch,
#                           validation_steps=validation_steps)
#
#             import keras.backend as K
#             K.clear_session()
#                 # print "After Training"
#                 # print model.model.get_weights()
#                 # print before == model.model.optimizer.weights
#                 # print model.model.optimizer.weights
#
#         model.load_weights(best_weigths_path)
#         self.history = pd.read_csv(os.path.join(filepath, 'history.csv'))
#         self.model = model
#         return self
#
#     def predict(self, X):
#         return self.model.predict(X)
#
#     def score(self, X, y, sample_weight=None):
#         if y.ndim > 1:
#             y = y.argmax(axis=1)
#
#         out = self.predict(X)
#
#         cat_out = out.argmax(axis=1)
#         return spIndex(recall_score(y, cat_out),
#                        n_classes=len(np.unique(y)))
#
#     @staticmethod
#     def _getGradientWeights(y_train, mode='standard'):
#         if y_train.ndim > 1:
#             y_train = y_train.argmax(axis=1)
#
#         cls_indices, event_count = np.unique(np.array(y_train), return_counts=True)
#         min_class = min(event_count)
#
#         return {cls_index: float(min_class) / cls_count
#                 for cls_index, cls_count in zip(cls_indices, event_count)}
#
#     @staticmethod
#     def build_model_checkpoint(filepath,
#                                monitor='val_loss',
#                                verbose=0,
#                                save_weights_only=False,
#                                mode='auto',
#                                period=1,
#                                save_best=True):
#
#         m_filepath = os.path.join(filepath, 'end_weights')
#         b_filepath = os.path.join(filepath, 'best_weights')
#
#         m_check =  {"type": "ModelCheckpoint",
#                     "filepath": m_filepath,
#                     "monitor": monitor,
#                     "verbose": verbose,
#                     "save_weights_only": save_weights_only,
#                     "mode": mode,
#                     "period": period}
#         if save_best:
#             best_check = {"type": "ModelCheckpoint",
#                           "filepath": b_filepath,
#                           "monitor": monitor,
#                           "verbose" : verbose,
#                           "save_weights_only": save_weights_only,
#                           "mode" : mode,
#                           "period" : period,
#                           "save_best_only":True}
#         else:
#             best_check = None
#
#         return m_check, best_check
#
#     @staticmethod
#     def build_early_stopping(monitor,
#                              min_delta,
#                              patience,
#                              verbose,
#                              mode,
#                              baseline,
#                              restore_best_weights):
#         return  {"type": "EarlyStopping",
#                  "monitor":monitor,
#                  "min_delta":min_delta,
#                  "patience":patience,
#                  "verbose":verbose,
#                  "mode":mode,
#                  "baseline":baseline,
#                  "restore_best_weights":restore_best_weights}
#
#     @staticmethod
#     def build_layer(units,
#                     activation,
#                     kernel_initializer,
#                     bias_initializer,
#                     kernel_regularizer,
#                     bias_regularizer,
#                     activity_regularizer,
#                     kernel_constraint,
#                     bias_constraint,
#                     input_shape=None):
#
#         layer = {"type": "Dense",
#                  "units": units,
#                  "kernel_initializer": kernel_initializer,
#                  "bias_initializer": bias_initializer,
#                  "kernel_regularizer": kernel_regularizer,
#                  "bias_regularizer": bias_regularizer,
#                  "activity_regularizer": activity_regularizer,
#                  "kernel_constraint": kernel_constraint,
#                  "bias_constraint": bias_constraint}
#
#         if input_shape is not None:
#             layer["input_shape"] = input_shape
#
#         if activation != "":
#             layer["activation"] = activation
#         return layer
#
#     @staticmethod
#     def build_optimizer(solver,
#                         momentum=0.9,
#                         nesterov=True,
#                         decay=0.0,
#                         learning_rate=0.001,
#                         amsgrad=False,
#                         beta_1=0.9,
#                         beta_2=0.999,
#                         epsilon=1e-08):
#         solver = solver.lower()
#
#         optimizer = {}
#         if solver not in ['sgd', 'adam']:
#             raise NotImplementedError
#
#         if solver == 'sgd':
#             optimizer = {"type": "SGD",
#                          "momentum": momentum,
#                          "decay": decay,
#                          "nesterov": nesterov}
#
#         elif solver == 'adam':
#             optimizer = {"type": "Adam",
#                          "lr": learning_rate,
#                          "beta_1": beta_1,
#                          "beta_2": beta_2,
#                          "epsilon": epsilon,
#                          "decay": decay,
#                          "amsgrad": amsgrad}
#
#         return optimizer
#
#
#
