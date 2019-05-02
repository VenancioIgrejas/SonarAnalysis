import os
import pandas as pd
from shutil import copyfile

import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import StandardScaler

from Functions.util import file_exist

from sklearn.base import BaseEstimator
from keras.utils import np_utils

import keras

from keras import Sequential
from keras.layers import Dense
from keras.models import load_model


def _check_sparce(y_parse, output_activation):
    if not output_activation in ['tanh']:
        if sum(np.unique(y_parse)) in [1]:
            return y_parse
        else:
            raise ValueError("target sparse need contain only 0 and 1")

    # return np.vstack(map(lambda x:2*x-1,y_parse)) # transform y_sparce in 1 and -1 target
    return 2 * y_parse - 1


def attr_update(package, name, config_dict={}):
    obj = getattr(package, name)
    tmp = obj()
    _dict = tmp.get_config()
    _dict.update(config_dict)
    return getattr(package, name)(**_dict)


def callback(attribute, **kwargs):
    return attribute(**kwargs)


class Preprocessing(object):
    """docstring for Preprocessing"""

    def __init__(self, class_weight=True, func_transfort=None, limits='tanh'):
        self.limits = limits
        self.class_weight = class_weight
        self.class_weight_value = None
        self._flag_fit = False
        if func_transfort is None:
            self.func_transfort = StandardScaler()

    def get_weights(self):
        return self.class_weight_value

    def get_transform(self):
        return self.transform

    def get_sparce(self):
        return self.y_categorical

    def get_samples(self):
        pass

    def set_class_weight(self, y, class_weight='balanced'):

        if self.class_weight:
            self.class_weight_value = dict(zip(
                np.unique(y), compute_class_weight(
                    class_weight=class_weight, classes=np.unique(y), y=y)))

        return self

    def set_transform(self, X, y=None, fit=False):
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
            self.func_transfort = StandardScaler()
            self.func_transfort.fit(X, y)

        self.transform = self.func_transfort.transform(X)

        self._flag_fit = True

        return self

    def set_sparce(self, y):

        if self.limits == 'tanh':
            self.y_categorical = 2 * np_utils.to_categorical(y) - 1
        else:
            self.y_categorical = np_utils.to_categorical(y)

        return self


class MLPKeras(BaseEstimator, ClassifierMixin):
    """docstring for MLPKeras."""

    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation=('tanh', 'softmax'),
                 optimizer='adam',
                 verbose=True,
                 optimizer_kwargs={},
                 loss='mean_squared_error',
                 metrics=['acc'],
                 compile_kwargs={},
                 n_init=1,
                 fit_kwargs={},
                 validation_id=None,
                 callbacks_list=[],
                 dir='./'):

        if len(hidden_layer_sizes) != 1:
            raise ValueError('only one hidden layer implemented')

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.loss = loss
        self.metrics = metrics
        self.compile_kwargs = compile_kwargs
        self.n_init = n_init
        self.fit_kwargs = fit_kwargs
        self.callbacks_list = callbacks_list
        self.dir = dir
        self.validation_id = validation_id
        self.verbose = verbose

        self.model = None

    def load_train(self):
        return pd.read_csv(os.path.join(self.get_params()['dir'], 'log_train.csv'))

    def fit(self, X, y, sample_weight=None):

        filepath_bestmodel = str(os.path.join(self.get_params()['dir'], 'best_model.h5'))

        if file_exist(filepath_bestmodel):
            print("model already exists in {0} file".format(filepath_bestmodel))
            self.model = load_model(str(filepath_bestmodel))
            return self

        best_loss = np.inf
        best_sp = -np.inf
        flag_csvlogger = False
        flag_modelcheckpoint = False

        for init in range(self.n_init):

            print("[+] {0} of {1} inits".format(init + 1, self.n_init))
            model = Sequential()

            model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='relu'))

            model.add(Dense(self.hidden_layer_sizes[0], activation=self.activation[0]))

            # model.add(Dense(self.hidden_layer_sizes[0], input_dim=X.shape[1], activation=self.activation[0]))

            model.add(Dense(y.shape[1], activation=self.activation[1]))

            opt = attr_update(keras.optimizers, self.optimizer, self.optimizer_kwargs)

            model.compile(loss=self.loss, optimizer=opt, metrics=self.metrics, **self.compile_kwargs)

            for icallback in self.callbacks_list:
                if isinstance(icallback, keras.callbacks.CSVLogger):
                    flag_csvlogger = True
                    icallback.filename = os.path.join(self.get_params()['dir'], 'log_train_init_{0}.csv'.format(init))

                if isinstance(icallback, keras.callbacks.ModelCheckpoint):
                    flag_modelcheckpoint = True
                    icallback.filepath = str(os.path.join(self.get_params()['dir'], 'best_model.h5'))

            if not self.callbacks_list:
                self.callbacks_list = None

            if self.validation_id != None:
                train_id, test_id = self.validation_id
                self.fit_kwargs.update({'x': X[train_id], 'y': y[train_id], 'callbacks': self.callbacks_list,
                                        'validation_data': (X[test_id], y[test_id])})
            else:
                self.fit_kwargs.update({'x': X, 'y': y, 'callbacks': self.callbacks_list})

            init_trn_desc = model.fit(**self.fit_kwargs)

            # print model_files
            if np.min(init_trn_desc.history['val_loss']) < best_loss:
                best_init = init
                best_model = model

        if flag_csvlogger:
            copyfile(os.path.join(self.get_params()['dir'], 'log_train_init_{0}.csv'.format(best_init)),
                     os.path.join(self.get_params()['dir'], 'log_best_from_{0}.csv'.format(best_init)))

        if flag_modelcheckpoint:
            self.model = load_model(str(os.path.join(self.get_params()['dir'], 'best_model.h5')))
        else:
            self.model = best_model

        return self

    def predict(self, x):
        return self.model.predict(x)
