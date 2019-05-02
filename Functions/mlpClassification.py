import inspect
import os
from shutil import copyfile
import re
import time
import warnings
import pandas as pd
#import matplotlib.pyplot as plt
from abc import abstractmethod
from itertools import product, cycle
from warnings import warn
#import seaborn as sns


#import keras
import numpy as np


#from keras import Sequential, Model
#from keras.layers import Dense, Activation
#from keras.callbacks import ModelCheckpoint
#from keras.models import load_model
#from keras.utils import np_utils

#from tensorflow import set_random_seed

from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection._search import BaseSearchCV, ParameterGrid, _check_param_grid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder

from Functions import TrainParameters
from Functions.callbackKeras import metricsAdd, StopTraining
from Functions.util import check_mount_dict,get_objects,update_paramns,file_exist,best_file

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.base import MetaEstimatorMixin

from lps_toolbox.metrics.classification import sp_index






def _check_sparce(y_parse,output_activation):
    if not output_activation in ['tanh']:
        if  sum(np.unique(y_parse)) in [1]:
            return y_parse
        else:
            raise ValueError("target sparse need contain only 0 and 1")

    #return np.vstack(map(lambda x:2*x-1,y_parse)) # transform y_sparce in 1 and -1 target
    return 2*y_parse - 1



class MLPKeras(BaseEstimator, ClassifierMixin):
    """docstring for MLPKeras."""
    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation=('tanh','softmax'),
                 optimize='adam',
                 lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,
                 loss='mean_squared_error',
                 n_init=1,
                 batch_size=None,
                 epoch=200,
                 shuffle=True,
                 random_state=None,
                 verbose=0,
                 train_log = False,
                 append=False,
                 class_weight = False,
                 early_stopping = False,
                 monitor='val_loss', min_delta=0.0001, patience=0,
                 mode='auto', baseline=None, restore_best_weights=True,
                 save_best_model = False,
                 save_weights_only=False, period=1,
                 metrics=['acc'],
                 validation_id = (None,None),
                 validation_fraction=0.0,
                 dir='./'):

        if len(hidden_layer_sizes) != 1:
            raise ValueError('only one hidden layer implemented')

        if not optimize in ['adam']:
            raise ValueError('choose \'adam\' as optimizer ')

        if batch_size is 'auto':
            batch_size = None




        nn_params = {
                    'hidden_layer_sizes':hidden_layer_sizes,
                    'activation':activation,
                    'optimize':optimize,
                    'batch_size':batch_size,
                    'epoch':epoch,
                    'verbose':verbose,
                    'n_init':n_init,
                    'loss':loss,
                    'metrics':metrics
        }

        callback_params = {
                           'early_stopping':early_stopping,
                           'save_best_model':save_best_model,
                           'train_log':train_log,
        }

        op_adam_kwargs = {'lr':lr,
                          'beta_1':beta_1,
                          'beta_2':beta_2,
                          'epsilon':epsilon,
                          'decay':decay,
                          'amsgrad':amsgrad}


        es_kwargs = {'monitor':monitor,
                     'min_delta':min_delta,
                     'patience':patience,
                     'verbose':verbose,
                     'mode':mode,
                     'baseline':baseline,
                     'restore_best_weights':restore_best_weights}

        mc_kwargs = {'filepath':dir+'model.h5',
                     'monitor':monitor,
                     'verbose':verbose,
                     'save_best_only':True,
                     'save_weights_only':save_weights_only,
                     'mode':mode,
                     'period':period}

        lg_kwargs = {'filename':dir+'log_train.csv',
                     'separator':',',
                     'append':append}


        add_params = {'value_class_weight':None,
                     'model':None,
                     'shuffle':shuffle,
                     'class_weight':class_weight,
                     'random_state':random_state,
                     'validation_id':validation_id,
                     'validation_fraction':validation_fraction,
                     'dir':dir
        }


        all_params = []
        all_params.append(nn_params)
        all_params.append(add_params)
        all_params.append(callback_params)
        all_params.append(op_adam_kwargs)
        all_params.append(es_kwargs)
        all_params.append(mc_kwargs)
        all_params.append(lg_kwargs)

        self.para = all_params
        list_tmp = []
        for each_dict in all_params:
            list_tmp = list_tmp + each_dict.items()

        self.__dict__ = dict(list_tmp)

        #self.op_adam_kwargs = check_mount_dict(def_op_adam_kwargs,op_adam_kwargs)
        #self.es_kwargs = check_mount_dict(def_es_kwargs,es_kwargs)
        #self.mc_kwargs = check_mount_dict(def_mc_kwargs,mc_kwargs)

        self.op_adam_kwargs = op_adam_kwargs
        self.es_kwargs = es_kwargs
        self.mc_kwargs = mc_kwargs
        self.lg_kwargs = lg_kwargs

        self._y = None



    def _check_pararms_change(self):
        exception = []
        dic = self.get_params()
        self.mc_kwargs = update_paramns(self.mc_kwargs,dic,exception)
        self.es_kwargs = update_paramns(self.es_kwargs,dic,exception)
        self.op_adam_kwargs = update_paramns(self.op_adam_kwargs,dic,exception)
        self.__dict__ = update_paramns(self.__dict__,dic,exception)

        return None

    def set_class_weight(self,function,y,**kwarg):
        self.value_class_weight = function(y,**kwarg)
        return self

    def _class_weight(self,y,class_weight='balanced'):

        if self.class_weight:
            return dict(zip(
                           np.unique(y),compute_class_weight(
                           class_weight=class_weight,classes=np.unique(y),y=y)))
        
        return self.value_class_weight

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
            self.scaler_ = StandardScaler()
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
        
        from keras.utils import np_utils

        if not y is None:
            #ohe = OneHotEncoder(sparse=False)#,categories='auto')
            #sparce_y = ohe.fit_transform(pd.DataFrame(y,columns=['target']))
            sparce_y = np_utils.to_categorical(y)

        else:
            return self._transform(X,y,fit,train_id)
        return self._transform(X,y,fit,train_id),sparce_y

    def load_train(self):
        return pd.read_csv(os.path.join(self.get_params()['dir'],'log_train.csv'))

    def fit(self, X, y, sample_weight=None):
            import keras

            from keras import Sequential, Model
            from keras.layers import Dense, Activation
            from keras.callbacks import ModelCheckpoint
            from keras.models import load_model
            from keras.utils import np_utils
            
            from tensorflow import set_random_seed


            train_id, test_id = self.validation_id

            self.classes_ = self._classes(y[train_id])

            self._check_pararms_change()

            #preproc_X,sparce_y = self._preprocess(X,y,fit=True,train_id=train_id)

            self.scaler_ = StandardScaler()
            self.scaler_.fit(X[train_id], y)


            

            

            self.mc_kwargs['filepath'] = os.path.join(self.get_params()['dir'],'best_model.h5')

            if file_exist(self.mc_kwargs['filepath']):
                print("model already exists in {0} file".format(self.mc_kwargs['filepath']))
                self.model = load_model(self.mc_kwargs['filepath'])
                return self


            best_loss = np.inf
            best_sp = -np.inf
            model_files={}
            log_files={}
            for init in range(self.n_init):

                    callbacks_list = []

                    print("[+] {0} of {1} inits".format(init+1,self.n_init))
                    model = Sequential()

                    model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='relu'))

                    model.add(Dense(self.hidden_layer_sizes[0], activation=self.activation[0]))

                    #model.add(Dense(self.hidden_layer_sizes[0], input_dim=X.shape[1], activation=self.activation[0]))

                    model.add(Dense(len(self.classes_), activation=self.activation[1]))

                    opt = getattr(keras.optimizers,self.optimize)

                    #add sp funtions in metrics if have 'sp' in self.metrics
                    if 'sp' in self.metrics:
                        self.metrics.remove('sp')

                    
                    callbacks_list.append(metricsAdd('sp',self.verbose))
                        #self.metrics.append(sp_index)

                        #self.es_kwargs['monitor'] = sp_index
                        #self.es_kwargs['mode'] = 'max'



                    model.compile(loss=self.loss, optimizer=opt(**self.op_adam_kwargs), metrics=self.metrics)


                    if self.early_stopping:
                        #callbacks_list.append(get_objects(keras.callbacks,
                        #                                  'EarlyStopping',
                        #                                  self.es_kwargs))

                        st = StopTraining(restore_best_weights=self.es_kwargs['restore_best_weights'],
                                          verbose=self.es_kwargs['verbose'],
                                          patience=self.es_kwargs['patience'],
                                          min_delta=0)

                        callbacks_list.append(st)


                    if self.save_best_model:
                        self.mc_kwargs['filepath'] = os.path.join(self.get_params()['dir'],'best_model.h5')
                        callbacks_list.append(get_objects(keras.callbacks,
                                                         'ModelCheckpoint',
                                                         self.mc_kwargs))

                    if self.train_log:
                        log_files[init] = os.path.join(self.get_params()['dir'],'log_train_init_{0}.csv'.format(init))
                        self.lg_kwargs['filename'] = log_files[init]
                        callbacks_list.append(get_objects(keras.callbacks,
                                                          'CSVLogger',
                                                          self.lg_kwargs))


                    if train_id is None:
                        if self.verbose: print ('Warning: train_id was passed as None')
                        validation_data = None
                        x_data = preproc_X
                        y_sparse = sparce_y
                        y_fit = y
                        monitor = 'loss'
                        sample_weight_tmp = sample_weight
                    else:
                        monitor = 'val_loss'

                        #I made this change beacuse of some problems on Hierarq Classe
                        x_data_train = self.scaler_.transform(X[train_id])
                        x_data_test = self.scaler_.transform(X[test_id])

                        sparce_y = np_utils.to_categorical(y[train_id])
                        sparce_y_train = _check_sparce(sparce_y,self.activation[1])

                        self._y_train = sparce_y_train

                        sparce_y = np_utils.to_categorical(y[test_id])
                        sparce_y_test = _check_sparce(sparce_y,self.activation[1])

                        self._y_test = sparce_y_test

                        x_data = x_data_train
                        y_sparse = sparce_y_train

                        y_fit = y[train_id]

                        self._y = y

                        if sample_weight is None:
                            sample_weight_tmp = sample_weight
                            validation_data = (x_data_test,sparce_y_test)
                        else:
                            sample_weight_tmp = sample_weight[train_id]
                            validation_data = (x_data_test,sparce_y_test,sample_weight[test_id])


                    init_trn_desc = model.fit(x_data,y_sparse,
                                              epochs=self.epoch,
                                              batch_size=self.batch_size,
                                              callbacks=callbacks_list,
                                              validation_split=self.validation_fraction,
                                              validation_data=validation_data,
                                              class_weight=self._class_weight(y_fit),
                                              sample_weight=sample_weight_tmp,
                                              verbose=self.verbose,
                                              shuffle=self.shuffle)

                    #print model_files

                    if self.monitor is 'sp':
                        if np.max(init_trn_desc.history[self.monitor]) > best_sp:
                            best_init = init
                    else:
                        if np.min(init_trn_desc.history[self.monitor]) < best_loss:
                            best_init = init


            if self.train_log:
                copyfile(os.path.join(self.get_params()['dir'], 'log_train_init_{0}.csv'.format(best_init)),
                         os.path.join(self.get_params()['dir'], 'log_best_from_{0}.csv'.format(best_init)))

                copyfile(os.path.join(self.get_params()['dir'], 'log_best_from_{0}.csv'.format(best_init)),
                         os.path.join(self.get_params()['dir'], 'log_train.csv'.format(best_init)))

                self.lg_kwargs['filename'] = os.path.join(self.get_params()['dir'],'log_train.csv')

            
            self.model = load_model(self.mc_kwargs['filepath'])


            return self

    def predict(self,X,y=None,predict='classes'):

        if self.model is None:
            raise Exception('use \'fit\' function first')

        preproc_X = self.scaler_.transform(X)
        
        if False:#len(self.classes_) == 2:
            #class and not class
            pred = self.model.predict(preproc_X)
            pred = np.array(map(lambda x:x[1],pred))

            if predict is 'sparce':
                return pred
            else:
                threshold = 0
                # if sample is higher then threshold, this sample belongs to class 1 otherwise 0
                return np.array(map(lambda x: 1 if x >= threshold else 0,pred))

        if predict is 'sparce':
            return self.model.predict(preproc_X)
        else:
            return self.model.predict_classes(preproc_X)

    def predict_proba(self,X):

        if self.model is None:
            raise Exception('use \'fit\' function first')

        preproc_X = self._preprocess(X)
        return self.model.predict_proba(preproc_X)

    def decision_function(self,X,predict='sparce'):


        if len(self.classes_) == 2:
            #class and not class

            if self.model is None:
                raise Exception('use \'fit\' function first')

            preproc_X = self._preprocess(X)

            pred = self.model.predict(preproc_X)
            pred = np.array(map(lambda x:x[1],pred))

            if predict is 'sparce':
                return pred
            else:
                threshold = 0
                # if sample is higher then threshold, this sample belongs to class 1 otherwise 0
                return np.array(map(lambda x: 1 if x >= threshold else 0,pred))
        return self.predict(X,predict=predict)

    def score(self,X,y=None):

        if self.model is None:
            raise Exception('use \'fit\' function first')

        preproc_X = self._preprocess(X)
        return self.model.evaluate(preproc_X,y)