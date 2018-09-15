import os

import sys
sys.path.insert(0,'..')

import pandas as pd
import numpy as np

from Functions import TrainParameters
from Functions import StatFunctions as sf
from SpecPath import SpecPath

from classificationConfig import CONFIG
from sklearn.externals import joblib

from sklearn import preprocessing

from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD,Adam
import keras.callbacks as callbacks
from keras.utils import np_utils
from sklearn.utils import class_weight


from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from matplotlib.ticker import FuncFormatter


from keras import backend as K

class OneSpecialistClass(SpecPath):
    """docstring for SpecialistTrain."""
    def __init__(self, pathResult,trnParams,spec_num):
        super(OneSpecialistClass, self).__init__(trnParams,spec_num)
        self.spec_num = spec_num
        self.trnparams = trnParams
        self.resultPath = pathResult
        self.result_path_fold = {}
        self.result_spec_model = os.path.join(pathResult,self.spec_model)
        self.result_spec_figures =  os.path.join(pathResult,self.spec_model)

        for ifold in range(trnParams.folds):
            self.result_path_fold[ifold] = os.path.join(pathResult,self.path_fold[ifold])
            if not os.path.exists(self.result_path_fold[ifold]):
                os.makedirs(self.result_path_fold[ifold])

        if not os.path.exists(self.result_spec_model):
            os.makedirs(self.result_spec_model)

        if not os.path.exists(self.result_spec_figures):
            os.makedirs(self.result_spec_figures)



    def set_cross_validation(self,all_trgt):
        return TrainParameters.ClassificationFolds(folder=self.result_spec_model,
                                                    n_folds=self.trnparams.folds,
                                                    trgt=all_trgt,
                                                    verbose=True)
    
    def prepare_target(self,trgt):
        index_spec, =np.where(trgt==self.spec_num)
        spec_trgt = trgt*0
        spec_trgt[index_spec] = 1
        return spec_trgt
    
    def preprocess(self,data,trgt,fold):
        prepro_file = os.path.join(self.result_path_fold[fold], "preprocess.jbl".format(fold))

        if not (os.path.exists(prepro_file)):
            train_id, test_id = self.set_cross_validation(trgt)[fold]

            print 'NeuralClassication preprocess function: creating scaler for fold %i'%(fold)
            # normalize data based in train set
            if self.trnparams.norm == 'mapstd':
                scaler = preprocessing.StandardScaler().fit(data[train_id,:])
            elif self.trnparams.norm == 'mapstd_rob':
                scaler = preprocessing.RobustScaler().fit(data[train_id,:])
            elif self.trnparams.norm == 'mapminmax':
                scaler = preprocessing.MinMaxScaler().fit(data[train_id,:])
            joblib.dump([scaler],prepro_file,compress=9)

        else:
            print "NeuralClassication preprocess function: loading scaler for fold %i"%(fold)
            [scaler] = joblib.load(prepro_file)

        data_proc = scaler.transform(data)

        if (self.trnparams.output_activation=='tanh'):
            trgt_sparse = 2*np_utils.to_categorical(trgt.astype(int)) -1
            trgt_sparse = trgt_sparse[:,self.spec_num]
        else:
            trgt_sparse = np_utils.to_categorical(trgt.astype(int))
            #trgt_sparse = np_utils.to_categorical(trgt.astype(int))
            trgt_sparse = trgt_sparse[:,self.spec_num]
        # others preprocessing process
        return [data_proc,trgt_sparse,prepro_file]

    def output(self,data,trgt,fold):
        output_path = os.path.join(self.result_path_fold[fold],'output.csv')

        if not os.path.exists(output_path):
            train_id , test_id = self.set_cross_validation(trgt)[fold]
            data_proj,trgt_sparse,_ = self.preprocess(data,trgt,fold)
            model = self.train(data,trgt,fold)
            output = model.predict(data_proj)
            df = pd.DataFrame(output)
            df.to_csv(output_path, index=False)
        else:
            output = pd.read_csv(output_path).values

        return output

    def train_n_folds(self,data,trgt):
        model = {}
        for ifold in range(self.trnparams.folds):
            model[ifold] = self.train(data,trgt,ifold)
        return model

    def train(self,data,trgt,fold):
        train_id , test_id = self.set_cross_validation(trgt)[fold]
        spec_trgt = self.prepare_target(trgt)
        if self.trnparams.weight:
            weight  = dict(enumerate(class_weight.compute_class_weight('balanced',np.unique(spec_trgt[train_id]),spec_trgt[train_id])))
            self.weight = weight
        else:
            weight = None
        data_proj,trgt_sparse,_ = self.preprocess(data,trgt,fold)
        model = self.fit(data_proj,trgt_sparse,train_id,test_id,fold,weight)
        return model

    def fit(self,data,target,train_ids,test_ids,num_fold,weight):

        best_model_path = os.path.join(self.result_path_fold[num_fold],'best_model.h5')
        best_train_path = os.path.join(self.result_path_fold[num_fold],'best_train.csv')

        
        if not os.path.exists(best_model_path):
            
            filename = {}
            filepath = {}
            
            best_loss = 999
            best_init = 0
            
            print "Starting train of fold {0}".format(num_fold+1)
            for init in range(self.trnparams.n_inits):
                print "[+] {0} of {1} inits".format(init+1,self.trnparams.n_inits)
                my_model = Sequential()
                
                my_model.add(Dense(data.shape[1],
                                input_dim=data.shape[1],
                                kernel_initializer='identity',
                                trainable=False))
                my_model.add(Activation('relu'))
                
                my_model.add(Dense(self.trnparams.n_neurons, input_dim=data.shape[1],
                                        kernel_initializer='uniform'))
                
                my_model.add(Activation(self.trnparams.hidden_activation))
                
                my_model.add(Dense(1,kernel_initializer='uniform'))
                
                my_model.add(Activation(self.trnparams.output_activation))

                if self.trnparams.optmizerAlgorithm == 'SGD':
                    opt = SGD(lr=self.trnparams.learning_rate,decay=self.trnparams.learning_decay,
                              momentum=self.trnparams.momentum, nesterov=self.trnparams.nesterov)
                
                if self.trnparams.optmizerAlgorithm == 'Adam':
                    opt = Adam(lr=self.trnparams.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	                

            # compile the model

                my_model.compile(loss=self.trnparams.loss, optimizer=opt, metrics=self.trnparams.metrics)

                filepath[init] = os.path.join(self.result_path_fold[num_fold],'{0}_init_model.h5'.format(init))

                saveBestModel = callbacks.ModelCheckpoint(filepath[init], monitor='val_loss', verbose=0,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)

                filename[init] = os.path.join(self.result_path_fold[num_fold],'{0}_init_train.csv'.format(init))

                csvLog = callbacks.CSVLogger(filename[init])

                earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=25, verbose=0,
                                                        mode='auto')

                init_trn_desc = my_model.fit(data[train_ids], target[train_ids],
                                                epochs=self.trnparams.n_epochs,
                                                batch_size=self.trnparams.batch_size,
                                                callbacks=[earlyStopping,saveBestModel,csvLog],
                                                class_weight=weight,
                                                verbose=self.trnparams.train_verbose,
                                                validation_data=(data[test_ids],
                                                                target[test_ids]),
                                                shuffle=True)



                if np.min(init_trn_desc.history['val_loss']) < best_loss:
                    best_init = init
                    best_loss = np.min(init_trn_desc.history['val_loss'])

            
            os.rename(filepath[best_init], best_model_path)
            os.rename(filename[best_init], best_train_path)

            for init in range(self.trnparams.n_inits):
                if os.path.exists(filename[init]):
                    os.remove(filename[init])

                if os.path.exists(filepath[init]):
                    os.remove(filepath[init])

            return load_model(best_model_path)
        else:
            print "Loading model file of fold {}".format(num_fold+1)
            return load_model(best_model_path)
