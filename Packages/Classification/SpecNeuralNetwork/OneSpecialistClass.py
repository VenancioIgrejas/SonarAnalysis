import os

import sys
sys.path.insert(0,'..')

import pandas as pd
import numpy as np
import keras.backend as K

from Functions import TrainParameters
from Functions import StatFunctions as sf
from SpecPath import SpecPath
from PCDSpecialist import PCDIndependent,TrnParams

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
        self.type_data = 'train'
        self.result_path_fold = {}
        self.result_path_fold = {}
        self.result_spec_model = os.path.join(pathResult,self.spec_model)
        self.result_spec_figures =  os.path.join(pathResult,self.spec_figures)

        for ifold in range(trnParams.folds):
            self.result_path_fold[ifold] = os.path.join(pathResult,self.path_fold[ifold])
            if not os.path.exists(self.result_path_fold[ifold]):
                os.makedirs(self.result_path_fold[ifold])

        if not os.path.exists(self.result_spec_model):
            os.makedirs(self.result_spec_model)

        if not os.path.exists(self.result_spec_figures):
            os.makedirs(self.result_spec_figures)
    
        self.set_trparams_pcd()

    
    def set_trparams_pcd(self,trn_params=None,n_componentes=10):
        self.n_componets = n_componentes
        if trn_params is None:
            self.trn_params_PCD = TrnParams(learning_rate=0.001,batch_size=128,init=10,optimizers='adam')
        else:
            self.trn_params_PCD = trn_params
        
    def set_partition_data(self,type='train'):
        print 'set '+type+' for transform with PCD'
        self.type_data = type
    
    def check_train(self):
        flag = True
        for ifold in range(self.trnparams.folds):
            file = os.path.join(self.result_path_fold[ifold], "preprocess.jbl".format(ifold))
            if not os.path.exists(file):
                print "train first Fold {0} of Specialist {1}".format(ifold,self.spec_num)
                flag = False
        return flag

    def set_cross_validation(self,all_trgt,path=None):
            if path is None:
                path = self.result_spec_model
            return TrainParameters.ClassificationFolds(folder=path,
                                                    n_folds=self.trnparams.folds,
                                                    trgt=all_trgt,
                                                    verbose=False)

    def prepare_target(self,trgt):
        index_spec, = np.where(trgt==self.spec_num)
        spec_trgt = trgt*0
        spec_trgt[index_spec] = 1
        return spec_trgt

    def train_pcd(self,data,trgt,fold):
        if self.type_data == 'train':
            PCD_folder = os.path.join(self.result_path_fold[fold],"PCD")
            
        if self.type_data == 'test':
            print "accessing PCD folder for teste data"
            PCD_folder = os.path.join(self.result_path_fold[fold],"PCD_teste_data")

        if not os.path.exists(PCD_folder):
            os.makedirs(PCD_folder)

        train_id , test_id = self.set_cross_validation(trgt,PCD_folder)[fold]
        spec_trgt = self.prepare_target(trgt)
        
        data_proj,trgt_sparse,_ = self.preprocess(data,trgt,fold,PCD_folder+'/preprocess.jbl')
        
        ipcd = PCDIndependent(n_components=self.n_componets)
        
        self.trn_params_PCD.to_json(PCD_folder)

        ipcd.fit(data=data_proj,
                 target=trgt_sparse,
                 train_ids=train_id,
                 test_ids=test_id,
                 trn_params=self.trn_params_PCD,
                 path=PCD_folder,
                 class_weight=self._get_weight(spec_trgt,train_id))

        return ipcd
    
    def transform_data_pcd(self,data,trgt,fold):
        ipcd = self.train_pcd(data,trgt,fold)
        return ipcd.transform(data)

    def preprocess(self,data,trgt,fold,filename='preprocess.jbl'):
        prepro_file = os.path.join(self.result_path_fold[fold], filename)
        #print prepro_file
        if not (os.path.exists(prepro_file)):
            if self.type_data == 'test':
                train_id, test_id = self.set_cross_validation(trgt,path=os.path.join(self.result_path_fold[fold],"PCD_teste_data"))[fold]
            else:
                train_id, test_id = self.set_cross_validation(trgt)[fold]
            if self.trnparams.verbose == True:
                print 'NeuralClassication preprocess function: creating scaler for fold %i'%(fold)
            # normalize data based in train set
            if self.trnparams.norm == 'mapstd':
                print data.shape
                print len(train_id)
                scaler = preprocessing.StandardScaler().fit(data[train_id,:])
            elif self.trnparams.norm == 'mapstd_rob':
                scaler = preprocessing.RobustScaler().fit(data[train_id,:])
            elif self.trnparams.norm == 'mapminmax':
                scaler = preprocessing.MinMaxScaler().fit(data[train_id,:])
            joblib.dump([scaler],prepro_file,compress=9)

        else:
            if self.trnparams.verbose == True:
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

    def output(self,data,trgt,fold,met_compact=None,flag_already_trained = False):
        output_path = os.path.join(self.result_path_fold[fold],'output.csv')

        if not os.path.exists(output_path):

            train_id , test_id = self.set_cross_validation(trgt)[fold]
            
            if met_compact is 'PCD':
                PCD_folder = os.path.join(self.result_path_fold[fold],"PCD")
                data_proj,trgt_sparse,_ = self.preprocess(data,
                                                          trgt,
                                                          fold,'preprocess_PCD.jbl')
            else:
                data_proj,trgt_sparse,_ = self.preprocess(data,trgt,fold)
                
            if flag_already_trained:
                model = self.load_model(fold)
            else:
                model = self.train(data,trgt,fold,met_compact=met_compact)
                
                
                
            output = model.predict(data_proj)[:,0]
            df = pd.DataFrame(output)
            df.to_csv(output_path, index=False)
            
        else:
            
            output = pd.read_csv(output_path).values[:,0]

        return output

    def train_n_folds(self,data,trgt,met_compact=None):
        model = {}
        for ifold in range(self.trnparams.folds):
            model[ifold] = self.train(data,trgt,ifold,met_compact)

        return model
    
    def load_model(self,num_fold):
        best_model_path = os.path.join(self.result_path_fold[num_fold],'best_model.h5')
        try:
            return load_model(best_model_path)
        except:
            print "file {0} not found ".format(best_model_path)
            
    def train(self,data,trgt,fold,met_compact=None):
        
        train_id , test_id = self.set_cross_validation(trgt)[fold]
        spec_trgt = self.prepare_target(trgt)
        
        if met_compact is "PCD":
            #data = self.transform_data_pcd(data,trgt,fold)
            data_proj,trgt_sparse,_ = self.preprocess(data,trgt,fold,'preprocess_PCD.jbl')
        else:
            data_proj,trgt_sparse,_ = self.preprocess(data,trgt,fold)
        
        model = self.fit(data_proj,trgt_sparse,train_id,test_id,fold,self._get_weight(spec_trgt,train_id))
        return model

    def fit(self,data,target,train_ids,test_ids,num_fold,weight):

        best_model_path = os.path.join(self.result_path_fold[num_fold],'best_model.h5')
        best_train_path = os.path.join(self.result_path_fold[num_fold],'best_train.csv')


        if not os.path.exists(best_model_path):

            filename = {}
            filepath = {}

            best_loss = 999
            best_init = 0

            print "Starting train of fold {0}".format(num_fold)
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
                    opt = SGD(lr=self.trnparams.learning_rate,
                              decay=self.trnparams.learning_decay,
                              momentum=self.trnparams.momentum, 
                              nesterov=self.trnparams.nesterov)

                if self.trnparams.optmizerAlgorithm == 'Adam':
                    opt = Adam(lr=self.trnparams.learning_rate, 
                               beta_1=0.9, 
                               beta_2=0.999, 
                               epsilon=None, 
                               decay=0.0, 
                               amsgrad=False)


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
                    print "spec {0}: fold -> {1}  best init -> {2} with min val_loss -> {3}".format(self.spec_num,
                                                                                                    num_fold,
                                                                                                    best_init,
                                                                                                    best_loss)


            os.rename(filepath[best_init], best_model_path)
            os.rename(filename[best_init], best_train_path)

            for init in range(self.trnparams.n_inits):
                if os.path.exists(filename[init]):
                    os.remove(filename[init])

                if os.path.exists(filepath[init]):
                    os.remove(filepath[init])

            return load_model(best_model_path)
        else:
            if self.trnparams.verbose == True:
                print "Loading model file of fold {0}".format(num_fold)
            return load_model(best_model_path)

    def _get_weight(self,spec_trgt,train_id):
        if self.trnparams.weight:
            if (self.trnparams.output_activation=='tanh'):
                list_weight = class_weight.compute_class_weight('balanced',
                                                                np.unique(spec_trgt[train_id]),
                                                                spec_trgt[train_id])
                weight = {-1:list_weight[0],1:list_weight[1]}
            else:
                weight  = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                           np.unique(spec_trgt[train_id]),
                                                                           spec_trgt[train_id])))
            self.weight = weight
        else:
            weight = None
        return weight

    def train_analysis(self,x='epoch',y='loss',fold=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if not fold is None:
            best_train_file = os.path.join(self.result_path_fold[fold],'best_train.csv')
            dt_train = pd.read_csv(best_train_file)
            dt_train.plot(x,y,ax=ax)
        else:
            for ifold in range(self.trnparams.folds):
                best_train_file = os.path.join(self.result_path_fold[ifold],'best_train.csv')
                dt_train = pd.read_csv(best_train_file)


        return fig, ax, dt_train
