import os
import sys
sys.path.insert(0,'..')

import pickle
import shutil
import numpy as np
import time
import string

import multiprocessing

from Functions import TrainParameters
from Functions import StatFunctions as sf

from Functions.MetricCustom import ind_SP
from Functions.ModelPath import ModelPath
from classificationConfig import CONFIG
from NeuralNetworkAnalysis import NeuralNetworkAnalysis
from PCD import PCDIndependent
from sklearn.externals import joblib

from sklearn import preprocessing

from keras.models import load_model
from Functions import PreProcessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD,Adam
import keras.callbacks as callbacks
from keras.utils import np_utils
from sklearn.utils import class_weight


import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from matplotlib.ticker import FuncFormatter



num_processes = multiprocessing.cpu_count()


class NeuralNetworkModule(ModelPath):

    def __init__(self,pathResult,trnParams,create_folders=True):
        super(NeuralNetworkModule, self).__init__(trnParams)
        self.pathResult = pathResult
        self.trnparams = trnParams
        self.trn_params = None
        self.CVO = None
        self.model_path = self.fold_model_path()
        self.figure_path = self.fold_figure_path()
        
    def fold_model_path(self,fold=None):
        fold_model_list = []
        for ifold in range(self.trnparams.folds):
            folder = os.path.join(self.pathResult,self.model() + "/fold{0}".format(ifold))
            if not os.path.exists(folder):
                print ("Creating " + folder)
                os.makedirs(folder)
            fold_model_list.append(folder)
        if not fold is None:
            return fold_model_list[fold]
        else:
            return fold_model_list
        
    def fold_figure_path(self,fold=None):
        fold_figure_list = []
        for ifold in range(self.trnparams.folds):
            folder = os.path.join(self.pathResult,self.figures() + "/fold{0}".format(ifold))
            if not os.path.exists(folder):
                print ("Creating " + folder)
                os.makedirs(folder)
            fold_figure_list.append(folder)
        if not fold is None:
            return fold_figure_list[fold]
        else:
            return fold_figure_list
        
    def set_cross_validation(self,all_trgt,verbose=False):
        self.CVO = TrainParameters.ClassificationFolds(folder=self.pathResult,
                                                       n_folds=self.trnparams.folds,
                                                       trgt=all_trgt,
                                                       verbose=verbose)

    def check_CVO(self,all_trgt):
        if self.CVO is None:
            self.set_cross_validation(all_trgt)
    
    def loadModelTrain(self,fold=None):
        
        if fold==None:
            #Load all model in the path and return a dictionary
            list_model = [load_model(os.path.join(self.model_path[ifold],'best_model.h5')) for ifold in range(self.n_folds)]
            return list_model 
        else:
            #Load a especific model according to especific fold
            return load_model(os.path.join(self.model_path[fold],'best_model.h5'))

    def preprocess(self,data,trgt,fold,filename='preprocess.jbl'):
        prepro_file = os.path.join(self.model_path[fold],filename)

        if not (os.path.exists(prepro_file)):
            
            self.check_CVO(trgt)
            
            train_id, test_id = self.CVO[fold]

            if self.trnparams.verbose == True:
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
            if self.trnparams.verbose == True:
                print "NeuralClassication preprocess function: loading scaler for fold %i"%(fold)
            [scaler] = joblib.load(prepro_file)

        data_proc = scaler.transform(data)
        
        if (self.trnparams.output_activation=='tanh'):
            trgt_sparse = 2*np_utils.to_categorical(trgt.astype(int)) -1
        else:
            trgt_sparse = np_utils.to_categorical(trgt.astype(int))
            #trgt_sparse = np_utils.to_categorical(trgt.astype(int))
        # others preprocessing process
        self.trgt_sparse = trgt_sparse
        return [data_proc,trgt_sparse,prepro_file]
    
    def eff(self,all_data,all_trgt,fold,score='recall'):
        ''' Function for return efficiency of each class'''
        self.analysis_data_file = os.path.join(self.model_path[fold],
                                               "analysis_eff_output.jbl".format(fold))
        self.check_CVO(all_trgt)
        
        train_id, test_id = self.CVO[fold]
        
        output = None
        if not os.path.exists(self.analysis_data_file):
            model = self.train(all_data,all_trgt,fold)
            norm_data,norm_trgt,_ = self.preprocess(preprocess(all_data,all_trgt,fold))
                                                    
            output = model.predict_classes(norm_data)#self.all_data)
            if score == 'recall':
                eff = recall_score(y_true=all_trgt[test_id],y_pred=output[test_id],average=None)
            joblib.dump([eff,output],self.analysis_data_file,compress=9)
        else:
            print "loading analysis output file of fold {0}".format(fold)
            [eff,output] = joblib.load(self.analysis_data_file)
        
        return [eff,output]
                                                    
    def train_n_folds(self,data,trgt):
        model = {}
        for ifold in range(self.trnparams.folds):
            model[ifold] = self.train(data,trgt,ifold)
        return model
            
                                                    
    def train(self,data,trgt,fold):
        
        self.check_CVO(trgt)
                                                    
        train_id, test_id = self.CVO[fold]
        data_proj,trgt_sparse,_ = self.preprocess(data,trgt,fold)
        model = self.fit(data_proj,
                         trgt_sparse,
                         train_id,
                         test_id,
                         fold,
                         self._get_weight(trgt,train_id))
        return model
        
    
    def fit(self,data,trgt_sparse,train_ids,test_ids,num_fold,weight):
        best_model_path = os.path.join(self.model_path[num_fold],'best_model.h5')
        best_train_path = os.path.join(self.model_path[num_fold],'best_train.csv')
        
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
                
                my_model.add(Activation('tanh'))
                
                my_model.add(Dense(self.trnparams.n_neurons, input_dim=data.shape[1],
                                        kernel_initializer='uniform'))

                my_model.add(Activation(self.trnparams.hidden_activation))

                my_model.add(Dense(trgt_sparse.shape[1],kernel_initializer='uniform'))

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
                
                my_model.compile(loss=self.trnparams.loss, optimizer=opt, metrics=self.trnparams.metrics)

                filepath[init] = os.path.join(self.model_path[num_fold],'{0}_init_model.h5'.format(init))

                saveBestModel = callbacks.ModelCheckpoint(filepath[init], monitor='val_loss', verbose=0,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)

                filename[init] = os.path.join(self.model_path[num_fold],'{0}_init_train.csv'.format(init))

                csvLog = callbacks.CSVLogger(filename[init])

                earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=25, verbose=0,
                                                        mode='auto')

                init_trn_desc = my_model.fit(data[train_ids], trgt_sparse[train_ids],
                                                epochs=self.trnparams.n_epochs,
                                                batch_size=self.trnparams.batch_size,
                                                callbacks=[earlyStopping,saveBestModel,csvLog],
                                                class_weight=weight,
                                                verbose=self.trnparams.train_verbose,
                                                validation_data=(data[test_ids],
                                                                trgt_sparse[test_ids]),
                                                shuffle=True)
                
                if np.min(init_trn_desc.history['val_loss']) < best_loss:
                    best_init = init
                    best_loss = np.min(init_trn_desc.history['val_loss'])
                    print "fold -> {0}  best init -> {1} with min val_loss -> {2}".format(num_fold,best_init,best_loss)

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
    
    
    
    def _get_weight(self,trgt,train_id):
        if self.trnparams.weight:
            weight  = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                       np.unique(trgt[train_id]),
                                                                       trgt[train_id])))
        else:
            weight = None
        self.weight = weight
        return weight
    
    
    
    def PCDfold(self,all_data,all_trgt,ifold,CVO,trn_params):

        if self.trn_params.params['pcd'] == False:
            print "pcd doesn't exist"
            return -1
        
        self.foldPCDPath = os.path.join(self.getBaseResultsPath(),"{0}_fold_PCDs".format(ifold))
        
        if not os.path.exists(self.foldPCDPath):
            print ("Creating" + self.foldPCDPath)
            os.makedirs(self.foldPCDPath)

        train_id, test_id = CVO[ifold]
        
        if os.path.exists(os.path.join(self.getBaseResultsPath(), "{0}_fold_preprocess.jbl".format(ifold))):
            os.remove(os.path.join(self.getBaseResultsPath(), "{0}_fold_preprocess.jbl".format(ifold)))
            
        data_preproc, trgt_preproc,_ = self.preprocess(all_data,self.getBaseResultsPath(),all_trgt,trn_params,CVO,ifold)
        
        os.remove(os.path.join(self.getBaseResultsPath(), "{0}_fold_preprocess.jbl".format(ifold)))
        trgt_sparse = trgt_preproc

        pcds = PCDIndependent(trn_params.params['pcd'])
        
        model = pcds.fit(data=data_preproc, target=trgt_sparse, 
                         train_ids=train_id, test_ids=test_id,
                         path=self.foldPCDPath,
                         class_weight=self.getClassesWeight(trn_params.params['class_weight_flag']))
        
        # after trained with pcd in Auto mode, the algoritm pass the new number of components for trn_params
        if trn_params.params['pcd'] == 'Auto':
            trn_params.params['pcd'] = pcds.n_components

        self.PCD_data = pcds.transform(all_data)
        self.pcds_degreeMatrix[ifold] = pcds.get_degree_matrix()
        return [self.PCD_data,self.all_trgt,pcds]

    def confMatrix_sb_each(self,all_data,all_trgt,figX=30,figY=30, fold=0, cmap=None):
        import matplotlib.pyplot as plt
        import seaborn as sn
        
        print 'NeuralClassication analysis analysis conf mat function'
        
        train_id, test_id = self.CVO[fold]
        _,output = self.eff(all_data,all_trgt,fold)

        cm = confusion_matrix(all_trgt[test_id], output[test_id])

        fig, ax = plt.subplots(figsize=(figX,figY),nrows=1, ncols=1)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # saving matrix of confusion for future analysis 
        cm_file = os.path.join(self.getBaseResultsPath(), "{0}_fold_confusionMatrix.csv".format(fold))
        if not os.path.exists(cm_file):
            pd.DataFrame(cm_normalized).to_csv(cm_file)
        
        uniform_data = np.random.rand(10, 12)
        #fmt = lambda x,pos: '{:.0%}'.format(x)
        
        #remember -> heatmap is better if use cmap= 'Greys' 
        ax = sn.heatmap(cm_normalized, annot=True, fmt='.0%',
                        #cbar_kws={'format': FuncFormatter(fmt)},
                        ax=ax,cmap=cmap,vmin=0, vmax=1)
        ax.set_title('Confusion Matrix',fontweight='bold',fontsize=40)
        ax.set_xticklabels(self.getClassLabels())
        ax.set_yticklabels(self.getClassLabels(),rotation=45, va='top')

        ax.set_ylabel('True Label',fontweight='bold',fontsize=40)
        ax.set_xlabel('Predicted Label',fontweight='bold',fontsize=40)


        self.cm_picture_file = os.path.join(self.figure_path[fold], "cm_figure.png".format(fold))
        fig.savefig(self.cm_picture_file)

        return fig,cm_normalized
    
    def analysis_train_CSV(self,fold,x='epoch',y=['val_loss','loss'],ylabel='MSE',xlim=None,ylim=None,xsize=6,ysize=6):
        import matplotlib.pyplot as plt
        
        print 'NeuralClassication analysis train plot function'
        csv_file = os.path.join(self.model_path[fold],'best_train.csv')
        
        dataset = pd.read_csv(csv_file)
        ax = dataset.plot(x=x,y=y,xlim=xlim,ylim=ylim,figsize=(xsize,ysize))
        ax.set_ylabel(ylabel,fontweight='bold',fontsize=15)
        ax.set_xlabel(x,fontweight='bold',fontsize=15)
        ax.grid()
        ax.get_figure().savefig(os.path.join(self.figure_path[fold], "trn_desc_figure.png".format(fold)))
        return ax

    def vectorSP_analysis(self,data,trgt,CVO,trn_params):
        file = os.path.join(self.pathResult,self.model() + "/vector_sp_{0}_folds.csv".format(self.trnparams.folds))
        self.vector_sp = []
        
        if os.path.exists(file):
            self.vector_sp = [pd.read_csv(file).values[i][0] for i in range(pd.read_csv(file).values.shape[0])]
            return self.vector_sp    
        
        for ifold in range(trn_params.params['folds']):
            if type(data) is dict:
                eff_recall,_ = self.eff(data[ifold],trgt,CVO,trn_params,ifold)
            else:
                eff_recall,_ = self.eff(data,trgt,CVO,trn_params,ifold)
            
            self.vector_sp.append(sf.sp(eff_recall))
        
        pd.DataFrame(self.vector_sp).to_csv(file,index=False,index_label=False)
        print "vector of SP was saved in result folder as CSV file"
        return self.vector_sp
