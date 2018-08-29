import os
import sys
sys.path.insert(0,'..')

import numpy as np
import time
import pandas as pd
import json
import hashlib


from PCD2 import PCDIndependent,TrnParams
from Functions import TrainParameters,PreProcessing
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.externals import joblib
from keras.utils import np_utils
from keras.models import load_model

class PCDModule(PCDIndependent):
    '''
    module to train the pcd using a determining number of initializations and folds
    '''
    def __init__(self,path,n_components,n_folds,n_init=0):
        super(PCDModule,self).__init__(n_components)

        self.n_folds=n_folds
        self.__flag = False
        self.path = path
        self.data = None
        self.__dataComp = None
        
    
    def trainFlag(self,data,trgt):
        '''
        automated flag to compare existing pcds with the entered data
            data: data to be analyzed (events x features)
            target: class labels - targets (events)
            
            return: boolean flag
        '''
        nrow,ncolumn = data.shape
        
        self.path_pcd = os.path.join(self.path,'PCD_{0}x{1}_{2}nFolds'.format(nrow,ncolumn,self.n_folds))
        
        self.CVO = TrainParameters.ClassificationFolds(folder=self.path_pcd, n_folds=self.n_folds, trgt=trgt,
                                                           dev=False, verbose=True)
        param = {'DataSize':nrow*ncolumn,
                          'numEvents':ncolumn,
                          'numSample':nrow,
                          'numFolds':self.n_folds
                }
        
        if not os.path.exists(self.path_pcd):
            #create the folder for save model and analyses files of PCDs
            print "creating " + self.path_pcd 
            os.makedirs(self.path_pcd)
            
            #save important information about especific PCD
            with open(os.path.join(self.path_pcd,'infoPCD.json'), 'w') as outfile:
                json.dump(param, outfile)
            
            self.__flag = True
        else:
            
            #load infoPCD.json for compare with the new data input
            with open(os.path.join(self.path_pcd,'infoPCD.json'), "r") as read_file:
                data = json.load(read_file)
            
            hash_info = hashlib.sha384(json.dumps(data)).hexdigest()
            hash_atual = hashlib.sha384(json.dumps(param)).hexdigest()
            
            #if new data input is the same as the data in the json file, the program will continue
            if hash_info==hash_atual:
                self.__flag = True

        return self.__flag
        
    
    def train(self,data,trgt,ifold,trn_params=None,flag_class_weight=False,params='mapstd'):
        '''
        train function of PCD 
            data: data to be fitted (events x features)
			target: class labels - sparse targets (events x number of classes)
			ifold: choose which partition to train
            
            trn_params: PCD training parameter (optional)
            flag_class_weight: true if the data is unbalanced (boolean)
            params: types of normalize data function ('mapstd','mapstd_rob','mapminmax')
            
            return: boolean flag
        '''
        if not self.trainFlag(data,trgt):
            print 'infomation about PCD data is not compatible'
            return -1
        
        self.foldPCDPath = os.path.join(self.path_pcd,"{0}_fold_PCDs".format(ifold))
        
        if not os.path.exists(self.foldPCDPath):
            print ("Creating" + self.foldPCDPath)
            os.makedirs(self.foldPCDPath)
        
        data_preproc, trgt_preproc,_ = self.preprocess(data,self.foldPCDPath,trgt,self.CVO,ifold,params)
        
        train_ids, test_ids = self.CVO[ifold]
        
        if  flag_class_weight:
            class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                             np.unique(trgt[train_ids]),
                                                                             trgt[train_ids])))
        else:
            class_weights = None
        
        if trn_params==None:
            trn_params = TrnParams(learning_rate=0.001,batch_size=512,init=2)
        trn_params.to_json(self.foldPCDPath)
        
        self.fit(data=data_preproc,
                 target=trgt_preproc,
                 train_ids=train_ids,
                 test_ids=test_ids,
                 trn_params=trn_params,
                 path=self.foldPCDPath,
                 class_weight=class_weights)
        
        self.__dataComp = self.transform(data)
        
        return self
    
    def preprocess(self,data,path,trgt,CVO,fold,params='mapstd'):
        prepro_file = os.path.join(path, "{0}_fold_preprocess.jbl".format(fold))

        if not (os.path.exists(prepro_file)):
            train_id, test_id = CVO[fold]

            print 'NeuralClassication preprocess function: creating scaler for fold %i'%(fold)
            # normalize data based in train set
            if params == 'mapstd':
                scaler = preprocessing.StandardScaler().fit(data[train_id,:])
            elif params == 'mapstd_rob':
                scaler = preprocessing.RobustScaler().fit(data[train_id,:])
            elif params == 'mapminmax':
                scaler = preprocessing.MinMaxScaler().fit(data[train_id,:])
            joblib.dump([scaler],prepro_file,compress=9)

        else:
            print "NeuralClassication preprocess function: loading scaler for fold %i"%(fold)
            [scaler] = joblib.load(prepro_file)

        data_proc = scaler.transform(data)
        
        trgt_sparse = 2*np_utils.to_categorical(trgt.astype(int)) -1

        # others preprocessing process
        self.trgt_sparse = trgt_sparse
        return [data_proc,trgt_sparse,prepro_file]
    
    def getDataCompact(self):
        '''
        returns the data compressed by the PCD
        '''
        if self.__flag==False:
            print 'please, use train function for compact your data'
            return -1
        
        return self.__dataComp
        
    
    
    