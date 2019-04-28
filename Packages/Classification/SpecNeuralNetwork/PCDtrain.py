import os
import sys
import time
sys.path.insert(0,'..')





import argparse
#Argument Parse config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-c", "--components", default=30,type=int, help="Select number of neurouns in Neural Network")
parser.add_argument("-s", "--specialist",default=0,type=int, help= "Select a specialist ")
args = parser.parse_args()


from classificationConfig import CONFIG
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from sklearn import metrics
from multiprocessing import Pool, TimeoutError

import multiprocessing
from Functions.StatFunctions import sp

from NeuralNetworkAnalysis import NeuralNetworkAnalysis as nnm


num_processes = multiprocessing.cpu_count()

analysis_name = 'Classification'

data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']
class_interest = [1,2,3,4,5,6,7,8,9,10]#,21,23]

nums_componets = None
dev = True
#DataModule

analysis = nnm(analysis_name="NeuralNetwork",development_flag=True,development_events=400,verbose=False)
#analysis.infoData()
#analysis.setRangeClass([i-1 for i in class_interest])
#analysis.balanceData()
#df_data = pd.DataFrame(analysis.infoData())
weight_class = analysis.getClassesWeight(flag=True)
all_data,all_trgt,_ = analysis.getData()

#PCD Module
from PCD2 import TrnParams
from PCDModule import PCDModule as Pmod

nums_componets = args.components
list_all_data = {}

pcd1 = Pmod(path='/home/venancio/Workspace/SonarAnalysis/Results/Classification',n_components=nums_componets,n_folds=10)
for ifold in [0]:
    pcd_params = TrnParams(learning_rate=0.001,batch_size=512,init=10)
    pcd1.train(all_data,all_trgt,ifold,trn_params=pcd_params,flag_class_weight=True)
    analysis.all_data = pcd1.getDataCompact()
    
X_train,X_test,y_train,y_test = analysis.train_test_split()

if dev:
    results_path = results_path+'/dev/{0}_data_{1}_classes_{2}_PCD_{3}_fold'.format(X_train.shape[0],len(analysis.class_labels),nums_componets,ifold)
    
    
#only one Specialist
from Functions import TrainParameters
from OneSpecialistClass import OneSpecialistClass as ospc
import keras.backend as K

trparamns = TrainParameters.SpecialistClassificationTrnParams(weight=True,n_neurons=20,n_inits=5,optmizerAlgorithm='Adam',n_folds=10,batch_size=128,n_epochs=1000,verbose=True)

ispec = ospc(results_path,trparamns,args.specialist)
print ispec.result_path_fold[0]
model = ispec.train_n_folds(X_train,y_train)#,fold=7)

