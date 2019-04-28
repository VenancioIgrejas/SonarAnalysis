import os
import sys
import time

sys.path.insert(0,'/home/venancio/WorkPlace/SonarAnalysis/Packages/Classification')




import argparse
#Argument Parse config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-c", "--components", default=30,type=int, help="Select number of neurouns in Neural Network")
parser.add_argument("-f", "--fold",default=10,type=int, help= "Select number of folds in Neural Network")
parser.add_argument("-s", "--specialist",default=0,type=int, help= "Select the especialist")
parser.add_argument("--dev",default=True,type=bool,help= "development flag")
args = parser.parse_args()

from classificationConfig import CONFIG
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from sklearn import metrics

import multiprocessing

from NeuralNetworkAnalysis import NeuralNetworkAnalysis as nnm

from OneSpecialistClass import OneSpecialistClass as ospc
from Functions import TrainParameters
from PCDSpecialist import TrnParams
import keras.backend as K

num_processes = multiprocessing.cpu_count()


analysis_name = 'Classification'

data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']
class_interest = [1,2,3,4,5,6,7,8,9,10]#,21,23]

dev = args.dev

#DataModule

analysis = nnm(analysis_name="NeuralNetwork",development_flag=True,development_events=400,verbose=False)
#analysis.infoData()
#analysis.setRangeClass([i-1 for i in class_interest])
#analysis.balanceData()
#analysis.infoData()
weight_class = analysis.getClassesWeight(flag=True)
all_data,all_trgt,_ = analysis.getData()
X_train,X_test,y_train,y_test = analysis.train_test_split()

if dev:
    results_path = results_path+'/dev/{0}_data_{1}_classes_each_specwithPCD'.format(X_train.shape[0],len(analysis.class_labels))


trparamns = TrainParameters.SpecialistClassificationTrnParams(weight=True,n_neurons=20,n_inits=5,optmizerAlgorithm='Adam',n_folds=10,batch_size=128,n_epochs=1000,verbose=True)

trn_params_new = TrnParams(learning_rate=0.001,batch_size=128,init=10,optimizers='adam')
ispec = ospc(results_path,trparamns,args.specialist)
ispec.set_trparams_pcd(trn_params_new,10)
#ispec.train_pcd(X_train,y_train,fold=args.fold)

data_red = ispec.transform_data_pcd(data=X_train,trgt=y_train,fold=args.fold)
print "shape {0} of X_train -> shape {1}".format(X_train.shape,data_red.shape)
model = ispec.train(data=data_red,trgt=y_train,fold=args.fold,met_compact='PCD')

ispec.output(data_red,y_train,args.fold,met_compact='PCD',flag_already_trained = True)
print "train finished spec->{0},fold->{1} with PCD".format(args.specialist,args.fold)
