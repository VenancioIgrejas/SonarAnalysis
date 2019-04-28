#!/usr/bin/python

import os
import sys
import telegram_send

sys.path.insert(0,'/home/venancio/Workspace/SonarAnalysis/Packages/Classification')




import argparse
#Argument Parse config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-n", "--neurons", default=30,type=int, help="Select number of neurouns in Neural Network")
parser.add_argument("-e", "--epochs", default=1000,type=int, help="Select number of epochs in Neural Network")
parser.add_argument("-i", "--init",default=10,type=int, help="Select number of inicialization in Neural Network")
parser.add_argument("-b", "--batch",default=512,type=int, help="Select number of inicialization in Neural Network")
parser.add_argument("-f", "--fold",default=10,type=int, help= "Select number of folds in Neural Network")
parser.add_argument("-o", "--optmizer",default="sgd",type=str,help= "Select the optmizer Algorithm for Neural Network")
parser.add_argument("-l", "--loss",default="mean_squared_error",type=str,help= "Select loss type for Neural Network")
parser.add_argument("-p", "--pcd",default=0,type=int,help= "number of pcds -> -1 for Auto, 0 for None")
parser.add_argument("-w", "--weight",default=True,type=bool,help= "number of pcds -> -1 for Auto, 0 for None")
parser.add_argument("-c", "--countCPU",default=8,type=int,help= "number of cores for multiprocessing, default 8 cores")
parser.add_argument("--each-fold",default=0,type=int,help= "choose which fold train (default: train all folds)")

args = parser.parse_args()

import time
import datetime

from classificationConfig import CONFIG
import numpy as np

from PCD2 import TrnParams
from PCDModule import PCDModule as Pmod
from Functions.MetricCustom import ind_SP
import multiprocessing
from functools import partial
from contextlib import contextmanager

from NeuralNetworkModule import NeuralNetworkModule as nnm



ipcd = args.pcd
print ipcd
if args.pcd==0:
    ipcd=None
if args.pcd==-1:
    ipcd='Auto'


num_processes = args.countCPU

analysis_name = 'Classification'

data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']

class_interest = [1,2,3,4,12,20]#[6,7,9,16,21,23]

flag = args.weight

analysis = nnm(analysis_name="NeuralNetwork",development_flag=False,development_events=5000,verbose=False)
#analysis.setRangeClass([i-1 for i in class_interest])
#all_data,all_trgt,trgt_sparse = analysis.getData()
weight = analysis.getClassesWeight(flag=args.weight)
print 'weight: {0}'.format(weight)

analysis.setTrainParameters(n_inits=args.init,neuron = args.neurons,hidden_activation='tanh',class_weight_flag=True,
                            output_activation='tanh',classifier_output_activation = 'tanh',n_epochs=args.epochs,
                            n_folds=args.fold, patience=30,batch_size=args.batch, verbose=False,pcd=ipcd,
                            optmizerAlgorithm=args.optmizer, metrics=['accuracy','categorical_accuracy',ind_SP], loss=args.loss)
path_result=analysis.getBaseResultsPath()


m_time = time.time()

all_data,all_trgt,trgt_sparse = analysis.getData()

eachFold = args.each_fold -1

if not ipcd==None:
    pcd1 = Pmod(path=results_path,n_components=args.pcd,n_folds=args.fold)
    pcd_params = TrnParams(learning_rate=0.001,batch_size=512,init=args.init,optimizers='adam')
    pcd1.train(all_data,all_trgt,eachFold,trn_params=pcd_params,flag_class_weight=True)
    all_data = pcd1.getDataCompact()

from multiprocessing import Pool
from functools import partial



if ipcd == None and eachFold == -1 :
    def train(ifold):
        analysis.train_each(all_data,all_trgt,analysis.CVO,analysis.trn_params,ifold,flag_class_weight=args.weight)
    p = multiprocessing.Pool(processes=num_processes)
    result = p.map(train,range(args.fold))

    p.close()
    p.join()


elif ipcd == None:
    analysis.train_each(all_data,all_trgt,analysis.CVO,analysis.trn_params,eachFold)
elif eachFold == -1:
    def trainPCD(ifold):
        analysis.train_each(all_data,all_trgt,analysis.CVO,analysis.trn_params,ifold,)
    p = multiprocessing.Pool(processes=num_processes)
    result = p.map(train,range(args.fold))

    p.close()
    p.join()
else:
    print 'Data size: {0}x{1}'.format(all_data.shape[0],all_data.shape[1])
    analysis.train_each(all_data,all_trgt,analysis.CVO,analysis.trn_params,eachFold)
    analysis.eff(all_data,analysis.all_trgt,analysis.CVO,analysis.trn_params,fold=eachFold)


       
        
m_time = time.time()-m_time

telegram_send.send(messages=["O seu treinamento com os seguintes parametros acabou de terminar!",
                             str(args),
                             "Ele levou "+str(datetime.timedelta(seconds=m_time))+" ao total",
                             "tenha um bom dia!!!"])

print 'Time to train data : '+str(datetime.timedelta(seconds=m_time))
