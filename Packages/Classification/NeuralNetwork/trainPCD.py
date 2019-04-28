#!/usr/bin/python

#ONLY USE THIS CODE IF YOU ALREADY TRAINED PCD COMPONENTS 
import os
import sys
import telegram_send

sys.path.insert(0,'/home/venancio/Workspace/SonarAnalysis/Packages/Classification')




import argparse
#Argument Parse config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-p", "--pcd",default=2,type=int,help= "number of pcds ")
parser.add_argument("-i", "--init",default=10,type=int, help="Select number of inicialization in Neural Network")
parser.add_argument("-f", "--fold",default=10,type=int, help= "Select number of folds in Neural Network")
parser.add_argument("-w", "--weight",default=True,type=bool, help= "Flag for balance dataset inside of train function in Keras")
parser.add_argument("-c", "--countCPU",default=-1,type=int,help= "number of cores for multiprocessing, default 8 cores")
parser.add_argument("--each-fold",default=0,type=int,help= "choose which fold training with pcd")



args = parser.parse_args()
print args

import time
import datetime

from classificationConfig import CONFIG
import numpy as np

import multiprocessing
from functools import partial
from Functions.MetricCustom import ind_SP
from contextlib import contextmanager
from PCD2 import TrnParams
from PCDModule import PCDModule as Pmod

from NeuralNetworkModule import NeuralNetworkModule as nnm

if args.countCPU == -1:
    num_processes = multiprocessing.cpu_count()
else:
    num_processes = args.countCPU

analysis_name = 'Classification'
data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']

class_interest = [1,2,3,4]#[6,7,9,16,21,23]

analysis = nnm(analysis_name="NeuralNetwork",development_flag=False,development_events=5000,verbose=False)
#analysis.setRangeClass([i-1 for i in class_interest])
all_data,all_trgt,trgt_sparse = analysis.getData()

#PCD Module


pcd1 = Pmod(path=results_path,n_components=args.pcd,n_folds=args.fold)
pcd_params = TrnParams(learning_rate=0.001,batch_size=512,init=args.init,optimizers='adam')


eachFold = args.each_fold -1 


m_time = time.time()


if eachFold == -1:
    def train(ifold):
        pcd1.train(all_data,all_trgt,ifold,trn_params=pcd_params,flag_class_weight=True)
    #for ifold in range(args.pcd):
    p = multiprocessing.Pool(processes=num_processes)
    result = p.map(train,range(args.fold))
        #t = multiprocessing.Process(target=pcd1.train,args=(all_data,all_trgt,ifold,pcd_params,args.weight))
        #t.start()
    p.close()
    p.join()
        #p.apply_async(pcd1.train,args=(all_data,all_trgt,ifold,pcd_params,args.weight))

else:
    pcd1.train(all_data,all_trgt,eachFold,trn_params=pcd_params,flag_class_weight=True)



#analysis.setTrainParameters(n_inits=args.init,neuron = args.neurons,hidden_activation='tanh',class_weight_flag=True,
#                            output_activation='tanh',classifier_output_activation = 'tanh',n_epochs=args.epochs,
#                            n_folds=args.fold, patience=30,batch_size=512, verbose=False,pcd=ipcd,
#                            optmizerAlgorithm=args.optmizer, metrics=['accuracy','categorical_accuracy',ind_SP], loss=args.loss)
#path_result=analysis.getBaseResultsPath()


#m_time = time.time()

#ifold = args.each_fold

#modePCD = pcdi(args.pcd)
#modePCD.loadPCDs(path=os.path.join(analysis.getBaseResultsPath(),'{0}_fold_PCDs'.format(ifold)),n_components=args.pcd)
    
#dataPCD = modePCD.transform(analysis.all_data)
#analysis.train_each(dataPCD,analysis.all_trgt,analysis.CVO,analysis.trn_params,fold=ifold)
    
#analysis.analysis_train_CSV(fold=ifold)
#analysis.confMatrix_sb_each(dataPCD,analysis.all_trgt,analysis.CVO,analysis.trn_params,fold=ifold)

        
m_time = time.time()-m_time

telegram_send.send(messages=["O seu treinamento com os seguintes parametros acabou de terminar!",
                             str(args),
                             "Ele levou "+str(datetime.timedelta(seconds=m_time))+" ao total",
                             "tenha um bom dia!!!"])

print 'Time to train data : '+str(datetime.timedelta(seconds=m_time))
