import os
import sys
import time
sys.path.insert(0,'..')

from classificationConfig import CONFIG
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
#!/usr/bin/python

import os
import sys

sys.path.insert(0,'/home/venancio/Workspace/SonarAnalysis/Packages/Classification')




import argparse
#Argument Parse config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-f", "--fold", default=0,type=int, help="Select number of neurouns in Neural Network")
parser.add_argument("-p", "--pcd", default=70,type=int, help="Select number of neurouns in Neural Network")

args = parser.parse_args()
import multiprocessing

from NeuralNetworkModule import NeuralNetworkModule as nnm

from Functions.MetricCustom import ind_SP

num_processes = multiprocessing.cpu_count()

analysis_name = 'Classification'

data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']
class_interest = [1,2,3,4]#[6,7,9,16,21,23]

nums_componets = None

#DataModule

analysis = nnm(analysis_name="NeuralNetwork",development_flag=True,development_events=400,verbose=False)
#analysis.infoData()
#analysis.setRangeClass([i-1 for i in class_interest])
#analysis.balanceData()
analysis.infoData()
weight_class = analysis.getClassesWeight(flag=True)
all_data,all_trgt,_ = analysis.getData()

#PCD Module
from PCD2 import TrnParams
from PCDModule import PCDModule as Pmod

nums_componets = args.pcd
list_all_data = {}

pcd1 = Pmod(path=results_path,n_components=nums_componets,n_folds=10)

# Train Module
analysis.setTrainParameters(n_inits=10,neuron = 150,hidden_activation='tanh',class_weight_flag=True,
                            output_activation='tanh',classifier_output_activation = 'tanh',n_epochs=1000,
                            n_folds=10, patience=30,batch_size=512, verbose=False,pcd=nums_componets,
                            optmizerAlgorithm='sgd', metrics=['accuracy','categorical_accuracy',ind_SP], loss='mean_squared_error')

for ifold in [args.fold]:
    pcd_params = TrnParams(learning_rate=0.001,batch_size=512,init=10)
    pcd1.train(all_data,all_trgt,ifold,trn_params=pcd_params,flag_class_weight=True)
    all_data_reduc = pcd1.getDataCompact()
    analysis.eff(all_data_reduc,analysis.all_trgt,analysis.CVO,analysis.trn_params,fold=ifold,score='recall')
 
