
import os
import sys
sys.path.insert(0,'..')

from classificationConfig import CONFIG

import time
import string
import multiprocessing
import numpy as np

from sklearn.externals import joblib

from keras.utils import np_utils

from Functions import TrainParameters as trnparams
from Functions import DataHandler as dh

class NeuralNetworkAnalysis(object):
    def __init__(self, analysis_name='', database='24classes', n_pts_fft=1024, decimation_rate=3, 
                 spectrum_bins_left=400,development_flag=False,development_events=10,verbose = True):
        
        m_time = time.time()
        
        #Analysis Caracteristcs
        self.analysis_name = analysis_name
        
        #for multiprocessing purpose
        
        self.num_processes = multiprocessing.cpu_count()
        
        #Enviroment variables
        
        self.DATA_PATH = CONFIG['OUTPUTDATAPATH']
        
        self.RESULTS_PATH = CONFIG['PACKAGE_NAME']
        
        # paths to export results
        self.base_results_path = os.path.join(self.RESULTS_PATH,self.analysis_name)
        
        # Database caracteristics
        self.database = database
        self.n_pts_fft = n_pts_fft
        self.decimation_rate = decimation_rate
        self.spectrum_bins_left = spectrum_bins_left
        self.development_flag = development_flag
        self.development_events = development_events
        
        self.verbose = verbose
        
        # Check if LofarData has already been created...
        data_file = os.path.join(self.DATA_PATH,self.database,
                                 "lofar_data_withoutNaN_file_fft_%i_decimation_%i_spectrum_left_%i.jbl"%(self.n_pts_fft,
                                                                                                         self.decimation_rate,
                                                                                                         self.spectrum_bins_left)
                                )
        
        if not os.path.exists(data_file):
            print 'No Files in %s/%s\n'%(self.DATA_PATH,
                                         self.database)
        else:
            #Read lofar data
            [data,trgt,class_labels] = joblib.load(data_file)


            m_time = time.time()-m_time
            
            print '[+] Time to read data file: '+str(m_time)+' seconds'

            
            # correct format
            
            self.all_data = data
            self.PCD_data = data
            self.all_trgt = trgt
            self.trgt_sparse = np_utils.to_categorical(self.all_trgt.astype(int))

            
            # Process data
            
            # unbalanced data to balanced data with random data creation of small classes

            
            # Same number of events in each class
            
            self.qtd_events_biggest_class = 0
            
            self.biggest_class_label = ''
            
            
            # Define the class labels with ascii letters in uppercase
            #self.class_labels = list(string.ascii_uppercase)[:self.trgt_sparse.shape[1]]
            self.class_labels = class_labels
            
            for iclass, class_label in enumerate(self.class_labels):
                if sum(self.all_trgt==iclass) > self.qtd_events_biggest_class:
                    self.qtd_events_biggest_class = sum(self.all_trgt==iclass)
                    self.biggest_class_label = class_label
                if self.verbose:
                    print "Qtd event of %s is %i"%(class_label,sum(self.all_trgt==iclass))
            if self.verbose:
                print "\nBiggest class is %s with %i events"%(self.biggest_class_label,self.qtd_events_biggest_class)

                
            

    def balanceData(self):
        balanced_data={}
        balanced_trgt={}
    
        from Functions import DataHandler as dh
    
        m_datahandler = dh.DataHandlerFunctions()
    
        for iclass, class_label in enumerate(self.class_labels):
            if self.development_flag:
                class_events = self.all_data[self.all_trgt==iclass,:]
                if len(balanced_data) == 0:

                    if self.all_data[self.all_trgt==iclass,:].shape[0] < self.development_events:
                        class_events = self.all_data[self.all_trgt==iclass,:]
                        balanced_data = m_datahandler.CreateEventsForClass(class_events,
                                                                           self.development_events-(len(class_events)))
                        balanced_trgt = (iclass)*np.ones(self.development_events)
                    else:
                        
                        balanced_data = class_events[0:self.development_events,:]
                        balanced_trgt = (iclass)*np.ones(self.development_events)
                else:
                
                    if self.all_data[self.all_trgt==iclass,:].shape[0] < self.development_events:
                        class_events = self.all_data[self.all_trgt==iclass,:]
                        created_events = (m_datahandler.CreateEventsForClass(self.all_data[self.all_trgt==iclass,:],
                                                                             self.development_events - (len(class_events))))
                        balanced_data = np.append(balanced_data,created_events,axis=0)
                        balanced_trgt = np.append(balanced_trgt,(iclass)*np.ones(created_events.shape[0]),axis=0)
                    else:
                        class_events = self.all_data[self.all_trgt==iclass,:]
                        balanced_data = np.append(balanced_data,                                              
                                                  class_events[0:self.development_events,:],                                              
                                                  axis=0)
                        balanced_trgt = np.append(balanced_trgt,(iclass)*np.ones(self.development_events))
            else:
                if len(balanced_data) == 0:
                    class_events = self.all_data[self.all_trgt==iclass,:]
                    balanced_data = m_datahandler.CreateEventsForClass(
                        class_events,self.qtd_events_biggest_class-(len(class_events)))
                    balanced_trgt = (iclass)*np.ones(self.qtd_events_biggest_class)
                else:
                    class_events = self.all_data[self.all_trgt==iclass,:]
                    created_events = (m_datahandler.CreateEventsForClass(self.all_data[self.all_trgt==iclass,:],
                                                                     self.qtd_events_biggest_class-
                                                                     (len(class_events))))
                    balanced_data = np.append(balanced_data,created_events,axis=0)
                    balanced_trgt = np.append(balanced_trgt,(iclass)*np.ones(created_events.shape[0]),axis=0)
    
        self.all_data = balanced_data
        self.PCD_data = balanced_data
        self.all_trgt = balanced_trgt
        # turn targets in sparse mode
    
    def getData(self):
        return [self.all_data, self.all_trgt, self.trgt_sparse]

    def getPCDData(self):
        return [self.PCD_data, self.all_trgt, self.trgt_sparse]    
    
    def setRangeClass(self,class_range):
        
        import pandas as pd
        
        
        list_class_label = [self.class_labels[iclass] for iclass in class_range]
        list_class_events = [pd.DataFrame(self.all_data[self.all_trgt==iclass,:]) for iclass in class_range]
        list_trgt_events = [pd.DataFrame(value*np.ones(self.all_data[self.all_trgt==iclass,:].shape[0]))
                            for value, iclass in enumerate(class_range)]
        
        self.all_data = pd.concat(list_class_events).values
        self.all_trgt = pd.concat(list_trgt_events)[0].values
        
        self.class_labels = list_class_label
        self.trgt_sparse = np_utils.to_categorical(self.all_trgt.astype(int))
    
    def getPCDData(self):
        return [self.PCD_data, self.all_trgt, self.trgt_sparse]
    
    def infoEachData(self,class_specific):
        index_class = self.getClassLabels().index(class_specific)
        specific_data = self.all_data[self.all_trgt==index_class,:]
        print "analysis of {0}: \nsize: {1}x{2} -- maxValue: {3} -- minValue: {4}".format(class_specific,
                                                                                          specific_data.shape[0],
                                                                                          specific_data.shape[1],
                                                                                          np.amax(np.array(specific_data)),
                                                                                          np.amin(np.array(specific_data)))
    
    def infoData(self):
        for iclass in self.getClassLabels():
            self.infoEachData(iclass)
        print "end of class analysis"
    
    def getClassLabels(self):
        return self.class_labels
    
    def getClassesWeight(self,flag):
        
        if flag == False:
            return None

        dicWeight={}
        
        list_size_class  = np.array([self.all_data[self.all_trgt==iclass,:].shape[0] 
                                     for iclass , eachClass in enumerate(self.class_labels)]).astype('double')
        
        list_multiplic = list_size_class[list_size_class.argmin()]/list_size_class
        
        
        for index, value in enumerate(list_multiplic):
            dicWeight[index] = value
        
        return dicWeight
        
                                
                