import os
import sys
import io
import time
import numpy as np

import pandas as pd

from scipy.io import loadmat

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

#from envConfig import CONFIG
from Functions.util import inverse_dict,pydata


class MountData(object):
    """docstring for MountData."""
    def __init__(self,data_file=None,database='24classes',number_of_classes = 24 ,n_pts_fft=1024, decimation_rate=3,
                 spectrum_bins_left=400,class_label=None):
        #super(MountData, self).__init__()

        self.DATA_PATH = os.environ['OUTPUTDATAPATH'] #CONFIG['OUTPUTDATAPATH']

        self.database = database
        self.number_of_classes = number_of_classes
        self.n_pts_fft = n_pts_fft
        self.decimation_rate = decimation_rate
        self.spectrum_bins_left = spectrum_bins_left

        classes = []

        for num in range(number_of_classes):
            data_path = os.path.join(self.DATA_PATH,self.database,'Class{0}'.format(num+1))
            self.file_lofar = "lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i"%(self.n_pts_fft,
                                                                                    self.decimation_rate,
                                                                                    self.spectrum_bins_left)

            classes.append(loadmat(os.path.join(data_path,self.file_lofar)))

        self.classes = classes


        
        #treating the data
        alldata={}
        for i in range(number_of_classes):
            shape_classes = classes[i][sorted(classes[i].keys())[-1]].shape
            if shape_classes[0] == 1:
                data = [pydata(classes[i],eachCell) 
                        for eachCell in range(classes[i][sorted(classes[i].keys())[-1]].shape[1])]
                alldata[i] = pd.concat(data)
            else:
                alldata[i] = pydata(classes[i])

        self.data = pd.concat([pd.DataFrame(alldata[key]) for key in range(number_of_classes)]).values
        #creating trgt of data
        eachtrgt = []

        for i in range(number_of_classes):
            numRow = alldata[i].shape[0]
            eachtrgt.append(i*np.ones(shape=(numRow,)))

        self.trgt = np.hstack(eachtrgt)

        #creating labels classes
        if  class_label is None:
            self.class_label = ['Class{0:02d}'.format(i+1) for i in range(number_of_classes)]
        else:
            self.class_label = class_label

    def to_save(self,path=None,type_file='jbl'):
        if not type_file in ['jbl','csv']:
            raise ValueError("only 'jbl' and 'csv' extensions")    

        if path is None:
            file_path = os.path.join(self.DATA_PATH,self.database,self.file_lofar + '.' + type_file)
            print "save the dataset in {0} file".format(file_path)

            if type_file=='csv':
                df_dt = pd.DataFrame(self.data)
                df_trgt = pd.DataFrame({'target':self.trgt})
                df_classes = pd.DataFrame({'labels':self.class_label})

                pd.concat([df_dt,df_trgt,df_classes],axis=1).to_csv(file_path,index=False)
            else:
                joblib.dump([self.data,self.trgt,self.class_label],file_path,compress=9)

        else:
            print "save the dataset in {0} file".format(path + '.' + type_file)
            joblib.dump([self.data,self.trgt,self.class_label],path + '.' + type_file,compress=9)

    def getData(self):
        return self.data,self.trgt,self.class_label


class LoadData(object):
    """ Load Dataset for analysis
    ----------
    data_file : {string}
               name of file inside database path

    database: {string}
               database of result path

    n_pts_fft: {int}
                number of frequency bins in fft(Fast Fourier transform )

    decimation_rate: {int}

    spectrum_bins_left: {int}


    """
    def __init__(self,data_file=None,database='24classes', n_pts_fft=1024, decimation_rate=3,
                 spectrum_bins_left=400,dev=False,dev_size=100):
        #super(LoadData, self).__init__()

        m_time = time.time()

        self.database = database
        self.n_pts_fft = n_pts_fft
        self.decimation_rate = decimation_rate
        self.spectrum_bins_left = spectrum_bins_left
        self.dev_size=dev_size
        self.dev = dev
        self.datapath_for_load = None


        self.DATA_PATH = os.environ['OUTPUTDATAPATH'] #CONFIG['OUTPUTDATAPATH']

        if data_file is None:
            data_file = "lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i.jbl"%(self.n_pts_fft,
                                                                                    self.decimation_rate,
                                                                                    self.spectrum_bins_left)

        data_path = os.path.join(self.DATA_PATH,self.database,data_file)
        print(data_path)
        if not os.path.exists(data_path):
            print 'No Files in %s/%s\n'%(self.DATA_PATH,
                                         self.database)
        else:
            #Read lofar data
            [data,trgt,class_labels] = joblib.load(data_path)


            m_time = time.time()-m_time

            print '[+] Time to read data file: '+str(m_time)+' seconds'

            if self.dev:
                data, trgt = self._dev_dataset(data, trgt,size_samples=self.dev_size)

            # correct format

            self.all_data = data
            self.PCD_data = data
            self.all_trgt = trgt


            self.class_labels = class_labels

    def _dev_dataset(self,data,trgt,size_samples=100):
        """ reduce each class for only 100 random samples"""

        print("WARNING: each class was reduced to {0} random samples".format(size_samples))

        df_alldata=[]

        for iclass in np.unique(trgt):
            idata = data[trgt==iclass]
            sample = np.random.choice(idata.shape[0],size_samples,replace=False)
            df_alldata.append(pd.DataFrame(idata[sample]))

        return pd.concat(df_alldata).values, np.array(sorted(range(24)*size_samples))

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

    def getData(self):
        return [self.all_data, self.all_trgt]

    def getClassLabels(self):
        return self.class_labels

    def setRangeClass(self,class_range):

        if not isinstance(class_range,(list,np.array)):
            raise ValueError("type of classes is {0} but expected list type".format(type(class_range)))

        list_class_label = [self.class_labels[iclass] for iclass in class_range]
        list_class_events = [pd.DataFrame(self.all_data[self.all_trgt==iclass,:]) for iclass in class_range]
        list_trgt_events = [pd.DataFrame(value*np.ones(self.all_data[self.all_trgt==iclass,:].shape[0]))
                            for value, iclass in enumerate(class_range)]

        self.all_data = pd.concat(list_class_events).values
        self.all_trgt = pd.concat(list_trgt_events)[0].values

        self.class_labels = list_class_label

    def mapClasses(self,map_class):
        """ map the class labels of dataset in new class labels
            changing the target of dataset according to new labels
        ----------
        map_class : {dict of list/string}
                    map old classes in classes labels in new classe

                    eg. class_label = ['class1','class2','class3','class4','class5']
                        map_class = {'classA':['class1','class2'],
                                     'classB':['class3','class4'],
                                     'classC':'class5'}
                        new_class_label = ['classA','classB','classC']
        """
        if not isinstance(map_class,dict):
            raise ValueError("{0} isn't dict type".format(type(map_class)))

        if not isinstance(self.all_trgt,list):
            self.all_trgt = self.all_trgt.tolist()

        le = LabelEncoder()
        le.fit(self.class_labels)

        #if not le.classes_ == self.class_labels:
        #    raise ValueError("classes labels need to be sorted")

        y_label = le.inverse_transform(map(int,self.all_trgt))

        inv_map_class = inverse_dict(map_class)

        #change all old class label in new class label in target vector
        y_new_label = map(lambda x:inv_map_class[x],y_label)

        le_new = LabelEncoder()

        self.all_trgt = le_new.fit_transform(y_new_label)
        self.class_labels = le_new.classes_.tolist()

        return self
