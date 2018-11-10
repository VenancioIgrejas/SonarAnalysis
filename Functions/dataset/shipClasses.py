import os
from sklearn.preprocessing import LabelEncoder
from envConfig import CONFIG
from Functions.util import inverse_dict

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
                 spectrum_bins_left=400):
        #super(LoadData, self).__init__()

        self.database = database

        self.DATA_PATH = CONFIG['OUTPUTDATAPATH']

        if data_file is None:
            data_file = "lofar_data_withoutNaN_file_fft_%i_decimation_%i_spectrum_left_%i.jbl"%(self.n_pts_fft,
                                                                                    self.decimation_rate,
                                                                                    self.spectrum_bins_left)

        data_path = os.path.join(self.DATA_PATH,self.database,data_file)

        if not os.path.exists(data_path):
            print 'No Files in %s/%s\n'%(self.DATA_PATH,
                                         self.database)
        else:
            #Read lofar data
            [data,trgt,class_labels] = joblib.load(data_path)


            m_time = time.time()-m_time

            print '[+] Time to read data file: '+str(m_time)+' seconds'


            # correct format

            self.all_data = data
            self.PCD_data = data
            self.all_trgt = trgt

            self.class_label = class_labels

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
        return [self.all_data, self.all_trgt, self.trgt_sparse]

    def getClassLabels(self):
        return self.class_labels

    def setRangeClass(self,class_range):

        import pandas as pd


        list_class_label = [self.class_labels[iclass] for iclass in class_range]
        list_class_events = [pd.DataFrame(self.all_data[self.all_trgt==iclass,:]) for iclass in class_range]
        list_trgt_events = [pd.DataFrame(value*np.ones(self.all_data[self.all_trgt==iclass,:].shape[0]))
                            for value, iclass in enumerate(class_range)]

        self.all_data = pd.concat(list_class_events).values
        self.all_trgt = pd.concat(list_trgt_events)[0].values

        self.class_labels = list_class_label

    def mapClasses(map_class):
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

        le = LabelEncoder()
        le.fit(self.class_labels)

        y_label = le.inverse_transform(self.all_trgt)

        inv_map_class = inverse_dict(map_class)

        #change all old class label in new class label in target vector
        y_new_label = map(lambda x:inv_map_class[x],y_label)

        le_new = LabelEncoder()

        self.all_trgt = le_new.fit_transform(y_new_label)
        self.class_labels = le_new.classes_

        return self
