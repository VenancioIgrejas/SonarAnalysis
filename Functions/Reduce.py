import os
import time
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from ClassificationAnalysis import NeuralClassification
from Functions import PreProcessing
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import FuncFormatter
import seaborn as sn
 

class ReduceData(NeuralClassification):
    def PCData(self,data,trgt,
               fold=0,n_components=1,trn_params=None,use_activation=False):
        """
           PCD independent for reduce dimensionality reduction 
        
        """
        print 'PCD function'
        file_name = '%s/%s_%s_PCDdata_comp_%i_fold_%i.jbl'%(self.preproc_path,
                                                            self.trn_info.date,
                                                            self.name,n_components,fold)
        
        #model_file = '%s/%s_%s_PCDdata_comp_%i_fold_%i_model.h5'%(self.preproc_path,
        #                                                   self.trn_info.date,
        #                                                   self.name,n_components,fold)

        if not os.path.exists(file_name):
            train_id, test_id = self.trn_info.CVO[fold]
            
            [data_preproc, trgt_preproc]=self.preprocess(data,trgt,self.trn_info,fold)
            trgt_sparse = np_utils.to_categorical(trgt_preproc.astype(int))
            
            pcdc = PreProcessing.PCDIndependent(n_components)
            model = pcdc.fit(data_preproc, trgt_sparse, train_id, test_id, trn_params)
            
            data_PCD = pcdc.transform(data,use_activation)
            
            joblib.dump([data_PCD,trgt],file_name,compress=9)
            
            
        else:
            
            [data_PCD,trgt] = joblib.load(file_name)
        
        return [data_PCD,trgt]
    
    def confMatrix_sb(self,data,trgt,trn_info=None,figX=30,figY=30, class_labels=None, n_neurons=1,fold=0):
        
        print 'NeuralClassication analysis analysis conf mat function'
        file_name = '%s/%s_%s_analysis_model_output_fold_%i_neurons_%i.jbl'%(self.anal_path,
                                                                             self.trn_info.date,
                                                                             self.name,fold,
                                                                             n_neurons)
        output = None
        if not os.path.exists(file_name):
            [model,trn_desc] = self.train(data,trgt,trn_info=trn_info,n_neurons=n_neurons,fold=fold)
            output = model.predict(data)
            joblib.dump([output],file_name,compress=9)
        else:
            [output] = joblib.load(file_name)
        
        train_id, test_id = self.trn_info.CVO[fold]
        
        num_output = np.argmax(output,axis=1)
        num_tgrt = np.argmax(trgt,axis=1)

        cm = confusion_matrix(num_tgrt[test_id], num_output[test_id])
        
        fig, ax = plt.subplots(figsize=(figX,figY),nrows=1, ncols=1)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(cm_normalized)
    
        uniform_data = np.random.rand(10, 12)
        fmt = lambda x,pos: '{:.0%}'.format(x)
        ax = sn.heatmap(df_cm, annot=True, fmt='.0%',
                        cbar_kws={'format': FuncFormatter(fmt)},
                        ax=ax,cmap='Greys',vmin=0, vmax=1)
        ax.set_title('Confusion Matrix',fontweight='bold',fontsize=40)
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels,rotation=45, va='top')

        ax.set_ylabel('True Label',fontweight='bold',fontsize=40)
        ax.set_xlabel('Predicted Label',fontweight='bold',fontsize=40)
        
        return fig,cm