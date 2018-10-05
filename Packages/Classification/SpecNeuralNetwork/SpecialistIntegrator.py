import numpy as np
import os
import sys
from IntegratorPath import IntegratorPath
import pandas as pd

from Functions.StatFunctions import sp

class SpecialistIntegrator(IntegratorPath):
    """docstring for SpecialistClass."""
    def __init__(self,resultPath,list_specialist,folds):
        super(SpecialistIntegrator, self).__init__(resultPath,folds)
        self.list_specialist = list_specialist
        self.dic_output = {}
        self.dic_trgt = {}
        self.dic_model = {}
        self.dic_test_pre = {}
        self.list_num_spec = [ispec.spec_num for ispec in list_specialist]
            
    
    
    def check_train_all_spec(self):
        flag = True
        for ispec in self.list_specialist:
            spec_num = ispec.spec_num
            if not ispec.check_train():
                flag = False
        return flag
    
    def generate_dic(self,data,trgt,fold,verbose=False):
        dic_output = {}
        dic_trgt = {}
        dic_model = {}
        
        for ispec in self.list_specialist:
            spec_num = ispec.spec_num
            if verbose is True:
                print "Specialist {0} is loading".format(spec_num)
            cvo = ispec.set_cross_validation(all_trgt=trgt)
            train_id,test_id = cvo[fold]
            dic_model[spec_num] = ispec.train(data,trgt,fold)
            dic_trgt[spec_num] = ispec.prepare_target(trgt)[test_id]
            dic_output[spec_num] = ispec.output(data,trgt,fold)[test_id]
        
        return dic_output,dic_trgt,dic_model
    
    def fit_by_max_criteria(self,data,trgt,fold):
        dic_weight={}
        dic_output,dic_trgt,dic_model = self.generate_dic(data,trgt,fold)
        output_conc = [dic_output[i] for i in self.list_num_spec]
        M = np.dstack(output_conc)[0]
        M_ = np.linalg.pinv(M) #gerar a pseudo matriz inversa
        for i in self.list_num_spec:
            dic_weight[i] = np.dot(M_,dic_trgt[i])
            #np.savetxt(X=dic_weight[i],fname=self.weight_file(fold,i), delimiter=',')
        return dic_weight,dic_model
    
    def predict(self,X_train,X_test,y_train,y_test,fold,predict_form = 'maxcriteria',verbose=True):
        
        if not self.check_train_all_spec():
            return -1
        
        preproc_data_test = {}
        for ispec in self.list_specialist:
            spec_num = ispec.spec_num
            preproc_data_test[spec_num],_,_ = ispec.preprocess(X_test,y_test,fold)
            
        if predict_form == 'maxcriteria':
            dic_weight,dic_model = self.fit_by_max_criteria(X_train,y_train,fold)
            M_tst = np.hstack([dic_model[i].predict(preproc_data_test[i]) for i in self.list_num_spec])
            output_sparce = np.dstack([np.dot(M_tst,dic_weight[i]) for i in self.list_num_spec])[0]
            output = np.argmax(output_sparce,axis=1)
            if verbose is True:
                print "output of fold {0} load finished".format(fold)
            return output
    
    
    
    def create_eff_sp_dataframe(self,num_analyze,eff_fold=None,names=['fold','classe']):
        """ create dataframe of Classe Especialist efficiency and SP function.
            
            # Arguments
                eff_fold: list, efficient of each class for each fold.
                num_analyze: Integer, number of analyze for save in a specific folder.
                names: list, name of each index
            
            
            # Returns: list of dataframe
                 output of each class multindex,
                 output of each class without multindex,
                 SP of each fold
        """
        
        folder = self.analysis_fold(num_analyze)
        
        index = pd.MultiIndex.from_product([range(len(eff_fold)), range(eff_fold[0].shape[0])],
                                           names=names)
  
        self._df_classes_params(path=folder+'/paramns_spClass.csv')
    
        output_class_specialist = pd.DataFrame({'eff':np.concatenate(eff_fold)},index=index)
        
        file = folder+'/paramns_spClass.csv'
        if os.path.exists(file):
            output_class_specialist = pd.read_csv(file,index_col=names)
        else:
            output_class_specialist = pd.DataFrame({'eff':np.concatenate(eff_fold)},index=index)
            output_class_specialist.to_csv(file)
        
        df_without_index = pd.read_csv(file)
          
        output_sp = output_class_specialist.groupby(['fold']).apply(lambda x:sp(x))
        output_sp.columns = ['sp']
        output_sp.to_csv(folder+'/SP_spClass.csv')
        
        return output_class_specialist, df_without_index, output_sp
    
    def _df_classes_params(self,path=None):
        list_df = []
        for ispec in self.list_specialist:
            num_spec = ispec.spec_num
            dict_params = ispec.trnparams.__dict__
            each_spec = pd.DataFrame(dict_params,
                                     index=['Classe{0}'.format(num_spec+1)])
            
            list_df.append(each_spec)
        
        df_concat = pd.concat(list_df)
        
        if not path is None:
            df_concat.to_csv(path,header=True)
        
        return df_concat
        
        