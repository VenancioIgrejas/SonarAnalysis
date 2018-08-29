"""
    This file contents some classification analysis functions
"""

import os
import platform
import numpy as np
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.model_selection import StratifiedKFold

class TrnParams(object):
    """
        Basic class
    """
    def __init__(self, analysis="None"):
        self.analysis = analysis
        self.params = None

    def save(self, name="None"):
        joblib.dump([self.params],name,compress=9)

    def load(self, name="None"):
        [self.params] = joblib.load(name)

    def printParams(self):
        for iparameter in self.params:
            print iparameter + ': ' + str(self.params[iparameter])


# classification

def ClassificationFolds(folder, n_folds=2, trgt=None, dev=False, verbose=False):

    if n_folds < 2:
        print 'Invalid number of folds'
        return -1

    if not dev:
        file_name = '%s/%i_folds_cross_validation.jbl'%(folder,n_folds)
    else:
        file_name = '%s/%i_folds_cross_validation_dev.jbl'%(folder,n_folds)

    if not os.path.exists(file_name):
        if verbose:
            print "Creating %s"%(file_name)

        if trgt is None:
            print 'Invalid trgt'
            return -1

        CVO = cross_validation.StratifiedKFold(trgt,n_folds)
        CVO = list(CVO)
        joblib.dump([CVO],file_name,compress=9)
    else:
        if verbose:
            print "File %s exists"%(file_name)
        [CVO] = joblib.load(file_name)

    return CVO

class NeuralClassificationTrnParams(TrnParams):
    """
        Neural Classification TrnParams
    """

    def __init__(self,
                 n_inits=2,
                 norm='mapstd',
                 verbose=False,
                 train_verbose=False,
                 n_epochs=10,
                 learning_rate=0.001,
                 beta_1 = 0.9,
                 beta_2 = 0.999,
                 epsilon = 1e-08,
                 learning_decay=1e-6,
                 momentum=0.3,
                 nesterov=True,
                 patience=5,
                 batch_size=4,
                 hidden_activation='tanh',
                 output_activation='tanh',
                 metrics=['accuracy'],
                 loss='mean_squared_error',
                 optmizerAlgorithm='SGD'
                ):
        self.params = {}

        self.params['n_inits'] = n_inits
        self.params['norm'] = norm
        self.params['verbose'] = verbose
        self.params['train_verbose'] = train_verbose

        # train params
        self.params['n_epochs'] = n_epochs
        self.params['learning_rate'] = learning_rate
        self.params['beta_1'] = beta_1
        self.params['beta_2'] = beta_2
        self.params['epsilon'] = epsilon
        self.params['learning_decay'] = learning_decay
        self.params['momentum'] = momentum
        self.params['nesterov'] = nesterov
        self.params['patience'] = patience
        self.params['batch_size'] = batch_size
        self.params['hidden_activation'] = hidden_activation
        self.params['output_activation'] = output_activation
        self.params['metrics'] = metrics
        self.params['loss'] = loss
        self.params['optmizerAlgorithm'] = optmizerAlgorithm

    def get_params_str(self):
        param_str = ('%i_inits_%s_norm_%i_epochs_%i_batch_size_%s_hidden_activation_%s_output_activation'%
                     (self.params['n_inits'],self.params['norm'],self.params['n_epochs'],self.params['batch_size'],
                      self.params['hidden_activation'],self.params['output_activation']))
        for imetric in self.params['metrics']:
            param_str = param_str + '_' + imetric
      
        param_str = param_str + '_metric_' + self.params['loss'] + '_loss'
        return param_str

class NeuralNetworkTrnParams(TrnParams): 
    
    def __init__(self,
                 n_inits=2,
                 folds=2,
                 neuron = 10,
                 norm='mapstd',
                 verbose=False,
                 train_verbose=False,
                 n_epochs=10,
                 learning_rate=0.001,
                 beta_1 = 0.9,
                 beta_2 = 0.999,
                 epsilon = 1e-08,
                 learning_decay=1e-6,
                 momentum=0.3,
                 nesterov=True,
                 patience=5,
                 batch_size=4,
                 hidden_activation='tanh',
                 output_activation='tanh',
                 classifier_output_activation = 'softmax',
                 metrics=['accuracy'],
                 pcd=None,
                 class_weight_flag=False,
                 loss='mean_squared_error',
                 optmizerAlgorithm='sgd'):
        
        self.params = {}
        
        
        self.params['n_inits'] = n_inits
        self.params['folds'] = folds
        self.params['neuron'] = neuron
        self.params['norm'] = norm
        self.params['verbose'] = verbose
        self.params['train_verbose'] = train_verbose

        # train params
        self.params['n_epochs'] = n_epochs
        self.params['learning_rate'] = learning_rate
        self.params['beta_1'] = beta_1
        self.params['beta_2'] = beta_2
        self.params['epsilon'] = epsilon
        self.params['learning_decay'] = learning_decay
        self.params['momentum'] = momentum
        self.params['nesterov'] = nesterov
        self.params['patience'] = patience
        self.params['batch_size'] = batch_size
        self.params['hidden_activation'] = hidden_activation
        self.params['output_activation'] = output_activation
        self.params['metrics'] = metrics
        self.params['loss'] = loss
        self.params['optmizerAlgorithm'] = optmizerAlgorithm
        self.params['class_weight_flag'] = class_weight_flag
        if pcd=='None':
            self.params['pcd'] = False
        else:
            self.params['pcd'] = pcd
    
    def get_params_str(self):
        param_str = ('%i_inits_%s_norm_%i_epochs_%i_batch_size_%s_hidden_activation_%s_output_activation'%
                (self.params['n_inits'],self.params['norm'],self.params['n_epochs'],self.params['batch_size'],
                      self.params['hidden_activation'],self.params['output_activation']))
        for imetric in self.params['metrics']:
            param_str = param_str + '_' + imetric
      
        param_str = param_str + '_metric_' + self.params['loss'] + '_loss'
        return param_str
    
    def getModelPath(self):
        
        self.path = "NeuralNetwork,{0}_optmizer,{1}_hidden_activation,{2}_output_activation,{3}_pcd,{4}_balanced,{5}_init_{6}_folds_{7}_norm_{8}_epochs_{9}_batch_size_{10}_neuron".format(
            self.params['optmizerAlgorithm'],
            self.params['hidden_activation'],
            self.params['output_activation'],
            self.params['pcd'],
            self.params['class_weight_flag'],
            self.params['n_inits'],
            self.params['folds'],
            self.params['norm'],
            self.params['n_epochs'],
            self.params['batch_size'],
            self.params['neuron']
         )
        
        for imetric in self.params['metrics']:
             self.path = self.path + '_' + 'TesteSP'#imetric
        self.path = self.path + '_metric_' + self.params['loss'] + '_loss'
        
        if platform.system() == "Linux":
            delimiter = '/'
        
        self.path = self.path.replace(',',delimiter)
        
        return self.path
            
        
# novelty detection

def NoveltyDetectionFolds(folder, n_folds=2, trgt=None, dev=False, verbose=False):
    if n_folds < 2:
        print 'Invalid number of folds'
        return -1

    if not dev:
        file_name = '%s/%i_folds_cross_validation.jbl'%(folder,n_folds)
    else:
        file_name = '%s/%i_folds_cross_validation_dev.jbl'%(folder,n_folds)

    if not os.path.exists(file_name):
        if verbose:
            print "Creating %s"%(file_name)

        if trgt is None:
            print 'Invalid trgt'
            return -1

        CVO = {}
        for inovelty,novelty_class in enumerate(np.unique(trgt)):
            skf = model_selection.StratifiedKFold(n_splits=n_folds)
            process_trgt = trgt[trgt!=novelty_class]
            CVO[inovelty] = skf.split(X = np.zeros(process_trgt.shape), y=process_trgt)
            CVO[inovelty] = list(CVO[inovelty])
        if verbose:
            print 'Saving in %s'%(file_name)

        joblib.dump([CVO],file_name,compress=9)

    else:
        if verbose:
            print "Reading from %s"%(file_name)

        [CVO] = joblib.load(file_name)

    return CVO

class SVMNoveltyDetectionTrnParams(TrnParams):
    """
        SVM Novelty Detection TrnParams
    """

    def __init__(self,
                 norm='mapstd',
                 gamma=0.01,
                 kernel='rbf',
                 verbose=False):
        self.params = {}

        self.params['norm'] = norm
        self.params['verbose'] = verbose
        self.params['gamma'] = gamma
        self.params['kernel'] = kernel

    def get_params_str(self):
        gamma_str = ('%1.5f'%(self.params['gamma'])).replace('.','_')
        param_str = ('%s_norm_%s_gamma_%s_kernel'%
                     (self.params['norm'],gamma_str,self.params['kernel']))
        return param_str

class NNNoveltyDetectionTrnParams(NeuralClassificationTrnParams):
    """
        NN Novelty Detection TrnParams
    """

class SAENoveltyDetectionTrnParams(NeuralClassificationTrnParams):
    """
        SAE Novelty Detection TrnParams
    """
