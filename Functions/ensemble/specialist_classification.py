import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import clone
from Functions.util import file_exist, class_weight_keras
from pandas import Series

from keras.utils import np_utils

from .bases import BaseClassifier, clone_callback

class SpecialistBase(BaseClassifier):
	"""docstring for SpecialistBase"""
	def __init__(self, estimator, classes_name, dir=None, verbose=False):
		super(SpecialistBase, self).__init__(estimator=estimator,classes_name=classes_name,dir=dir,verbose=verbose)
 
	def _mount(self,trgt,load=False):

		if load:
			self._load_df()
			return

		for iclass,name in enumerate(self.classes_name):
			self._add_class(name=name, classe_binary=iclass, trgt=trgt)
		return

	def _add_class(self,name,classe_binary,trgt):
		self.dict_df['target'] = trgt
		new_trgt = -np.ones(shape=trgt.shape,dtype=int)
		new_trgt[trgt==classe_binary] = 1
		new_trgt[trgt!=classe_binary] = 0
		self.dict_df[name] = new_trgt
		return


	#adicionei essa funcao pra testar o novo codigo da mlp!!!
	def _fit_member(self, member, X, y, train_id=None, test_id=None, ifold=None):

		if isinstance(self.estimator,dict):
			estimator = clone(self.estimator[member])
		else:
			estimator = clone(self.estimator)

		if hasattr(estimator,'validation_id') & hasattr(estimator,'dir'):
			
			paramns_update = {}

			paramns_update['validation_id'] = (train_id,test_id)

			folder = self.dir + '/{0}'.format(member)
			if not os.path.exists(folder):
				os.makedirs(folder)

			paramns_update['dir'] = folder

			# if hasattr(estimator,'ifold') & hasattr(estimator,'classes_name'):
			# 	class_name_spec = ['{0:02d}_specialist'.format(i) for i in range(len(np.unique(y[train_id])))] #because of values -1 in hierarque classifier

			# 	estimator.set_params(**dict([('classes_name',class_name_spec),('ifold',ifold),('dir',folder),('validation_id',(train_id,test_id))]))

			# else:
			# 	estimator.set_params(**dict([('dir',folder),('validation_id',(train_id,test_id))]))
			# 	print 'aqui'
			if hasattr(estimator, 'callbacks_list'):
				#creating new istances of callbacks for each specialist
				new_callbacks = []
				for icallback in self.estimator.callbacks_list:
					new_callbacks.append(clone_callback(icallback))
				paramns_update['callbacks_list'] = new_callbacks
			
			if hasattr(estimator,'fit_kwargs') &  self.class_weight>0:
				fit_kwargs = estimator.get_params()['fit_kwargs']

				fit_kwargs.update({'class_weight':class_weight_keras(y)})

				paramns_update['fit_kwargs'] = fit_kwargs
			
			estimator.set_params(**paramns_update)
		
		estimator.fit(X, y)

		return estimator


class SpecialistClassification(SpecialistBase,BaseEstimator):
	"""docstring for SpecialistClassification"""
	def __init__(self, estimator, classes_name, active_fold=True,
				dir=None,verbose=False,integrator='wta',
				validation_id=(None,None),ifold=None,class_weight=0):

		super(SpecialistClassification, self).__init__(estimator=estimator, 
											classes_name=classes_name, 
											dir=dir, 
											verbose=verbose)

		type_integrator = ['weight','wta'] #wta -> Winner Take All(defaul)
		if not integrator in type_integrator:
			self.integrator = 'wta'
		else:
			self.integrator = integrator

		self.estimators_={}
		self.weight={}
		self.active_fold = active_fold
		self.validation_id = validation_id
		self.ifold = ifold
		self.class_weight = class_weight



	def fit_member(self, member, X, y, ifold):
		df = self.master_table
		
		train_id = df[
						(df['fold_{0:02d}'.format(ifold)]==0)
						]['fold_{0:02d}'.format(ifold)].index.values

		test_id = df[
						(df['fold_{0:02d}'.format(ifold)]==1)
						]['fold_{0:02d}'.format(ifold)].index.values

		if y.ndim > 1:
			y_member = y

		y_member = df[member].values

		self.estimators_[member] = self._fit_member(member, X, y_member, train_id, test_id, ifold)
		return self.estimators_[member]

	def _fit_ifold(self, X, y, train_id, test_id, ifold):

		if (self._add_ifold_df(y, train_id, test_id, ifold))&(self.verbose):print("Add vector of the fold {0} in master table".format(ifold))
		self.ifold = ifold
		for iclass,name in enumerate(self.classes_name):
			if self.verbose: print('fit Specialist {0}'.format(iclass))
			self.fit_member(member=name, X=X, y=y, ifold=ifold)
		return

	def _fit_ifold_member(self, member, X, y, train_id, test_id, ifold):
		if (self._add_ifold_df(y, train_id, test_id, ifold))&(self.verbose):print("Add vector of the fold {0} in master table".format(ifold))
		self.ifold = ifold
		if self.verbose: print('fit member {0}'.format(member))
		self.fit_member(member=member, X=X, y=y, ifold=ifold)
		return


	def _fit_weights(self, X):
		df = self.master_table
		
		train_id = df[
						(df['fold_{0:02d}'.format(self.ifold)]==0)
						]['fold_{0:02d}'.format(self.ifold)].index.values

		pred_mt = pd.DataFrame(dict([(name,self.predict_member(name, X)[:,1]) for name in self.classes_name]),
								columns=self.classes_name).loc[train_id,:]
		pred_inv = pd.DataFrame(np.linalg.pinv(pred_mt.values))

		for name in self.classes_name:
			if self.verbose:print("weight of {0}".format(name))
			s_trgt = df[name]
			self.weight[name] = np.dot(pred_inv,s_trgt[train_id])
			estimator = self.estimators_[name]

			if hasattr(estimator,'dir'):
				file_weight = estimator.get_params()['dir'] + '/weigth.csv'
				pd.DataFrame(self.weight[name]).to_csv(file_weight,header=[name],index=False)

		return

	def fit(self, X, y, sample_weights=None,train_id=None, test_id=None,ifold=None):
		load = False

		if file_exist(self.dir+'/master_table.csv'):
			load = True

		self._prepare_table(y,load=load)

		

		if self.active_fold:
			if self.validation_id[0] is None:
				self._fit_ifold(X, y, train_id, test_id, ifold)
			else:
				self._fit_ifold(X, y, self.validation_id[0], self.validation_id[1] , self.ifold)

		if self.verbose:print("creating weights")

		self._fit_weights(X)

		return

	def fit_only_member(self, member, X, y, sample_weights=None,train_id=None, test_id=None,ifold=None):
		load = False

		if file_exist(self.dir+'/master_table.csv'):
			load = True

		self._prepare_table(y,load=load)

		

		if self.active_fold:
			if self.validation_id[0] is None:
				self._fit_ifold_member(member, X, y, train_id, test_id, ifold)
			else:
				self._fit_ifold_member(member, X, y, self.validation_id[0], self.validation_id[1] , self.ifold)


	def predict_member(self, member, X):
		
		estimator = self.estimators_[member]

		pred = estimator.predict(X,predict='sparce')

		if hasattr(estimator,'dir'):

			file_predict = estimator.get_params()['dir'] + '/predict.csv'
			pd.DataFrame(pred,columns=['neuron_{0}'.format(i) for i in range(pred.shape[1])]).to_csv(file_predict,index=False)


		return pred

	def predict(self, X, y=None, predict='sparce'):
		df = self.master_table

		test_id = df[
						(df['fold_{0:02d}'.format(self.ifold)]==1)
						]['fold_{0:02d}'.format(self.ifold)].index.values

		df_pred_out = pd.DataFrame(dict([(name,self.predict_member(name, X)[:,1]) for name in self.classes_name]),
								columns=self.classes_name)

		df_pred_out.to_csv(self.dir + '/pred_all.csv',index=False)

		df_weights = pd.DataFrame([self.weight[name] for name in self.classes_name],columns=self.classes_name)

		df_pred_out_weights = np.dot(df_pred_out,df_weights)

		pd.DataFrame(df_pred_out_weights,columns=self.classes_name).to_csv(self.dir + '/pred_all_withWeights.csv',index=False)

		if self.integrator is 'weight':
			return df_pred_out_weights
		else:
			# in this case, integrator will be 'wta'
			return df_pred_out.values

