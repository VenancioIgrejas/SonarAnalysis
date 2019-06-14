import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import clone
from Functions.util import file_exist
from pandas import Series

def clone_callback(callback):

	if not callable(getattr(callback, "get_arguments",None)):
		raise ValueError("{0} dont have 'get_arguments' method".format(callback))
	klass = callback.__class__
	return klass(**callback.get_arguments())


class BaseClassifier(object):
	"""docstring for HierarBase"""
	def __init__(self, estimator, classes_name, dir=None, verbose=False):
		self.classes_name = classes_name
		self.dir = dir
		self.dict_df = {}
		self.master_table = None
		self.estimator = estimator
		self.verbose = verbose

	def convert_to_1dim(self,target):
		if target.ndim > 1:
			#if trgt is in sparce mode
			target = np.argmax(target,axis=1)
		return target

	def _add(self,name,value):
		self.dict_df[name] = value
		return

	def _add_df(self,name,value):
		self.master_table.loc[:,name] = Series(value, index=self.master_table.index)
		return

	def _add_class(self, name, classes_of, trgt):
		#rewrite this function acoording to multiclassifier
		self.dict_df['target'] = trgt
		new_trgt = -np.ones(shape=trgt.shape,dtype=int)
		#----------
		#write the map trgt here


		#----------
		self.dict_df[name] = new_trgt

	def _add_ifold_df(self, y, train_id, test_id, ifold):
		if not self._check_col('fold_{0:02d}'.format(ifold)):
			y = self.convert_to_1dim(y)
			fold = -np.ones(shape=y.shape,dtype=int)
			fold[test_id] = 1
			fold[train_id] = 0
			self._add_df('fold_{0:02d}'.format(ifold),fold)
			self._save_df()
			return True
		return False

	def _load_df(self):

		if not self.dir is None:
			self.master_table = pd.read_csv(self.dir+'/master_table.csv')

	def _save_df(self):
		if not self.dir is None:
			self.master_table.to_csv(self.dir+'/master_table.csv',index=False)
		return

	def _check_col(self,name):

		if not self.master_table is None:
			if name in self.master_table.columns:
				return True
			return False

		return None

	def _create_df(self):
		if not self.dict_df:
			#dict is empty
			raise ValueError("somethings bad happens, dict_df is None")

		self.master_table = pd.DataFrame(self.dict_df)
		return

	def _mount(self,trgt,load=False):

		
		#if trgt is in sparce mode
		trgt = self.convert_to_1dim(trgt)

		#rewrite this function acoording to multiclassifier
		if load:
			self._load_df()
			return
		#----------
		#write the map trgt here
		for key,value in self.classes_name.iteritems():
			self._add_class(key,value,trgt)

		#----------
		return

	def _prepare_table(self,y=None,load=False):
		
		if (load==False):
			self._mount(y)
			self._create_df()
		elif load==True:
			self._load_df()
		else:
			raise ValueError("invalid paramns 'y' and 'load'")
		return

	def _fit_member(self, member, X, y, train_id=None, test_id=None, ifold=None):

		if isinstance(self.estimator,dict):
			estimator = clone(self.estimator[member])
		else:
			estimator = clone(self.estimator)

		if hasattr(estimator,'validation_id') & hasattr(estimator,'dir'):
			folder = self.dir + '/{0}'.format(member)
			if not os.path.exists(folder):
				os.makedirs(folder)

			if hasattr(estimator,'ifold') & hasattr(estimator,'classes_name'):
				class_name_spec = ['{0:02d}_specialist'.format(i) for i in range(len(np.unique(y[train_id])))] #because of values -1 in hierarque classifier

				estimator.set_params(**dict([('classes_name',class_name_spec),('ifold',ifold),('dir',folder),('validation_id',(train_id,test_id))]))

			else:
				estimator.set_params(**dict([('dir',folder),('validation_id',(train_id,test_id))]))

		estimator.fit(X, y)

		return estimator



	def fit_member(self, member, X, y):
		df = self.master_table
		y_new = df[df[member]!=-1][member].values
		ids = df[df[member]!=-1].index.values
		X_new = X[ids]

		return self._fit_member(member, X_new, y_new)

	


	def fit(self, X, y):
		self._mount(y)
		self._create_df()
