#HierarqNet
import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import clone
from Functions.util import file_exist
from pandas import Series

class HierarqBase(object):
	"""docstring for HierarBase"""
	def __init__(self, estimator, dict_classes, file=None, verbose=False):
		self.dict_classes = dict_classes
		self.file = file
		self.dict_df = {}
		self.master_table = None
		self.estimator = estimator
		self.verbose = verbose

	def _add(self,name,value):
		self.dict_df[name] = value
		return

	def _add_df(self,name,value):
		self.master_table.loc[:,name] = Series(value, index=self.master_table.index)
		return

	def _add_class(self,name,classes_of,trgt):
		self.dict_df['target'] = trgt
		new_trgt = -np.ones(shape=trgt.shape,dtype=int)
		for new_class, values in enumerate(classes_of):
			values = map(lambda x:x-1,values)
			for iclass in values:
				new_trgt[trgt==iclass] = new_class
		self.dict_df[name] = new_trgt

	def _load_df(self):

		if not self.file is None:
			self.master_table = pd.read_csv(self.file+'/master_table.csv')

	def _save_df(self):
		if not self.file is None:
			self.master_table.to_csv(self.file+'/master_table.csv',index=False)
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
		
		if load:
			self._load_df()
			return

		for key,value in self.dict_classes.iteritems():
			self._add_class(key,value,trgt)
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
		estimator = clone(self.estimator)
		if hasattr(estimator,'validation_id') & hasattr(estimator,'dir'):
			folder = self.file + '/fold{0:02d}'.format(ifold) + '/{0}'.format(member)
			if not os.path.exists(folder):
				os.makedirs(folder)
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

class HierarqClassification(HierarqBase):
	"""docstring for HierarqClassification"""
	def __init__(self, estimator, dict_classes,n_lvl,map_members, has_fold=True,file=None,verbose=False):
		super(HierarqClassification, self).__init__(estimator=estimator, 
													dict_classes=dict_classes, 
													file=file,
													verbose=verbose)

		self.n_lvl = n_lvl
		self.map_members = map_members
		self.dict_classes = dict_classes
		self.estimators_={}
		self.active_fold = has_fold

	def _add_ifold_df(self, y, train_id, test_id, ifold):
		if not self._check_col('fold_{0:02d}'.format(ifold)):
			fold = -np.ones(shape=y.shape,dtype=int)
			fold[test_id] = 1
			fold[train_id] = 0
			self._add_df('fold_{0:02d}'.format(ifold),fold)
			self._save_df()
			return True
		return False

	def fit_member(self, member, X, y, ifold):
		df = self.master_table
		
		train_id = df[
						(df[member]!=-1)&(df['fold_{0:02d}'.format(ifold)]==0)
						]['fold_{0:02d}'.format(ifold)].index.values

		test_id = df[
						(df[member]!=-1)&(df['fold_{0:02d}'.format(ifold)]==1)
						]['fold_{0:02d}'.format(ifold)].index.values

		y_member = df[member].values

		self.estimators_[member] = self._fit_member(member, X, y_member, train_id, test_id, ifold)
		return self.estimators_[member]

	def fit_level(self, members, X, y, ifold):
		estimators_lvl = []
		for member in members:
			estimators_lvl.append(self.fit_member(member, X, y, ifold))
		return estimators_lvl



	def _fit_ifold(self, X, y, train_id, test_id, ifold):

		if (self._add_ifold_df(y, train_id, test_id, ifold))&(self.verbose):print("Add vector of the fold {0} in master table".format(ifold))
		self.ifold = ifold
		for key in self.dict_classes.keys():
			if self.verbose: print('fit member {0}'.format(key))
			self.fit_member(member=key, X=X, y=y, ifold=ifold)

		#for i in range(n_lvl):
		#	lvl_members = []
		#	for key,values in self.map_members.items():
		#		if values == i:
		#			lvl_members.append(key)

		#	self.fit_level(members=lvl_members, X=X, y=y, ifold=ifold)

		return



	def fit(self, X, y, train_id=None, test_id=None,ifold=None):
		load = False

		if file_exist(self.file+'/master_table.csv'):
			load = True

		self._prepare_table(y,load=load)

		if self.active_fold:
			self._fit_ifold(X, y, train_id, test_id, ifold)

	def predict_member(self, member, X):
		
		estimator = self.estimators_[member]

		pred = estimator.predict(X,predict='sparce')

		if hasattr(estimator,'dir'):

			file_predict = estimator.get_params()['dir'] + '/predict.csv'
			pd.DataFrame(pred).to_csv(file_predict,header=['neuron_{0}'.format(i) for i in range(pred.shape[1])],index=False)

		return pred


	def _pred_table(self,df,classes):
	    for oldclass,newclass in enumerate(classes['class_AA']):
	        newclass = map(lambda x:x-1,newclass)
	        ix = df[(df['class_S']==0)&(df['class_A']==0)&(df['class_AA']==oldclass)]['pred'].index.values
	        df.loc[ix,'pred'] = newclass

	    
	    for oldclass,newclass in enumerate(classes['class_AB']):
	        newclass = map(lambda x:x-1,newclass)
	        ix = df[(df['class_S']==0)&(df['class_A']==1)&(df['class_AB']==oldclass)]['pred'].index.values
	        df.loc[ix,'pred'] = newclass

	    #predict of class 21 (AC)
	    ix = df[(df['class_S']==0)&(df['class_A']==2)]['pred'].index.values
	    df.loc[ix,'pred'] = 20

	    for oldclass,newclass in enumerate(classes['class_B']):
	        newclass = map(lambda x:x-1,newclass)
	        ix = df[(df['class_S']==1)&(df['class_B']==oldclass)]['pred'].index.values
	        df.loc[ix,'pred'] = newclass

	    for oldclass,newclass in enumerate(classes['class_C']):
	        newclass = map(lambda x:x-1,newclass)
	        ix = df[(df['class_S']==2)&(df['class_C']==oldclass)]['pred'].index.values
	        df.loc[ix,'pred'] = newclass

	    for oldclass,newclass in enumerate(classes['class_DA']):
	        newclass = map(lambda x:x-1,newclass)
	        ix = df[(df['class_S']==3)&(df['class_D']==0)&(df['class_DA']==oldclass)]['pred'].index.values
	        df.loc[ix,'pred'] = newclass    

	    for oldclass,newclass in enumerate(classes['class_DB']):
	        newclass = map(lambda x:x-1,newclass)
	        ix = df[(df['class_S']==3)&(df['class_D']==1)&(df['class_DB']==oldclass)]['pred'].index.values
	        df.loc[ix,'pred'] = newclass    
	    
	    return df


	def predict(self, X, y=None):
		pred_m = {}
		pred_m['pred'] = -np.ones(shape=(X.shape[0],),dtype=int)
		for key in self.dict_classes.keys():
			if self.verbose: print('predict member {0}'.format(key))
			pred_m[key] = np.argmax(self.predict_member(key, X),axis=1)

		pred_df = pd.DataFrame(pred_m)

		pred_df = self._pred_table(pred_df,self.dict_classes)

		pred_df.to_csv(self.file + '/fold{0:02d}'.format(self.ifold) + '/pred_all.csv',index=False)

		if -1 in np.unique(pred_df['pred'].values):
			raise ValueError("something wrong happend with predict of all classes")

		return	pred_df['pred'].values




