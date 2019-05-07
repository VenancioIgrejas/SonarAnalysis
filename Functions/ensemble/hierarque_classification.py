#HierarqNet
import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import clone
from Functions.util import file_exist
from pandas import Series

class BaseClassifier(object):
	"""docstring for HierarBase"""
	def __init__(self, estimator, classes_name, dir=None, verbose=False):
		self.classes_name = classes_name
		self.dir = dir
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

class HierarqBase(BaseClassifier):
	"""docstring for HierarqBase"""
	def __init__(self, estimator, dict_classes, dir=None, verbose=False):
		super(HierarqBase, self).__init__(estimator=estimator,classes_name=dict_classes,dir=dir,verbose=verbose)

	def _mount(self,trgt,load=False):
		
		if load:
			self._load_df()
			return

		for key,value in self.classes_name.iteritems():
			self._add_class(key,value,trgt)
		return

	def _add_class(self,name,classes_of,trgt):
		
		self.dict_df['target'] = trgt
		new_trgt = -np.ones(shape=trgt.shape,dtype=int)

		for new_class, values in enumerate(classes_of):
			values = map(lambda x:x-1,values)
			for iclass in values:
				new_trgt[trgt==iclass] = new_class
		
		self.dict_df[name] = new_trgt

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

class HierarqClassification(HierarqBase):
	"""docstring for HierarqClassification"""
	def __init__(self, estimator, dict_classes,n_lvl,map_members, active_fold=True,dir=None,verbose=False):
		super(HierarqClassification, self).__init__(estimator=estimator, 
													dict_classes=dict_classes, 
													dir=dir,
													verbose=verbose)

		self.n_lvl = n_lvl
		self.map_members = map_members
		self.dict_classes = dict_classes
		self.estimators_={}
		self.active_fold = active_fold
		self.len_classes = None

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


	def _fit_ifold(self, X, y, train_id, test_id, ifold):

		if (self._add_ifold_df(y, train_id, test_id, ifold))&(self.verbose):print("Add vector of the fold {0} in master table".format(ifold))
		self.ifold = ifold
		for key in self.dict_classes.keys():
			if self.verbose: print('fit member {0}'.format(key))
			self.fit_member(member=key, X=X, y=y, ifold=ifold)

		return



	def fit(self, X, y, train_id=None, test_id=None,ifold=None):

		self.len_classes = len(np.unique(y))

		load = False

		if file_exist(self.dir+'/master_table.csv'):
			load = True

		self._prepare_table(y,load=load)

		if self.active_fold:
			self._fit_ifold(X, y, train_id, test_id, ifold)

		return

	def _fit_ifold_member(self, member, X, y, train_id, test_id, ifold):
		if (self._add_ifold_df(y, train_id, test_id, ifold))&(self.verbose):print("Add vector of the fold {0} in master table".format(ifold))
		self.ifold = ifold
		if self.verbose: print('fit member {0}'.format(member))
		self.fit_member(member=member, X=X, y=y, ifold=ifold)
		return

	def fit_only_member(self, member, X, y, sample_weights=None,train_id=None, test_id=None,ifold=None):
		load = False

		if file_exist(self.dir+'/master_table.csv'):
			load = True

		self._prepare_table(y,load=load)

		

		if self.active_fold:
				self._fit_ifold_member(member, X, y, train_id, test_id, ifold)
		return

	def predict_member(self, member, X):
		df = self.master_table

		estimator = self.estimators_[member]

		pred = estimator.predict(X,predict='sparce')

		if not self._check_col('{0}_pred'.format(member)):
			pred_class = np.argmax(pred,axis=1)
			self._add_df(name='{0}_pred'.format(member), value=pred_class)
			self._save_df()

		if hasattr(estimator,'dir'):

			file_predict = estimator.get_params()['dir'] + '/predict.csv'
			pd.DataFrame(pred,columns=['neuron_{0}'.format(i) for i in range(pred.shape[1])]).to_csv(file_predict,index=False)

		return pred


	def _pred_table_24(self,df,classes):
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

	def _pred_table_31(self,df,classes):

	    for oldclass,newclass in enumerate(classes['class_A']):
	        newclass = map(lambda x:x-1,newclass)
	        ix = df[(df['class_S']==0)&(df['class_A']==oldclass)]['pred'].index.values
	        df.loc[ix,'pred'] = newclass

	    #predict of class 5 (B1)
	    ix = df[(df['class_S']==1)&(df['class_B']==0)]['pred'].index.values
	    df.loc[ix,'pred'] = 4

	    #predict of class 6 (B2)
	    ix = df[(df['class_S']==1)&(df['class_B']==1)]['pred'].index.values
	    df.loc[ix,'pred'] = 5

	    for oldclass,newclass in enumerate(classes['class_B3']):
	        newclass = map(lambda x:x-1,newclass)
	        ix = df[(df['class_S']==1)&(df['class_B']==2)&(df['class_B3']==oldclass)]['pred'].index.values
	        df.loc[ix,'pred'] = newclass

	    #predict of class 12 (B4)
	    ix = df[(df['class_S']==1)&(df['class_B']==3)]['pred'].index.values
	    df.loc[ix,'pred'] = 11

	    for oldclass,newclass in enumerate(classes['class_B5']):
	        newclass = map(lambda x:x-1,newclass)
	        ix = df[(df['class_S']==1)&(df['class_B']==4)&(df['class_B5']==oldclass)]['pred'].index.values
	        df.loc[ix,'pred'] = newclass

	    #predict of class 16 (B6)
	    ix = df[(df['class_S']==1)&(df['class_B']==5)]['pred'].index.values
	    df.loc[ix,'pred'] = 15

	    #predict of class 17 (B7)
	    ix = df[(df['class_S']==1)&(df['class_B']==6)]['pred'].index.values
	    df.loc[ix,'pred'] = 16

	    for oldclass,newclass in enumerate(classes['class_B8']):
	        newclass = map(lambda x:x-1,newclass)
	        ix = df[(df['class_S']==1)&(df['class_B']==7)&(df['class_B8']==oldclass)]['pred'].index.values
	        df.loc[ix,'pred'] = newclass
	    
	    return df

	def _new_map_members(self,df,classes,members):
		for member in members:
			df[member+'_old'] = df[member]
			df[member] = -1
			for new_class,old_class in enumerate(classes[member]):
				
				print old_class , new_class

				for sub_old in old_class:
					print sub_old
					df.loc[df[(df[member+'_old']==(sub_old-1))].index,member] = new_class

		return df


	def predict(self, X, y=None, choose_pred=None):

		if not self.len_classes in [24,31]:
			raise ValueError("only 24 or 31 classes, {0} classes is not implemented".format(self.len_classes))
			
		pred_m = {}
		pred_m['pred'] = -np.ones(shape=(X.shape[0],),dtype=int)
		for key in self.dict_classes.keys():
			if self.verbose: print('predict member {0}'.format(key))
			pred_m[key] = np.argmax(self.predict_member(member=key, X=X),axis=1)

		pred_df = pd.DataFrame(pred_m)

		if choose_pred=="mlp_super_all":

			if self.len_classes is 31:
				dict_map = {'class_S':[[1,2,3,4],
                        [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]]}
				self.dict_classes.update(dict_map)
				pred_df = self._new_map_members(pred_df,self.dict_classes,dict_map.keys())
				pred_df = self._pred_table_31(pred_df,self.dict_classes)
			else:

				dict_map = {'class_S':[[9,10,13,14,16,23,1,2,22,21],
                                 [4,6,8,12,17,19],
                                 [11,24],
                                 [5,7,15,3,18,20]]}
				self.dict_classes.update(dict_map)
				pred_df = self._new_map_members(pred_df,self.dict_classes,dict_map.keys())
				pred_df = self._pred_table_24(pred_df,self.dict_classes)

		else:

			if self.len_classes is 31:
				pred_df = self._pred_table_31(pred_df,self.dict_classes)
			else:
				pred_df = self._pred_table_24(pred_df,self.dict_classes)


		pred_df.to_csv(self.dir + '/pred_all.csv',index=False)

		if -1 in np.unique(pred_df['pred'].values):
			raise ValueError("something wrong happend with predict of all classes")

		return	pred_df['pred'].values

class SpecialistClassification(SpecialistBase,BaseEstimator):
	"""docstring for SpecialistClassification"""
	def __init__(self, estimator, classes_name, active_fold=True,dir=None,verbose=False,integrator='wta',validation_id=(None,None),ifold=None):
		super(SpecialistClassification, self).__init__(estimator=estimator, classes_name=classes_name, dir=dir, verbose=verbose)

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



	def fit_member(self, member, X, y, ifold):
		df = self.master_table
		
		train_id = df[
						(df['fold_{0:02d}'.format(ifold)]==0)
						]['fold_{0:02d}'.format(ifold)].index.values

		test_id = df[
						(df['fold_{0:02d}'.format(ifold)]==1)
						]['fold_{0:02d}'.format(ifold)].index.values

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




		




