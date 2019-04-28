import os

import numpy as np
import pandas as pd

import hashlib


class BDControl(object):
	"""docstring for Control"""
	def __init__(self, main_path, columns,bd_file='bd.csv'):
		self.main_path = main_path
		self.bd = None
		self.df_control = self.main_path + '/'+bd_file
		self._exist = os.path.exists(self.df_control)
		
		columns = columns + ['hash','path']

		if self._exist:
			self.bd = pd.read_csv(self.df_control)
		else:
			print("Creating BD in path {0}".format(self.main_path))
			self.bd = pd.DataFrame(columns=columns)


	def get_path(self,**kwargs):
		s = self.get_row(**kwargs)
		return s['path'].values[0]

	def get_row(self,**kwargs):

		if not self.check_hash(**kwargs):
			print("Warning: Creating new line of BD")
			self.append(**kwargs)
			self.save()

		df = self.bd

		idx = df[df['hash']==self.hash].index.values

		self.row_bd = df.loc[idx,:]
		return self.row_bd

	def append(self,**kwargs):
		hash_path = hashlib.sha256(str(kwargs)).hexdigest()
		self.hash = hash_path
		path = self.main_path + '/'+hash_path

		kwargs.update({'hash':hash_path,'path':path})

		self.bd = self.bd.append(pd.DataFrame(
			kwargs,index=[0]
			),ignore_index=True,sort=True)

		return self

	def save(self):
		self.bd.to_csv(self.df_control,index=False)

	def check_hash(self,**kwargs):
		hash_path = hashlib.sha256(str(kwargs)).hexdigest()
		self.hash = hash_path

		df = self.bd

		has_hash = np.any(df['hash']==hash_path)

		return has_hash



		