"""
  This file contents some processing functions
"""

import numpy as np
import os
import pandas as pd
import json

from Functions.MetricCustom import ind_SP,f1_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD,Adam
import keras.callbacks as callbacks
from keras import metrics
from keras.utils import np_utils
from keras.layers import merge

from keras.models import load_model
from sklearn.externals import joblib

from keras import backend as K

class TrnParams(object):
	def __init__(self, learning_rate=0.0001,
	learning_decay=1e-6, momentum=0.3,
	nesterov=True, train_verbose=False, verbose= False,
	n_epochs=1000,batch_size=512,init=10,optimizers='sgd'):
		self.init = init
		self.learning_rate = learning_rate
		self.learning_decay = learning_decay
		self.momentum = momentum
		self.nesterov = nesterov
		self.train_verbose = train_verbose
		self.verbose = verbose
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.optimizers = optimizers

	def to_json(self,path):
		param = {'Learning Rate':self.learning_rate,
				'Learning Decay':self.learning_decay,
				'Momentum':self.momentum,
				'nesterov':self.nesterov,
				'train_verbose':self.train_verbose,
				'verbose':self.verbose,
				'num Epochs':self.n_epochs,
				'batch size':self.batch_size,
				'num Inicial':self.init,
				'optimizer':self.optimizers}
		with open(os.path.join(path,'infoParams.json'), 'w') as outfile:
                	json.dump(param, outfile)

	def Print(self):
		print 'Class TrnParams'
		print 'Learning Rate: %1.5f'%(self.learning_rate)
		print 'Learning Decay: %1.5f'%(self.learning_decay)
		print 'Momentum: %1.5f'%(self.momentum)
		if self.nesterov:
			print 'Nesterov: True'
		else:
			print 'Nesterov: False'
		if self.verbose:
			print 'Verbose: True'
		else:
			print 'Verbose: False'

		if self.train_verbose:
			print 'Train Verbose: True'
		else:
			print 'Train Verbose: False'
		print 'NEpochs: %i'%(self.n_epochs)
		print 'Batch Size: %i'%(self.batch_size)


class PCDIndependent(object):

	"""
	PCD Independent class
		This class implement the Principal Component of Discrimination Analysis in Independent Approach
	"""
	def __init__(self, n_components=2):
		"""
		PCD Independent constructor
			n_components: number of components to be extracted
		"""
		self.n_components = n_components
		self.models = {}
		self.best_desc = {}
		self.trn_descs = {}
		self.pcds = {}

	def fit(self, data, target, train_ids, test_ids, path, trn_params=None,class_weight=None):
                """
		PCD Independent fit function
			data: data to be fitted (events x features)
			target: class labels - sparse targets (events x number of classes)
			path: path where the models train will be save

			train_ids:  train indexes - user generated
			test_ids: test indexes - user generated
			trn_params: class TrnParams (optional)
			class_weight: set weight classes for unbalanced classes (like class_weight in fit Keras function)
		"""
		print 'PCD Independent fit function'




	# Save configuration about size of dataset that will use in PCD



        #load model and Trn

		if trn_params is None:
			trn_params = TrnParams()

		if self.n_components == 'Auto':
			if trn_params.verbose:
				print "train PCDs mode: Automatic"
			self.fit_auto(data, target, train_ids, test_ids, path, trn_params,class_weight)
			return self


		#print 'Train Parameters'
		#trn_params.Print()

		if trn_params.verbose:
			print 'PCD Independent Model Struct: %i - %i - %i'%(data.shape[1],1,target.shape[1])

		for ipcd in range(self.n_components):
			print 'Training %i PCD of %i PCDs'%(ipcd+1,self.n_components)

			if ipcd == 0:

                		fileModelPCD = os.path.join(path, "{0}comp_model_PCD.h5".format(ipcd))
                		fileTrnPCD = os.path.join(path, "{0}comp_trn_PCD.jbl".format(ipcd))

				if os.path.exists(fileModelPCD):
					self.models[ipcd] = load_model(fileModelPCD,custom_objects={'ind_SP': ind_SP})
                			self.trn_descs[ipcd] = joblib.load(fileTrnPCD)
                			self.pcds[ipcd] = load_model(fileModelPCD,custom_objects={'ind_SP': ind_SP}).layers[2].get_weights()[0]
					#print "loading file model: "+ "{0}comp_model_PCD.h5".format(ipcd)
				else:
					best_init = 0
					best_loss = 999
					best_model = None
					best_desc = {}
					for i_init in range(trn_params.init):
						print 'Init: %i of %i'%(i_init+1,trn_params.init)
						my_model = Sequential()

						# add a linear layer to isolate the input of NN model
						my_model.add(Dense(data.shape[1],input_dim=data.shape[1], kernel_initializer='identity',trainable=False))

						my_model.add(Activation('linear'))

						# add a non-linear single neuron layer to compress all information
						my_model.add(Dense(1, input_dim=data.shape[1], kernel_initializer='uniform'))
						my_model.add(Activation('tanh'))

						# add a non-linear output layer with max sparse target shape
						my_model.add(Dense(1, kernel_initializer='uniform'))
						my_model.add(Activation('tanh'))

						# creating a optimization function using steepest gradient
						if trn_params.optimizers == 'sgd':
							opt = SGD(lr=trn_params.learning_rate, decay=trn_params.learning_decay,
									momentum=trn_params.momentum, nesterov=trn_params.nesterov)
	                			if trn_params.optimizers == 'adam':
		                			opt = Adam(lr=trn_params.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

						# compile the model
						my_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

						# early stopping to avoid overtraining
						earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
	        	                                                patience=25, verbose=0,
	        	                                                mode='auto')

						# csv log to monitor the training in real time
						csv_logger = callbacks.CSVLogger(os.path.join(path,
										'{0}_numComponent_{1}_init_training.csv'.format(ipcd,i_init+1)))


						# reduce learning rate to train faster
						reduce_lrn = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,
												mode='auto',min_delta=0.0001, cooldown=0, min_lr=0)

						# Train model
						init_trn_desc = my_model.fit(data[train_ids], target[train_ids],
	                                        	  		  epochs=trn_params.n_epochs,
	                                        	  		  batch_size=trn_params.batch_size,
	                                        	  		  callbacks=[earlyStopping,csv_logger,reduce_lrn],
	                                        	  		  verbose=trn_params.train_verbose,
	                                        		          class_weight=class_weight,
	                                        	  		  validation_data=(data[test_ids],
	                                        	                  		   target[test_ids]),
	                                        	  		  shuffle=True)

	                			#saving the model and trn in the respective training folder

						if np.min(init_trn_desc.history['val_loss']) < best_loss:
							best_init = i_init
                            				best_loss = np.min(init_trn_desc.history['val_loss'])
                            				best_model = my_model
                            				best_desc['epochs'] = init_trn_desc.epoch
                            				best_desc['acc'] = init_trn_desc.history['acc']
                            				best_desc['loss'] = init_trn_desc.history['loss']
                            				best_desc['val_loss'] = init_trn_desc.history['val_loss']
                            				best_desc['val_acc'] = init_trn_desc.history['val_acc']
                                                        print "init {0} : epoc -> {1}/ val_loss -> {2}/ val_acc -> {3}".format(best_desc['epochs'],best_desc['val_loss'],best_desc['val_acc'])


					for i_init in range(trn_params.init):
                        			if best_init!=i_init:
                            				os.remove(os.path.join(path,
									'{0}_numComponent_{1}_init_training.csv'.format(ipcd,i_init+1)))
                        			else:
                            				os.rename(os.path.join(path,
									'{0}_numComponent_{1}_init_training.csv'.format(ipcd,i_init+1)),
                                      					os.path.join(path,
									'{0}_numComponent_training.csv'.format(ipcd)))

					best_model.save(fileModelPCD)

                			joblib.dump([best_desc],fileTrnPCD,compress=9)


					self.models[ipcd] = my_model
					self.trn_descs[ipcd] = best_desc
                			self.pcds[ipcd] = my_model.layers[2].get_weights()[0]


				if trn_params.verbose :
					print 'PCD %i - Train process is done, val_cost: %1.5f'%(ipcd+1,np.min(init_trn_desc.history['val_loss']))
			else:


                		fileModelPCD = os.path.join(path, "{0}comp_model_PCD.h5".format(ipcd))
                		fileTrnPCD = os.path.join(path, "{0}comp_trn_PCD.jbl".format(ipcd))

				if os.path.exists(fileModelPCD):
					self.models[ipcd] = load_model(fileModelPCD,custom_objects={'ind_SP': ind_SP})
                			self.trn_descs[ipcd] = joblib.load(fileTrnPCD)
                			self.pcds[ipcd] = load_model(fileModelPCD,custom_objects={'ind_SP': ind_SP}).layers[2].get_weights()[0]
					#print "loading file model: "+ "{0}comp_model_PCD.h5".format(ipcd)
				else:
					best_init = 0
					best_loss = 999
					best_model = None
					best_desc = {}
					for i_init in range(trn_params.init):
						print 'Init: %i of %i'%(i_init+1,trn_params.init)

						my_model = Sequential()

						# add a linear layer to isolate the input of NN model
						my_model.add(Dense(data.shape[1],input_dim=data.shape[1], kernel_initializer='identity',trainable=False))

						my_model.add(Activation('linear'))

						# add a non-linear single neuron layer to compress all information
						my_model.add(Dense(1, input_dim=data.shape[1], kernel_initializer='uniform'))
						my_model.add(Activation('tanh'))

						# add a non-linear output layer with max sparse target shape
						my_model.add(Dense(target.shape[1], kernel_initializer='uniform'))
						my_model.add(Activation('tanh'))

						# creating a optimization function using steepest gradient
						if trn_params.optimizers == 'sgd':
							opt = SGD(lr=trn_params.learning_rate, decay=trn_params.learning_decay,
									momentum=trn_params.momentum, nesterov=trn_params.nesterov)
	                			if trn_params.optimizers == 'adam':
		                			opt = Adam(lr=trn_params.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

						# compile the model
						my_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
						# early stopping to avoid overtraining
						earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                		                                        patience=25, verbose=0,
                		                                        mode='auto')

						# csv log to monitor the training in real time
						csv_logger = callbacks.CSVLogger(os.path.join(path,'{0}_numComponent_{1}_init_training.csv'.format(ipcd,i_init+1)))


						# reduce learning rate to train faster
						reduce_lrn = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,
												mode='auto',min_delta=0.0001, cooldown=0, min_lr=0)


						# remove the projection of previous extracted pcds from random init weights
						w = my_model.layers[2].get_weights()[0]
						w_proj = np.zeros_like(w)

						# loop over previous pcds
						for i_old_pcd in range(ipcd):
							w_proj = (w_proj + (np.inner(np.inner(self.pcds[i_old_pcd],w),self.pcds[i_old_pcd].T)/
									    np.inner(self.pcds[i_old_pcd].T,self.pcds[i_old_pcd].T)))
						w_remove_proj = w - w_proj
						weights = my_model.layers[2].get_weights()
						weights[0] = w_remove_proj
						my_model.layers[2].set_weights(weights)

						# remove the projection of previous extracted pcds from data
						data_proj = np.zeros_like(data)

						# loop over previous pcds
						for i_old_pcd in range(ipcd-1):
							data_proj = (data_proj + (np.inner(np.inner(self.pcds[i_old_pcd].T,data).T,self.pcds[i_old_pcd])/
										  np.inner(self.pcds[i_old_pcd].T,self.pcds[i_old_pcd].T)))

						data_with_proj = data - data_proj

						# Train model
						init_trn_desc = my_model.fit(data_with_proj[train_ids], target[train_ids],
                	                          		  	epochs=trn_params.n_epochs,
                	                          		  	batch_size=trn_params.batch_size,
                	                          		  	callbacks=[earlyStopping,csv_logger,reduce_lrn],
                	                          		  	class_weight=class_weight,
                	                          		  	verbose=trn_params.train_verbose,
                	                          		  	validation_data=(data[test_ids],
                	                                          			   target[test_ids]),
                	                          		  	shuffle=True)

						if np.min(init_trn_desc.history['val_loss']) < best_loss:
							best_init = i_init
                            				best_loss = np.min(init_trn_desc.history['val_loss'])
                            				best_model = my_model
                            				best_desc['epochs'] = init_trn_desc.epoch
                            				best_desc['acc'] = init_trn_desc.history['acc']
                            				best_desc['loss'] = init_trn_desc.history['loss']
                            				best_desc['val_loss'] = init_trn_desc.history['val_loss']
                            				best_desc['val_acc'] = init_trn_desc.history['val_acc']
                                                        print "init {0} : epoc -> {1}/ val_loss -> {2}/ val_acc -> {3}".format(best_desc['epochs'],best_desc['val_loss'],best_desc['val_acc'])

                			#saving the model and trn in the respective training folder

                			for i_init in range(trn_params.init):
                        			if best_init!=i_init:
                            				os.remove(os.path.join(path,
									'{0}_numComponent_{1}_init_training.csv'.format(ipcd,i_init+1)))
                        			else:
                            				os.rename(os.path.join(path,
									'{0}_numComponent_{1}_init_training.csv'.format(ipcd,i_init+1)),
                                      					os.path.join(path,
									'{0}_numComponent_training.csv'.format(ipcd)))

					best_model.save(fileModelPCD)

                			joblib.dump([best_desc],fileTrnPCD,compress=9)


					self.models[ipcd] = my_model
					self.trn_descs[ipcd] = best_desc
                			self.pcds[ipcd] = my_model.layers[2].get_weights()[0]


                		if trn_params.verbose:
                    			print 'PCD %i - Train process is done, val_cost: %1.5f'%(ipcd+1,np.min(init_trn_desc.history['val_loss']))

		return self

	# function deprecated
	def fit_auto(self, data, target, train_ids, test_ids, path, trn_params=None,class_weight=None):
                """
		PCD Independent fit function
			data: data to be fitted (events x features)
			target: class labels - sparse targets (events x number of classes)

			train_ids:  train indexes - user generated
			test_ids: test indexes - user generated
			trn_params: class TrnParams (optional)
		"""
		print 'PCD Independent fit function'

		if trn_params is None:
			trn_params = TrnParams()

		#print 'Train Parameters'
		#trn_params.Print()

        	#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        	opt = SGD(lr=trn_params.learning_rate, decay=trn_params.learning_decay,
				momentum=trn_params.momentum, nesterov=trn_params.nesterov)

		if trn_params.verbose:
			print 'PCD Independent Model Struct: %i - %i - %i'%(data.shape[1],1,target.shape[1])

		ipcd = 0

		while(True):
			print 'Training %i PCD'%(ipcd+1)

			if ipcd == 0:
				my_model = Sequential()

				# add a linear layer to isolate the input of NN model
				my_model.add(Dense(data.shape[1],input_dim=data.shape[1], kernel_initializer='identity',trainable=False))

				my_model.add(Activation('linear'))

				# add a non-linear single neuron layer to compress all information
				my_model.add(Dense(1, input_dim=data.shape[1], kernel_initializer='uniform'))
				my_model.add(Activation('tanh'))

				# add a non-linear output layer with max sparse target shape
				my_model.add(Dense(target.shape[1], kernel_initializer='uniform'))
				my_model.add(Activation('tanh'))

				# creating a optimization function using steepest gradient
				#sgd = SGD(lr=trn_params.learning_rate, decay=trn_params.learning_decay,
                #          		  momentum=trn_params.momentum, nesterov=trn_params.nesterov)

				# compile the model
				my_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy','mean_squared_error'])

				# early stopping to avoid overtraining
				earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=25, verbose=0,
                                                        mode='auto')

				# Train model
				init_trn_desc = my_model.fit(data[train_ids], target[train_ids],
                                          		  epochs=trn_params.n_epochs,
                                          		  batch_size=trn_params.batch_size,
                                          		  callbacks=[earlyStopping],
                                          		  verbose=trn_params.train_verbose,
                                                  class_weight=class_weight,
                                          		  validation_data=(data[test_ids],
                                                          		   target[test_ids]),
                                          		  shuffle=True)

                #saving the model and trn in the respective training folder
				fileModelPCD = os.path.join(path, "{0}comp_model_PCD.h5".format(ipcd))
				fileTrnPCD = os.path.join(path, "{0}comp_trn_PCD.jbl".format(ipcd))

				my_model.save(fileModelPCD)

                		self.best_desc['epochs'] = init_trn_desc.epoch
                		self.best_desc['acc'] = init_trn_desc.history['acc']
                		self.best_desc['loss'] = init_trn_desc.history['loss']
                		self.best_desc['val_loss'] = init_trn_desc.history['val_loss']
                		self.best_desc['val_acc'] = init_trn_desc.history['val_acc']

                		joblib.dump([self.best_desc],fileTrnPCD,compress=9)


				self.models[ipcd] = my_model
				self.trn_descs[ipcd] = self.best_desc
                		self.pcds[ipcd] = my_model.layers[2].get_weights()[0]

				ipcd+=1

				if trn_params.verbose :
					print 'PCD %i - Train process is done, val_cost: %1.5f'%(ipcd+1,np.min(init_trn_desc.history['val_loss']))
			else:
				my_model = Sequential()

				# add a linear layer to isolate the input of NN model
				my_model.add(Dense(data.shape[1],input_dim=data.shape[1], kernel_initializer='identity',trainable=False))

				my_model.add(Activation('linear'))

				# add a non-linear single neuron layer to compress all information
				my_model.add(Dense(1, input_dim=data.shape[1], kernel_initializer='uniform'))
				my_model.add(Activation('tanh'))

				# add a non-linear output layer with max sparse target shape
				my_model.add(Dense(target.shape[1], kernel_initializer='uniform'))
				my_model.add(Activation('tanh'))

				# creating a optimization function using steepest gradient
				sgd = SGD(lr=trn_params.learning_rate, decay=trn_params.learning_decay,
                          		  momentum=trn_params.momentum, nesterov=trn_params.nesterov)

				# compile the model
				my_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy','mean_squared_error'])

				# early stopping to avoid overtraining
				earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=25, verbose=0,
                                                        mode='auto')

				# remove the projection of previous extracted pcds from random init weights
				w = my_model.layers[2].get_weights()[0]
				w_proj = np.zeros_like(w)

				# loop over previous pcds
				for i_old_pcd in range(ipcd):
					w_proj = (w_proj + (np.inner(np.inner(self.pcds[i_old_pcd],w),self.pcds[i_old_pcd].T)/
							    np.inner(self.pcds[i_old_pcd].T,self.pcds[i_old_pcd].T)))
				w_remove_proj = w - w_proj
				weights = my_model.layers[2].get_weights()
				weights[0] = w_remove_proj
				my_model.layers[2].set_weights(weights)

				# remove the projection of previous extracted pcds from data
				data_proj = np.zeros_like(data)

				# loop over previous pcds
				for i_old_pcd in range(ipcd-1):
					data_proj = (data_proj + (np.inner(np.inner(self.pcds[i_old_pcd].T,data).T,self.pcds[i_old_pcd])/
								  np.inner(self.pcds[i_old_pcd].T,self.pcds[i_old_pcd].T)))

				data_with_proj = data - data_proj

				# Train model
				init_trn_desc = my_model.fit(data_with_proj[train_ids], target[train_ids],
                                          		  epochs=trn_params.n_epochs,
                                          		  batch_size=trn_params.batch_size,
                                          		  callbacks=[earlyStopping],
                                          		  class_weight=class_weight,
                                          		  verbose=trn_params.train_verbose,
                                          		  validation_data=(data[test_ids],
                                                          		   target[test_ids]),
                                          		  shuffle=True)

                		#saving the model and trn in the respective training folder
                		fileModelPCD = os.path.join(path, "{0}comp_model_PCD.h5".format(ipcd))
                		fileTrnPCD = os.path.join(path, "{0}comp_trn_PCD.jbl".format(ipcd))

                		my_model.save(fileModelPCD)

                		self.best_desc['epochs'] = init_trn_desc.epoch
                		self.best_desc['acc'] = init_trn_desc.history['acc']
                		self.best_desc['loss'] = init_trn_desc.history['loss']
                		self.best_desc['val_loss'] = init_trn_desc.history['val_loss']
                		self.best_desc['val_acc'] = init_trn_desc.history['val_acc']

                		joblib.dump([self.best_desc],fileTrnPCD,compress=9)


                		self.models[ipcd] = my_model
                		self.trn_descs[ipcd] = self.best_desc
                		self.pcds[ipcd] = my_model.layers[2].get_weights()[0]

				#condition for choose the best numbers of PCDs components

				self.n_components = ipcd + 1

				degree_matrix = self.get_degree_matrix().astype("double")

                		mean_without_diagonal = degree_matrix.sum()/(degree_matrix.size - degree_matrix.diagonal().size)

				#condition = degree_matrix[ipcd,:-1].mean()
				print mean_without_diagonal
				if mean_without_diagonal > 80:
					break

				ipcd+=1

                		if trn_params.verbose:
                    			print 'PCD %i - Train process is done, val_cost: %1.5f'%(ipcd+1,np.min(init_trn_desc.history['val_loss']))

		return self

	def sort_components(self):
        	'''
			function that orders the components of pcd according to the
		orthogonality between them

		'''

		def mean(x):
			return x.sum()/(x.size-1)
		tmpPCDs = {}
		tmpTrnDescs = {}
		tmpModel = {}

		mtpd = pd.DataFrame(self.get_degree_matrix())

		lis = mtpd.apply(mean,axis=1)
		list_sort = lis.argsort()[::-1][:].values

		for index,value in enumerate(list_sort):
			tmpPCDs[index] = self.pcds[value]
			tmpTrnDescs[index] = self.trn_descs[value]
			tmpModel[index] = self.models[value]

		self.pcds=tmpPCDs
		self.trn_descs = tmpTrnDescs
		self.models = tmpModel
		return list_sort

	def loadPCDs(self,path,n_components):
		"""
		load model train of PCD in especific path
			path: path for load each model
			n_components: number of componentes for load
		"""

		if self.n_components == 'Auto':
			print "doesn't exist PCDs file models"
			return 1

		if self.n_components == None:
			n_components = self.n_components

		print "loading models and analysis of PCDs in fold: {0}".format(path)
		self.models.clear()
		self.trn_descs.clear()
		self.pcds.clear()

		for ipcd in range(n_components):
            		fileModelPCD = os.path.join(path, "{0}comp_model_PCD.h5".format(ipcd))
            		fileTrnPCD = os.path.join(path, "{0}comp_trn_PCD.jbl".format(ipcd))
            		if os.path.exists(fileModelPCD):
                		self.models[ipcd] = load_model(fileModelPCD)
                		self.trn_descs[ipcd] = joblib.load(fileTrnPCD)
                		self.pcds[ipcd] = load_model(fileModelPCD).layers[2].get_weights()[0]

		return 0

	def deletePCDs(self,path):
		"""
		delete the path
		"""

		if self.n_components == 'Auto':
			print "doesn't exist PCDs file models"
			return 1

		for ipcd in range(self.n_components):
            		fileModelPCD = os.path.join(path, "{0}comp_model_PCD.h5".format(ipcd))
            		fileTrnPCD = os.path.join(path, "{0}comp_trn_PCD.jbl".format(ipcd))
            		os.remove(fileModelPCD)
			os.remove(fileTrnPCD)
		os.rmdir(path)
		print "the folder {0} has been deleted".format(path)
		return 0



	def transform_with_activation_function(self, data, num_components=None):
		"""
			PCD Independent auxiliar transform function
				data: data to be transformed (events x features)
		"""

		if num_components == None:
            		num_components = self.n_components

		output = []
		for ipcd in range(num_components):
			# get the output of an intermediate layer
			# with a Sequential model
			get_layer_output = K.function([self.models[ipcd].layers[0].input],[self.models[ipcd].layers[3].output])
			pcd_output = get_layer_output([data])[0]
			if ipcd == 0:
				output = pcd_output
			else:
				output = np.append(output,pcd_output, axis=1)
		return output

	def transform_without_activation_function(self, data,num_components=None):
		"""
			PCD Independent auxiliar transform function
				data: data to be transformed (events x features)
		"""

        	if num_components == None:
            		num_components = self.n_components

		output = []
		for ipcd in range(num_components):
			# get the output of an intermediate layer
			# with a Sequential model
			get_layer_output = K.function([self.models[ipcd].layers[0].input],[self.models[ipcd].layers[2].output])
			pcd_output = get_layer_output([data])[0]
			if ipcd == 0:
				output = pcd_output
			else:
				output = np.append(output,pcd_output, axis=1)
		return output

	def transform(self, data, use_activation=False,num_components=None):
		"""
			PCD Independent transform function
				data: data to be transformed (events x features)
				use_activation: boolean to use or not the activation function
		"""
		if use_activation:
			return self.transform_with_activation_function(data,num_components)
		else:
			return self.transform_without_activation_function(data,num_components)

	def get_degree_matrix(self):
		degree_matrix = np.zeros([self.n_components,self.n_components])

		if self.models == {}:
			return degree_matrix

		for ipcd in range(self.n_components):
			for jpcd in range(self.n_components):
				degree = (np.inner(self.pcds[ipcd].T,self.pcds[jpcd].T)/
					 (np.linalg.norm(self.pcds[ipcd])*np.linalg.norm(self.pcds[jpcd])))
				degree = round(degree.real,6)
				degree = np.arccos(degree)
				degree = np.degrees(degree)
				if degree > 90 and degree < 180:
					degree = degree - 180
				if degree > 180 and degree < 270:
					degree = degree - 180
				degree_matrix[ipcd,jpcd] = np.abs(degree)
		return degree_matrix
