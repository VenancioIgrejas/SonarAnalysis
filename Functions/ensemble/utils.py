import numpy as np
import pandas as pd
__all__ = ["logical_or",
           "Nothing",]


def logical_or(y, vector_unique):
	""" Compute the truth value of y elements if it's in specific_classes ,
	 	like (y(each_element)==specific_classes[0] | y(each_element)==specific_classes[1] | ... | y(each_element)==specific_classes[-1])
    

    Parameters
    ----------
    y : (1d array-like) vector that contains or not elements in specific_classes.

    vector_unique: (1d array-like or list) vector that contain or not elements of y


    Returns
    -------
    bool_vector: (list) bool list 
    """
	dt_bool = pd.concat([pd.DataFrame(y==element) for element in vector_unique],axis=1)

	vector_bool = dt_bool.apply(np.sum,axis=1).values # vector data contains only 1 and 0

	for value in np.unique(vector_bool):
		if not value in [0,1]:
			raise  ValueError("vector of bool contain {0}, "
				"but expected only number 0(means False) and 1(means True)".format(value))

	return map(lambda x:bool(x), vector_bool)

def Nothing():
	pass


