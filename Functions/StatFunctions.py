"""
  This file contents stat functions
"""

import numpy as np

def KLDiv(p,q):
	if len(p) != len(q):
		print 'Different dimensions'
		return -1

	kl_values = []
	for i in range(len(p)):
		if p[i] == 0 or q[i] == 0 :
			kl_values = np.append(kl_values,0)
		else:
			kl_value = np.absolute(p[i])*np.log10(np.absolute(p[i])/np.absolute(q[i]))
			if np.isnan(kl_value):
				kl_values = np.append(kl_values,0)
			else:
				kl_values = np.append(kl_values,kl_value)
			#print "KLDiv: p= %f, q=%f, kl_div= %f"%(p[i],q[i],kl_values[i])

	return [np.sum(kl_values),kl_values]

def sp(eff):
    """
    SP of list
        paramter is a list with all efficient of each class
        
    References:
        Thesis TESE - Classificacao Neural de Sinais de Sonar Passivo - Joao Baptista 2007 - UFRJ pg.111
        
    """
    if (type(eff)==list):
        eff = np.array(eff)
    np.mean(eff)
    prod_nroot = np.product(np.power(eff,1/float(eff.shape[0])))
    return np.sqrt(np.mean(eff)*prod_nroot)
