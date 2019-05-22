from .weight_boosting import AdaBoost
#from .multiclass import SpecialistClass,HierarqNet
from .hierarque_classification import HierarqClassification
from .specialist_classification import SpecialistClassification

__all__ = ["AdaBoost",
           #"SpecialistClass",
           #"HierarqNet",
           "SpecialistClassification",
           "HierarqClassification"]
