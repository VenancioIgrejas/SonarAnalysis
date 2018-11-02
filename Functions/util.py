import time
import gc
import os
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{0}] done in {1}'.format(name,time.time() - t0))


def check_mount_dict(dict_defaul,list_of_tubles):

    if list_of_tubles is None:
        return dict_defaul

    dict_pass = dict(list_of_tubles)

    for keys,values in dict_pass.items():
        dict_defaul[keys] = dict_pass[keys]
        return dict_defaul

def update_paramns(old_dict,new_dict,exception=[]):
    for keys in old_dict:
        if not keys in exception:
            old_dict[keys] = new_dict[keys]
    return old_dict


def get_objects(object,name,kwargs):
    obj = getattr(object,name)
    return obj(**kwargs)
