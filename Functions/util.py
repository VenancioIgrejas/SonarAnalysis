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
    for keys in new_dict:
        if not keys in exception:
            if keys in old_dict:
                old_dict[keys] = new_dict[keys]
    return old_dict


def get_objects(object,name,kwargs):
    """create a object of some instance and pass a dict
    for the constructor
    """
    obj = getattr(object,name)
    return obj(**kwargs)

def file_exist(path_file):
    """check if a file exists. If exists, return True
    """
    return os.path.isfile(path_file)
