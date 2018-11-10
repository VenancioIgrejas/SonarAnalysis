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

def best_file(path_files,choose_key,path_rename_file=None):
    """ return the choose file in a dictionary of files
    ----------
    path_files : {dict}
               Dictionary of path files

    best_key : {key of dictionary}
              key of the choose file
    path_rename_file : {string}
                     rename the file choosed
    Returns
    ----------
    the chosen files' path according to choose_key
    """


    if not path_rename_file is None:
        os.rename(path_files[choose_key], path_rename_file)
        path_files[choose_key] = path_rename_file

    for keys in path_files:
        if not keys is choose_key:
            if os.path.exists(path_files[keys]):
                os.remove(path_files[keys])

    return path_files[choose_key]

def invert_dict(dicto):
    """ invert key,value in dictionary [{key:value} -> {value:key}]
    """
    tmp = []
    for key,values in dicto.items():
        if isinstance(values,(list,np.ndarray)):
            for subvalues in values:
                tmp.append((subvalues,key))
        else:
            tmp.append((values,key))

    return dict(tmp)
