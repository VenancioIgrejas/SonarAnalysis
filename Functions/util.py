import time
import gc
import os
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{0}] done in {1}'.format(name,time.time() - t0))


def check_mount_dict(dict_defaul,dict_pass):
    if dict_pass is None:
        return dict_defaul
    else:
        for keys,values in dict_pass.items():
            dict_defaul[keys] = dict_pass[keys]
        return dict_defaul

def get_objects(object,name,kwargs):
    obj = getattr(object,name)
    return obj(**kwargs)
