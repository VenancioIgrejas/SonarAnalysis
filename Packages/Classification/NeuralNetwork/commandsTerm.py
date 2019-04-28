import sys
import subprocess
from multiprocessing import Pool, TimeoutError

def f(i):
    #p = subprocess.call(["python", "trainFast.py","-n","40","-i","2","-p","50","--each-fold",str(i+1)])
    #p = subprocess.call(["python", "temp.py","-f",str(i+1),"-p",str(sys.argv[1])])
    p = subprocess.call(["python", "trainPCD.py","--each-fold",str(i+1),"-p","80"])
    return 0

p = Pool(processes=4)
p.map(f, range(10))
p.close()
p.join()
