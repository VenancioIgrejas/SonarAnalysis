import subprocess
import sys
import os
from multiprocessing import Pool, TimeoutError

sys.path.insert(0,'/home/venancio/Workspace/SonarAnalysis/Packages/Classification')




import argparse
#Argument Parse config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-n","--neurons", default=10,type=int, help="choose True for train PCDs or false for Specialist")
parser.add_argument("-p","--processes", default=3,type=int, help="choose True for train PCDs or false for Specialist")
parser.add_argument("-f","--fold", default=10,type=int, help="choose True for train PCDs or false for Specialist")
args = parser.parse_args()

n_folds = args.fold

if args.processes == -1:
    num_processes = multiprocessing.cpu_count()
else:
    num_processes = args.processes

params_for_train = []

for ifold in range(n_folds):
    params_for_train.append((args.neurons,ifold))
  
#print "list of specialist that will train: {0}".format(spc_class)
def f_spec_train(params):
    n_neurons = params[0]
    ifold = params[1]
    
    p = subprocess.call(["python", "train_24classes.py",
                         "-n",str(n_neurons),
                         "-i","10",
                         "-f","10",
                         "-b","512",
                         "--dev",str(False),
                         "--ifold",str(ifold)])
                         #"--output-activation","sigmoid","--hidden-activation",
                         #"relu","--loss","binary_crossentropy"])
    return 0

for paramns in params_for_train:
    f_spec_train(paramns)

#p = Pool(processes=num_processes)
#p.map(f_spec_train,params_for_train)
#p.close()
#p.join()