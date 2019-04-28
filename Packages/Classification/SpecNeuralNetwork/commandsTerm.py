import subprocess
import sys
import os
from multiprocessing import Pool, TimeoutError

sys.path.insert(0,'/home/venancio/WorkPlace/SonarAnalysis/Packages/Classification')




import argparse
#Argument Parse config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--pcd", default=False,type=bool, help="choose True for train PCDs or false for Specialist")
parser.add_argument("-spec", "--specialist-list",default=3,type=int, help= "Select list of specialist for train 1:[0,1,2,3,4,5,6,7,8,9,10,11], 2:[12,13,14,15,16,17,18,19,20,21,22,23], 3:all")
args = parser.parse_args()

if args.specialist_list == 1:
    spc_class = [0,1,2,3,4,5,6,7,8,9,10,11]
elif args.specialist_list == 2:
    spc_class = [12,13,14,15,16,17,18,19,20,21,22,23]
else:
    spc_class = range(24)    
    
print "list of specialist that will train: {0}".format(spc_class)
def f_spec_train(i):
    p = subprocess.call(["python", "train.py","-n",str(i),"-s","20","-i","10"])
                         #"--output-activation","sigmoid","--hidden-activation",
                         #"relu","--loss","binary_crossentropy"])
    return 0

def f_PCD(i):
    #p = subprocess.call(["python", "trainPCD.py","-c","10","-f",str(i)])
    p = subprocess.call(["python", "PCDtrain.py","-c","70","-s",str(i)])
    return 0

p = Pool(processes=4)
if args.pcd == True:
    p.map(f_PCD,spc_class)
else:
    p.map(f_spec_train, [10,15,25,30,35,40,45,50])
p.close()
p.join()