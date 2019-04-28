import subprocess
import sys
import os
from multiprocessing import Pool, TimeoutError

sys.path.insert(0,'/home/venancio/WorkPlace/SonarAnalysis/Packages/Classification')




import argparse
#Argument Parse config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--pcd", default=False,type=bool, help="choose True for train PCDs or false for Specialist")
parser.add_argument("--cpu", default=4,type=int, help="choose True for train PCDs or false for Specialist")
parser.add_argument("-s", "--specialist-list",default=3,type=int, help= "Select list of specialist for train 1:[0,1,2,3,4,5,6,7,8,9,10,11], 2:[12,13,14,15,16,17,18,19,20,21,22,23], 3:all")
args = parser.parse_args()

if args.specialist_list == 1:
    spc_class = [0,1,2,3,4,5,6,7,8,9,10,11]
elif args.specialist_list == 2:
    spc_class = [12,13,14,15,16,17,18,19,20,21,22,23]
else:
    spc_class = range(24)

def f_PCD(i):

    p = subprocess.call(["python", "trainPCD.py","-c","10","-f","0","-s",str(i),"--partition-data","test"])
    return 0

p = Pool(processes=args.cpu)
p.map(f_PCD, spc_class)
p.close()
p.join()
