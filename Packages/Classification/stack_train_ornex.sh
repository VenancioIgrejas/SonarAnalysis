#!/bin/bash

neurons=10
process=2
folds=10


#python ./HierarqNet/pool_train.py -n $neurons -p $process -f $folds --database 24classes
#python ./HierarqNet/pool_train.py -n $neurons -p $process -f $folds --database 31classes

for i in 10
do
	python ./Specialist/pool_train.py -n $i -p $process -f $folds --database 31classes
done


#python ./Specialist/pool_train.py -n $neurons -p $process -f $folds --database 31classes
#python ./Specialist/pool_train.py -n $neurons -p $process -f $folds --database 24classes

#python ./PCDHierarqNet/pool_train.py -n $neurons -p $process -f $folds --database 24classes
#python ./PCDHierarqNet/pool_train.py -n $neurons -p $process -f $folds --database 31classes

#python ./PCDSpecNeuralNetwork/pool_train.py -n $neurons -p $process -f $folds --database 24classes
#python ./PCDSpecNeuralNetwork/pool_train.py -n $neurons -p $process -f $folds --database 31classes
