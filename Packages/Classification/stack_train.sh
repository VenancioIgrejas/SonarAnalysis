#!/bin/bash

neurons=10
process=6
folds=2

#python ./HierarqNet/pool_train.py -n $neurons -p $process -f $folds --database 24classes
#python ./HierarqNet/pool_train.py -n $neurons -p $process -f $folds --database 31classes

for i in 50 100
do
	python ./Specialist/pool_train.py -n $i -p $process -f $folds --database 24classes
done

#python ./Specialist/pool_train.py -n $neurons -p $process -f $folds --database 24classes

#python ./PCDHierarqNet/pool_train.py -n $neurons -p $process -f $folds --database 24classes
#python ./PCDHierarqNet/pool_train.py -n $neurons -p $process -f $folds --database 31classes

#python ./PCDSpecNeuralNetwork/pool_train.py -n $neurons -p $process -f $folds --database 24classes
#python ./PCDSpecNeuralNetwork/pool_train.py -n $neurons -p $process -f $folds --database 31classes


#python ./MLP/train_mlp.py -n $i -p $process -f $folds -i 3 -b 64 --database 31classes
#python ./MLP/train_mlp.py -n $i -p $process -f $folds -i 3 -b 512 --database 24classes