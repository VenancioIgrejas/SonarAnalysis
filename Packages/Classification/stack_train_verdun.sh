#!/bin/bash

pwd = $PWD
neurons=100
process=6
folds=10

#python ./HierarqNet/pool_train.py -n $neurons -p $process -f $folds --database 24classes
#python ./HierarqNet/pool_train.py -n $neurons -p $process -f $folds --database 31classes

# python ./SVM/SVM_test.py -p $process -f $folds --devSize 100 --database 31classes &
# sleep 30s
# python ./SVM/SVM_test.py -p $process -f $folds --devSize 100 --database 24classes


#for i in 50 5 10 20 100
#do
#	python ./Specialist/pool_train.py -n $i -p $process -f $folds --database 24classes
#done


for c in 0.001 0.01 0.1 10 100 1000
do
    python ./SVM/SVM_test.py -c $c -p 10 -f $folds --database 31classes
done

#python ./Specialist/pool_train.py -n $neurons -p $process -f $folds --database 24classes

#python ./PCDHierarqNet/pool_train.py -n $neurons -p $process -f $folds --database 24classes
#python ./PCDHierarqNet/pool_train.py -n $neurons -p $process -f $folds --database 31classes

#python ./PCDSpecNeuralNetwork/pool_train.py -n $neurons -p $process -f $folds --database 24classes
#python ./PCDSpecNeuralNetwork/pool_train.py -n $neurons -p $process -f $folds --database 31classes


#python ./MLP/train_mlp.py -n $i -p $process -f $folds -i 3 -b 64 --database 31classes
#python ./MLP/train_mlp.py -n $i -p $process -f $folds -i 3 -b 512 --database 24classes

#python ./PCDMLP/train_PCDmlp.py -n $neurons -p $process -f $folds -i 10 -b 64 --database 31classes
#python ./PCDMLP/train_PCDmlp.py -n $neurons -p $process -f $folds -i 10 -b 512 --database 24classes

#python ./Specialist/train_specialists.py --ifold 5 -n 100 -i 3 -f $folds -p $process -b 64 --database 31classes --dev 0

#python ./Specialist/train_specialists.py --ifold 7 -n 10 -i 3 -f $folds -p $process -b 64 --database 31classes --dev 0 
#python ./Specialist/train_specialists.py --ifold 8 -n 10 -i 3 -f $folds -p $process -b 64 --database 31classes --dev 0 

#python ./Specialist/train_specialists.py --ifold 6 -n 20 -i 3 -f $folds -p $process -b 64 --database 31classes --dev 0 
#python ./Specialist/train_specialists.py --ifold 8 -n 20 -i 3 -f $folds -p $process -b 64 --database 31classes --dev 0 

#python ./Specialist/train_specialists.py --ifold 7 -n 5 -i 3 -f $folds -p $process -b 64 --database 31classes --dev 0 