#coding=utf-8

from pybrain.unsupervised.trainers import deepbelief
from pybrain.datasets import SupervisedDataSet
from matplotlib import pyplot as plt
import numpy as np
from pybrain.utilities import percentError
from pybrain.structure import *
import pybrain
from pybrain.supervised.trainers import BackpropTrainer

net=Network()

inlayer=LinearLayer(4,name='inpt')
hiddenlayer0=SigmoidLayer(8,name='hidden0')
hiddenlayer1=SigmoidLayer(10,name='hidden1')
outlayer=SoftmaxLayer(3,name='outpt')
net.addInputModule(inlayer)
net.addModule(hiddenlayer0)
net.addModule(hiddenlayer1)
net.addModule(outlayer)
net.addModule(BiasUnit(name='bias'))

net.addConnection(FullConnection(inlayer,hiddenlayer0))
net.addConnection(FullConnection(hiddenlayer0,hiddenlayer1))
net.addConnection(FullConnection(hiddenlayer1,outlayer))
net.addConnection(FullConnection(net['bias'],hiddenlayer0))
net.addConnection(FullConnection(net['bias'],hiddenlayer1))
net.addConnection(FullConnection(net['bias'],outlayer))
net.sortModules()


#####################################
from sklearn import datasets
from matplotlib import pyplot as plt
iris=datasets.load_iris()
x=iris.data
y=iris.target
plt.scatter(x=iris.data[:,1],y=iris.data[:,3],marker='o',c=iris.target)
plt.grid(True)
plt.show()
#####################################
##input data##
#####################################
from pybrain.datasets import ClassificationDataSet
DS=ClassificationDataSet(4,1,nb_classes=3)#DS的格式是4个/列特征值，1列结果，其中1列结果有3类
for i in xrange(len(y)):
    DS.addSample(x[i],y[i])
#print DS['input'],DS['target']
trndata,tstdata=DS.splitWithProportion(0.8)
trndata._convertToOneOfMany()#将1列结果转换成3列结果，便于训练网络
tstdata._convertToOneOfMany()
########################################




trainer=deepbelief.DeepBeliefTrainer(net,dataset=trndata,epochs=50)

net.activate()










