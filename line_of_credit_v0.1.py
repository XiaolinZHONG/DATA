#coding=utf-8
#@author:xiaolin
#@file:temp.py
#@time:2016/8/10 17:22

###导入数据
from Data_Gather_Preprocess import *
import time, timeit
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
dgp=Data_Gather_Preprocess()
adress='D:\\data\\flashodermoney3.csv'
df_2,label=dgp.Read_Data(adress,0,0)


import numpy as np
score=np.array(df_2[['score']])
paystatus=np.array(df_2[['paystatus']])
quantity=np.array(df_2[['quantity']])
price=np.array(df_2[['price']])
cost=np.array(df_2[['cost']])








##求解样本数据的违约率
def ratiocal(start,end,leng):
    sum1, bad1 = 0, 0
    step=(end-start)/leng
    bad=[]
    gd=[]
    rat=[]
    for j in range(leng):
        for i in range(len(score)):
            if  score[i] >=start and score[i]<start+step:
                if paystatus[i]==1:
                    sum1+=1
                else:
                    bad1+=1

        start=start+step
        # print bad1,sum1,float(float(bad1)/float(sum1))
        rat.append(float(float(bad1)/float(sum1)))
        bad.append(bad1)
        gd.append(sum1)
    return bad,gd,rat
#求解样本数据的违约率的变化率
bad,gd,rat=ratiocal(600,850,50)
x=range(600,850,5)

# sns.pairplot(df_2,vars=['score','cost','price'],hue='paystatus')
# plt.title('Default rate by score')
# plt.xlabel('score')
# # plt.ylabel('default rate')

print np.mean(rat)
rat2=[]
for i in range(len(rat)-1):
    rat2.append(float(float(rat[i+1]-rat[i])/float(rat[i])))
    # print rat2
r1=np.mean(rat2)#得到平均违约率的变化率
# x=range(len(rat)-1)
# plt.scatter(x,rat2)
# plt.show()
print r1

import seaborn as sns
import matplotlib.pylab as plt
# from scipy.optimize import leastsq

#额度计算模型
def SXED(score,r,price,quantity,c):
    '''
    :param score: 信用分数
    :param r: 变化率，是基于违约率的变化率的风控要求得到，例如：这里是r=r1/10
    :param price: 用户平均消费价格
    :param quantity: 用户平均消费间夜数
    :param c: 修正常数，用来设置最大授信额度等
    :return: 授信额度
    '''
    return c+score+price*quantity+float((np.log((600*score)/((860-600)*(860-score))))/r)
#测试数据

print SXED(600,0.000681,300,1,-1000)
print SXED(600,0.000681,600,1,-1000)
print SXED(800,0.000681,300,1,-1000)
print SXED(800,0.000681,600,6,-1000)
print SXED(750,0.000681,300,2,-1000)
print SXED(850,0.0030,300,1.5,-1000)

#基于样本数据的授信分布和真实分布
y1=[]
for i in range(len(score)):
    s=score[i]
    p=price[i]
    q=quantity[i]
    y1.append(SXED(s,0.000681,p,q,-1000))
sns.distplot(y1)
sns.distplot(cost)
plt.xlim(0,10000)
plt.show()

####min风控最小授信额度
y=[]
x=range(0,855)
for i in range(0,855,1):
    y.append(SXED(i,0.0030,300,1.5,0))
plt.plot(x,y,label='the min line of credit ')

##测试数据分布
y=[]
x=range(0,855)
for i in range(0,855,1):
    y.append(SXED(i,0.00068,300,1.5,-1000))
plt.plot(x,y,label='the line of credit ')
plt.scatter(np.ravel(score),y1)

####max 风控最大授信额度
y=[]
x=range(0,855)
for i in range(0,855,1):
    y.append(SXED(i,0.000681,300,1.5,2500))
plt.plot(x,y,label='the max line of credit')
plt.legend()
plt.xlim(550,870)
plt.title('Line of Credit')
plt.show()
