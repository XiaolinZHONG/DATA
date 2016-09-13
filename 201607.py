#coding=utf-8

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing


data_0= pd.read_csv('D:\\data\\uidfeature2.csv',sep=',')#读取数据，注意如果是LINUX应该加..
df_0=DataFrame(data_0)


#去掉任何数据包含Nan的样本
df_1=df_0.dropna(how='any')
#print df_1.head()

#df_0.fillna()
#填充空值

print '#'*50
#创建样本和标签数据，数组形式
label=np.array(df_1.ix[:,1])
data_1=np.array(df_1.ix[:,2:])
print 'Samples_data_shape:',data_1.shape
df_2_balance=df_1.loc[:,'balance_tmoney':'balance_refund']
print 'Balance data:','\n',df_2_balance.head()
print '#'*50
print 'Balance data describe:','\n',df_2_balance.describe()
print '#'*50
print 'Balance data columns:','\n',df_2_balance.columns

def convert(x):
    for i,value in enumerate(x.values):
        if value>10000:
            x.values[i]=4
        elif value>2000:
            x.values[i]=3
        elif value>500:
            x.values[i]=2
        elif value>0:
            x.values[i]=1
        else:
            x.values[i]=0
    return x

#df_2_balance.apply(convert)
for i in xrange(df_2_balance.shape[1]):
    convert(df_2_balance.ix[:,i])

print df_2_balance