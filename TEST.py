
#coding=utf-8

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
adress1='D:/data/ltl_credit_model_woe.csv'
reader=pd.read_csv(adress1,sep=',', iterator = True)
loop=True
chunkSize=10000
chunks=[]
while loop:
    try:
        chunk=reader.get_chunk(chunkSize)
        chunks.append(chunk)
    except StopIteration:
        loop=False
        print 'Iteration is stopped.'
df_0=pd.concat(chunks,ignore_index=True)
df_1=df_0.dropna(how='any')
print df_1.head()
data  =df_1.ix[:,3:] # can input the data/samples num
label =df_1.ix[:,2]   # input the corr label num
print ('Data shape:',data.shape)
import seaborn as sns
sns.set(style='darkgrid',color_codes=True)
sns.distplot(label,axlabel=False)
plt.show()
