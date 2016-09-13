
# coding: utf-8

# In[66]:


def QFWD(data,savpath,threshold=0.95,x=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    '''
    :param data: DataFrame style
    :return: the quantile
    '''
    import pandas
    from pandas import DataFrame
    import numpy as np


    def juge(x):
        if x > threshold2:
            return threshold2
        else:
            return x
    
    thres=data.describe([threshold])
    for j in range(data.shape[1]):
        threshold2=thres.ix[4, j]
        data1 = np.array(data.ix[:,j]).ravel()
        data2 = list(map(juge, data1))
        data.ix[:,j]=data2
    datades=data.describe(x)
    import csv
    f=open(savpath,'a+')
    for j in range(data.shape[1]):
        val=np.round(datades.ix[4:-1, j].values,2)
#         print (val)
        f.write(datades.columns[j])
        f.write('=')
        for i in range(len(val)-1):
            f.write(str(val[i]))
            f.write(',')
        f.write(str(val[-1]))
        f.write('\n')
    f.close()




def Read_csv(adress):
    
    import pandas as pd
    from pandas import DataFrame
    import numpy as np
    import matplotlib.pylab as plt

    df_0 = pd.read_csv(adress, sep=',')
    # which means that the separate by ','
    data= df_0.dropna(how='any')
    #drop the nan value row
    print ('Reading data ....')
    print ('Samples_data_shape:', data.shape)
    return data

def Get_files(path,savpath):
    '''
    :param path: the data files path
    :param savpath: the result path
    '''
    import os
    for root,dirs,item in os.walk(path):
        for file in item:
            file=root + str('//') + file
            data=Read_csv(file)
            QFWD(data,savpath)
    print ('Done!')
    
if __name__ == '__main__':
    Get_files('../data/TEMP_TEST','../data/result.txt')


# In[ ]:



