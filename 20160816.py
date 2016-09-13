# coding=utf-8
# @author:xiaolin
# @file:20160816.py
# @time:2016/8/16 12:37
import numpy as np
import sklearn
# import keras
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd

adress='D:\data\\bigtable_2B.csv'

df=pd.read_csv(adress,sep=',')
cosum_a=['ord_success_flt_last_order_days','ord_success_flt_first_class_order_amount',
         'ord_success_flt_max_order_amount','ord_success_flt_avg_order_pricerate','ord_success_flt_aboard_order_count'
        ]
data=df[cosum_a].ix[:10,:]
def Desc_Scaler(data, des=[0.95], feature_range=(0, 1)):
    from sklearn import preprocessing
    import pandas as pd
    from pandas import DataFrame

    describe=data.describe(des)
    print data.head()
    y=describe.shape[1]
    x=[]
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
    for i in range(y):
        threshold=describe.ix[-2,i]
        minval=describe.ix[3,i]
        maxval=describe.ix[-1,i]
        temp = np.array(data.ix[:,i]).ravel()

        # use the map function to discrete data
        def juge1(x):  # define the function to discrete
            if x > threshold:
                return threshold
            elif x < 0:
                return 0
            else:
                return x
        def juge2(x):  # define the function to discrete
            if x > threshold:
                return threshold
            else:
                return x
        if minval==-1:
            if maxval==1 or maxval==0:
                x_temp = temp
            else:
                x_temp = list(map(juge2, temp))
        else:
            x_temp = list(map(juge1, temp))
        x.append(x_temp)
    data_new = pd.DataFrame(np.matrix(x).T, index=data.index, columns=data.columns)
    # print data_new.head()
    # x=min_max_scaler.fit_transform()
    df_scaler = min_max_scaler.fit_transform(data_new)
    #     df_scaler=np.round(df_scaler,3)
    data_new = pd.DataFrame(df_scaler, index=data.index, columns=data.columns)

    # print data_new.head()
    # for i in range(y):
    #     sns.distplot(data_new.ix[:,i])
    #     plt.show()
    return data_new

Desc_Scaler(data,[0.95])