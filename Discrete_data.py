#coding=utf-8
def Discretdata(data):
    '''
    :param data: DATA FRAME style
    :return: new discreted data
    '''
    import pandas as pd
    from pandas import DataFrame
    import numpy as np

    thres_matrix=data.describe()
    xlen=thres_matrix.shape[1]
    ylen=thres_matrix.shape[0]
    for i in xrange(data.shape[1]):
        x=data.ix[:,i].values
        for j,value in enumerate(x):
            if value > round(thres_matrix.ix[6,i]):
                x[j]=4
            elif value >round(thres_matrix.ix[5,i]):
                x[j]=3
            elif value >round(thres_matrix.ix[4,i]):
                x[j]=2
            elif value >round(thres_matrix.ix[3,i]):
                x[j]=1
            else:
                x[j]=0
    return x

if __name__ == '__main__':
    pass