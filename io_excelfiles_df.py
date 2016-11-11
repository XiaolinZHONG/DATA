
import re
import os
import xlrd
import csv
import pandas as pd
from pandas import DataFrame

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def readExcel(file):
    '''
    :param file: excel 文件
    :return: data frame
    '''
    datadf=pd.read_excel(file,header=4)#从这一行开始读取数据
    datadf=datadf.drop(datadf.index[-4:-1])#舍弃掉后面几行
    datadf=datadf.ix[:,(2,4,10)]
    return datadf

def getFiles(datapath,savepath):
    '''
    :param datapath: 存放excel 的文件夹
    :param savepath: 保存结果的路径 包含了文件名和后缀，后缀为 TXT
    :return: 直接保存文件
    '''
    tmp=DataFrame()
    for root,dirs,item in os.walk(datapath):
        i=1
        for file in item:
            file2 = root +str('//')+file
            print "READING THE NUM",i,"FILE"
            data = readExcel(file2)
            tmp=pd.concat([tmp,data],ignore_index=True)
            i+=1
    tmp.to_csv(savepath,index=False)
    print ("DONE! ")

if __name__ == '__main__':
    datapath="D:/data/moneydata"
    savepath="D:/data/moneydata2/result.txt"
    getFiles(datapath,savepath)
