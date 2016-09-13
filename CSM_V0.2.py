#coding=utf-8
#@author:xiaolin
#@file:CSM_V0.2.py
#@time:2016/8/11 10:29

###导入数据
from Data_Gather_Preprocess import *
import time, timeit
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
dgp=Data_Gather_Preprocess()
adress='D:\\data\\newmodel2.csv'
df_2,label=dgp.Read_Data(adress,0,1)

###个人信息数据处理

#个人信息全部数据
mobile=['samemobile','common_mobile_cnt','contact_mobile_count']
sign=['signmonths','is_quick_uid']
people=['grade','age','common_has_child']
valid=['emailvalid','isindentify','addressvalid']

person_info=['common_mobile_cnt','contact_mobile_count','signmonths','age']
#对mobile中的后两个数据进行离散化
mobile_temp=dgp.Discretedata(df_2[['common_mobile_cnt','contact_mobile_count']])
print mobile_temp.head() #查看离散化后的数据
# dgp.Plot_data_df(mobile_temp,label=label,strac=['common_mobile_cnt','contact_mobile_count'])
print df_2[['samemobile']].head() #查看需要拼接的数据
mobile=pd.concat([df_2[['samemobile']],mobile_temp],axis=1)#拼接数据
print mobile.head()#查看拼接后的数据
#对sign中的signmonths进行归一化（均衡化）
signmonths=dgp.Discretedata(df_2[['signmonths']])
# signmonths,signmscaler=dgp.Scaler_data(df_2[['signmonths']],strac=['signmonths'])
    #返回归一化的数据 和归一化函数，便于后面处理测试数据
# print '注册月份归一化函数：',signmscaler
print signmonths.head()#查看归一化的数据
sign=pd.concat([signmonths,df_2[['is_quick_uid']]],axis=1)#拼接
print sign.head()#查看拼接后
#对people中的age进行离散化
age_temp=dgp.Discretedata(df_2[['age']])
print age_temp.head() #查看
# dgp.Plot_data_df(age_temp,label,['age'])
#拼接people数据
people=pd.concat([age_temp,df_2[['grade','common_has_child']]],axis=1)
# print people.head()
# people=df_2[people]
#对于valid数据不需要处理
valid=df_2[valid]
#拼接全部数据
people_info=pd.concat([people,sign,mobile,valid],axis=1)
print people_info.head()
#初步处理后的数据为people_info
#对people_info 做影响因子分析
dgp.Importance_Plot(people_info,label) #结论是选三项
#对people_info 归一化后做影响因子分析
pi_snew,pi_scaler=dgp.Scaler_data(people_info,strac=people_info.columns)
print pi_snew.head()#查看归一化后的数据
# dgp.Importance_Plot(pi_snew,label)
#根据前面的分析我们选取排名前6的因子
people_info_sn=pi_snew[['signmonths','age','grade','emailvalid','addressvalid','samemobile']]
print people_info_sn.head()
print label[:5]
#最终留下的个人信息数据，LOGIST回归分析
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import numpy as np
# # np.random.shuffle(people_info_sn)# 数据随机打乱
# data_trn,data_tst,label_trn,label_tst=\
#     train_test_split(people_info_sn,label,test_size=0.7)
# # print data_tst,label_tst
# lr=LogisticRegression(penalty='l1',class_weight='balanced')
# lr.fit(data_trn,label_trn)
# proba=lr.predict_proba(data_tst)
# tst_pre=lr.predict(data_tst)
# print 'score:',lr.score(data_tst,label_tst)
# print tst_pre,label_tst
# from Learning_cure_Plot import *
# learn_curve_plot(lr,'people', people_info_sn, label, train_sizes=np.linspace(0.1, 1, 10))
#我们测试选取其中的最总要的三个变量来做回归分析
people_info_sn2=pi_snew[['signmonths','age','grade']]
print people_info_sn2.head()
print label[:5]
print '#'*10,'逻辑回归','#'*10
# dgp.Plot_data_df(people_info_sn2,label,strac=['signmonths','age','grade'])
data_trn,data_tst,label_trn,label_tst=\
    train_test_split(people_info_sn2,label,test_size=0.1)
# print data_tst,label_tst
lr=LogisticRegression(penalty='l1',C=10)
lr.fit(data_trn,label_trn)
proba=lr.predict_proba(data_tst)
tst_pre=lr.predict(data_tst)
print 'score:',lr.score(data_tst,label_tst)
# print proba
count_tn,count_fp,count_fn,count_tp=0,0,0,0
for i in xrange(len(label_tst)):
    if label_tst[i]==1:
        if proba[i,1]>0.5:
            count_tn+=1
        else:
            count_fp+=1
    else:
        if proba[i,1]>0.5:
            count_fn+=1
        else:
            count_tp+=1
print 'Total:', len(label_tst)
print 'FP被分为好人的坏人:', count_fp, 'TN正确分类的坏人:', count_tn
print 'FN被分为坏人的好人:', count_fn, 'TP正确分类的好人:', count_tp
from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(label_tst,proba[:,0])
plt.plot(fpr,tpr,)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')

plt.show()
'''
详细分析发现 由于由于正负样本的不平衡，如果不设置权重均衡，误判的数量非常大。
后面两条路走：继续更换模型来优化这一项;提取重要特征，后面统一建模
'''
#我们引入随机森林回归
print '#'*10,'随机森林','#'*10
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(max_features='auto',n_estimators=50,n_jobs=-1,oob_score=True,random_state=50)
rf.fit(data_trn,label_trn)
rf_pre=rf.predict(data_tst)

sns.distplot(rf_pre)
plt.show()
count_tn,count_fp,count_fn,count_tp=0,0,0,0
for i in xrange(len(label_tst)):
    if label_tst[i]==1:
        if rf_pre[i]>0.5:
            count_tn+=1
        else:
            count_fp+=1
    else:
        if rf_pre[i]>0.5:
            count_fn+=1
        else:
            count_tp+=1
print 'Total:', len(label_tst)
print 'FP被分为好人的坏人:', count_fp, 'TN正确分类的坏人:', count_tn
print 'FN被分为坏人的好人:', count_fn, 'TP正确分类的好人:', count_tp

from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(label_tst,rf_pre)
plt.plot(fpr,tpr,)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()

#计算分数，通过前面的分析，我们发现逻辑回归对于坏人的识别率比较高，
# 而随机深林对于好人的识别率比较高，所以我们综合两个模型
print '#'*10,'混合随机森林','#'*10
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(max_features='auto',n_estimators=50,n_jobs=-1,oob_score=True,random_state=50)
rfc.fit(data_trn,label_trn)
rfc_pre=rfc.predict_proba(data_tst)
# print 'score:',rfc.score(data_tst,label_tst)

score=100*(1-rf_pre)-50*(rfc_pre[:,1])

count_tn,count_fp,count_fn,count_tp=0,0,0,0
for i in xrange(len(label_tst)):
    if label_tst[i]==1:
        if score[i]<50:
            count_tn+=1
        else:
            count_fp+=1
    else:
        if score[i]<50:
            count_fn+=1
        else:
            count_tp+=1
print 'Total:', len(label_tst)
print 'FP被分为好人的坏人:', count_fp, 'TN正确分类的坏人:', count_tn
print 'FN被分为坏人的好人:', count_fn, 'TP正确分类的好人:', count_tp

sns.distplot(score)
plt.show()



