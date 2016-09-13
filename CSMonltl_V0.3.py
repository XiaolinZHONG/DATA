#coding=utf-8
#@author:xiaolin
#@file:CSMonltl_V0.3.py
#@time:2016/8/16 10:27

###导入数据
from Data_Gather_Preprocess import *
import time, timeit
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from pandas import DataFrame
import numpy as np
dgp=Data_Gather_Preprocess()
adress='D:\\data\\ltl_credit_model_30.csv'
df_0=pd.read_csv(adress,sep=',')
df_1=df_0.dropna(how='any')
print 'Reading data ....'
#get the data and the target/label
data=df_1.ix[:,0:-1]
label=df_1.ix[:,-1]
print 'Samples_data_shape:',data.shape
print '#'*50

noshow=['recent20_noshow_rate','recent15_noshow_rate',
        'recent10_noshow_rate']
people=['age','customer_value','real_name_certify','gender','bound_valid_email',
        'registerduration','member_level']
pricorder=['max_deal_price','avg_deal_price','fguarantee_last_year_deal_orders',
          'last_6m_deal_orders','last_6m_deal_amount','last_year_deal_orders','last_year_deal_amount']
prefer=['htl_consuming_capacity','avg_htl_star','ratio_business_htl','htl_star_prefer',
        'flt_price_sitivity','htl_advanced_date','flt_generous_index']
other=['generous_sting_tag','profile_advanced_date','ctrip_cash_amount',
       'credit_card_succ_rate','phone_type','base_active']

print '#############################'
print '##    对noshow数据处理   ####'
print '#############################'
#首先查看数据的分布，发现有很多数据的取值是-1，数据主要分布式是-1,0,1.
#暂时不对noshow数据做处理。

print '#############################'
print '##    对people数据处理   ####'
print '#############################'
# dgp.Plot_data_df(data[people],label,people)

age=dgp.Discretedata(data[['age']])
# print age.head()
'''
age 中0岁和100岁以上的的年龄实际上是违规的，但是我们发现，0岁的违约率非常高，所以不剔除0岁。
我们直接对数据进行等距切分
'''
# dgp.Plot_data_df(age,label,['age'])
regduration=dgp.Discretedata(data[['registerduration']])
# print regduration.head()
# dgp.Plot_data_df(regduration,label,['registerduration'])
customer_value=dgp.Discretedata(data[['customer_value']])
# print customer_value.head()
#拼接age 和 regduration 到 people_new
people_old=['real_name_certify','gender','bound_valid_email','member_level']
people_new=pd.concat([data[people_old],age,regduration,customer_value],axis=1)
print people_new.head() #不需要进一步处理直接使用

print '################################'
print '##    对pricorder数据处理   ####'
print '################################'

# dgp.Plot_data_df(data[pricorder],label,pricorder,pair=False)
#通过观察发现数据比较大需要进行离散化,在进行离散化之前需要对一些异常值进行处理
from sklearn.preprocessing import RobustScaler
rscaler=RobustScaler()
pricorder_temp=rscaler.fit_transform(data[pricorder])
# pricorder_new= pd.DataFrame(pricorder_temp, index=data.index, columns=pricorder)
#scaler the data to the IQR range
# print pricorder_new.head()
# print DataFrame(pricorder_new.describe())
# dgp.Plot_data_df(pricorder_new,label,pricorder,pair=False)

pricorder=['fguarantee_last_year_deal_orders',
          'last_6m_deal_orders','last_6m_deal_amount','last_year_deal_orders',
           'last_year_deal_amount']

max_deal_price=dgp.Balance_Convert(data[['max_deal_price']])
# dgp.Plot_data_df(max_deal_price,label,['max_deal_price'])
avg_deal_price=dgp.Balance_Convert(data[['avg_deal_price']])
# dgp.Plot_data_df(avg_deal_price,label,['avg_deal_price'])

last_year=np.array(data[['last_year_deal_orders']]).ravel()
def juge(x):
    if x>50:
        return 50
    elif x<0:
        return 0
    else:
        return x
last_year2=map(juge,last_year)
last_year_deal_orders=DataFrame(last_year2,index=data.index,
                                columns=['last_year_deal_orders'])

# dgp.Plot_data_df(last_year_deal_orders,label,['last_year_deal_orders'])
last_year_deal_orders=dgp.Discretedata(last_year_deal_orders)

last_year=np.array(data[['last_year_deal_amount']]).ravel()
def juge(x):
    if x>60000:
        return 60000
    elif x<0:
        return 0
    else:
        return x
last_year2=map(juge,last_year)
last_year_deal_amount=DataFrame(last_year2,index=data.index,
                                columns=['last_year_deal_amount'])
# dgp.Plot_data_df(last_year_deal_amount,label,['last_year_deal_amount'])
last_year_deal_amount=dgp.Discretedata(last_year_deal_amount)
# dgp.Plot_data_df(last_year_deal_amount,label,['last_year_deal_amount'])

f_last_year=np.array(data[['fguarantee_last_year_deal_orders']]).ravel()
def juge(x):
    if x>60:
        return 60
    elif x<0:
        return 0
    else:
        return x
last_year2=map(juge,f_last_year)
fguarantee_last_year_deal_orders=DataFrame(last_year2,index=data.index,
                                columns=['fguarantee_last_year_deal_orders'])
# dgp.Plot_data_df(fguarantee_last_year_deal_orders,label,['fguarantee_last_year_deal_orders'])
fguarantee_last_year_deal_orders=dgp.Discretedata(fguarantee_last_year_deal_orders)
# dgp.Plot_data_df(last_year_deal_amount,label,['last_year_deal_amount'])

last_6m=np.array(data[['last_6m_deal_orders']]).ravel()
def juge(x):
    if x>60:
        return 60
    elif x<0:
        return 0
    else:
        return x
last_6m2=map(juge,last_6m)
last_6m_deal_orders=DataFrame(last_6m2,index=data.index,
                                columns=['last_6m_deal_orders'])
# dgp.Plot_data_df(last_6m,label,['last_6m_deal_orders'])
last_6m_deal_orders=dgp.Discretedata(last_6m_deal_orders)
# dgp.Plot_data_df(last_6m_deal_orders,label,['last_6m_deal_orders'])

last_6m=np.array(data[['last_6m_deal_amount']]).ravel()
def juge(x):
    if x>30000:
        return 30000
    elif x<0:
        return 0
    else:
        return x
last_6m2=map(juge,last_6m)
last_6m_deal_amount=DataFrame(last_6m2,index=data.index,
                                columns=['last_6m_deal_amount'])
# dgp.Plot_data_df(last_6m,label,['last_6m_deal_amount'])
last_6m_deal_amount=dgp.Discretedata(last_6m_deal_amount)
# dgp.Plot_data_df(last_6m_deal_amount,label,['last_6m_deal_amount'])

#拼接数据

pricorder_new=pd.concat([max_deal_price,avg_deal_price,fguarantee_last_year_deal_orders,
          last_6m_deal_orders,last_6m_deal_amount,last_year_deal_orders,
           last_year_deal_amount],axis=1)
print pricorder_new.head()

#后面继续使用该数据时需要继续归一化到range(0-1)

print '##############################'
print '##    对prefer数据处理    ####'
print '##############################'


# dgp.Plot_data_df(data[prefer],label,prefer,dis=False)
# print data[prefer].describe()
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
# sns.distplot(data[['htl_advanced_date']])#有人会提前300天定房间？
# plt.show()

# advdate_temp=dgp.Discretedata(data[['htl_advanced_date']])
# sns.distplot(advdate_temp)
'''这种数据分布不适合使用等距离散化'''
# plt.show()
htl_advan_date=np.array(data[['htl_advanced_date']]).ravel()
# def why(x):
#     index=[]
#     for i in range(len(x)):
#         if x[i]>=250:
#             index.append(i)
#     return index
# index=why(htl_advan_date)
# for i in index:
#     print data.ix[i]
#上面的代码是查看异常的提前预定天数的人有没有特别之处
def juge(x):
    if x>30:
        return 31
    else:
        return x
htl_advan_date2=map(juge,htl_advan_date)
htl_advanced_date=DataFrame(htl_advan_date2,index=data.index,columns=['htl_advanced_date'])
# sns.distplot(htl_advanced_date)
# plt.show()
#拼接数据合成新的prefer
prefer=['htl_consuming_capacity','avg_htl_star','ratio_business_htl','htl_star_prefer',
        'flt_price_sitivity','flt_generous_index']
prefer_new=pd.concat([data[prefer],htl_advanced_date],axis=1)
print 'prefer_new_date'
print prefer_new.head()

print '#############################'
print '##    对other数据处理   #####'
print '#############################'

other=['profile_advanced_date','ctrip_cash_amount',
       'credit_card_succ_rate','phone_type','base_active']
# dgp.Plot_data_df(data[other],label,other)
#通过数据的分布图我们可以看到，相应的profile_advanced_date
#ctrip_cash_amount的数据分布比较的不合规则需要进行处理

#处理profile_advanceed_date，小于0，大于50的全部分别设置为0或50
#然后对数据进行等距离散化
profile_advdate=np.array(data[['profile_advanced_date']]).ravel()
def juge(x):
    if x>30:
        return 31
    elif x<0:
        return 0
    else:
        return x
profile_advdate2=map(juge,profile_advdate)
profile_advanced_date=DataFrame(profile_advdate2,index=data.index,columns=['profile_advanced_date'])

# dgp.Plot_data_df(profile_advanced_date,label,['profile_advanced_date'])
profile_advanced_date=dgp.Discretedata(profile_advanced_date)#对已经进行异常值处理的进行离散化
# dgp.Plot_data_df(profile_advanced_date,label,['profile_advanced_date'])
# print profile_advanced_date.head()
#处理ctrip_cash_amount
# ctrip_cash_amount=dgp.Discretedata(data[['ctrip_cash_amount']])
#通过分析发现5000块钱以上的人太少，当做异常处理全部赋值为5000
#后面我们会尝试使用Robust来测试
ctrip_cash=np.array(data[['ctrip_cash_amount']]).ravel()
def juge(x):
    if x>5000:
        return 5000
    # elif x<0:
    #     return 0
    else:
        return x
ctrip_cash2=map(juge,ctrip_cash)
ctrip_cash_amount=DataFrame(ctrip_cash2,index=data.index,columns=['ctrip_cash_amount'])
# dgp.Plot_data_df(ctrip_cash_amount,label,['ctrip_cash_amount'])
ctrip_cash_amount=dgp.Discretedata(ctrip_cash_amount)
# dgp.Plot_data_df(ctrip_cash_amount,label,['ctrip_cash_amount'])
#使用Robust来测试
# from sklearn.preprocessing import RobustScaler
# rscaler=RobustScaler()
# ctrip_cash_temp=rscaler.fit_transform(data[['ctrip_cash_amount']])
# ctrip_cash2= pd.DataFrame(ctrip_cash_temp, index=data.index, columns=['ctrip_cash_amount'])
# dgp.Plot_data_df(ctrip_cash2,label,['ctrip_cash_amount'])
#robust把数据压缩到了5000以下，但是分布形式还是保持原来的形式

#拼接数据
other=['credit_card_succ_rate','phone_type','base_active']
other_new=pd.concat([data[other],ctrip_cash_amount,profile_advanced_date],axis=1)

print other_new.head()

###拼接全部数据###
data_new=pd.concat([data[noshow],people_new,pricorder_new,prefer_new,other_new],axis=1)

print '#'*50
print 'modelling.......'
print '#'*50


from sklearn.cross_validation import train_test_split
label=np.array(label).ravel()
data_trn,data_tst,label_trn,label_tst=\
    train_test_split(data_new,label,test_size=0.2)
#
# from sklearn.linear_model import LogisticRegression
#
# lr=LogisticRegression(penalty='l2',C=10)
# print 'using logistic regression'
# lr.fit(data_trn,label_trn)
# proba=lr.predict_proba(data_tst)
# tst_pre=lr.predict(data_tst)
# print 'score:',lr.score(data_tst,label_tst)
# # print type(label_tst)
# # print proba
#
# count_tn,count_fp,count_fn,count_tp=0,0,0,0
# for i in xrange(len(label_tst)):
#     if label_tst[i]==0:
#         if proba[i,0]>0.5:
#             count_tn+=1
#         else:
#             count_fp+=1
#     else:
#         if proba[i,0]>0.5:
#             count_fn+=1
#         else:
#             count_tp+=1
# print 'Total:', len(label_tst)
# print 'FP被分为好人的坏人:', count_fp, 'TN正确分类的坏人:', count_tn
# print 'FN被分为坏人的好人:', count_fn, 'TP正确分类的好人:', count_tp
#
# from sklearn.metrics import roc_curve
# fpr,tpr,thresholds=roc_curve(label_tst,proba[:,1])
# plt.plot(fpr,tpr,)
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.show()

from sklearn.ensemble import RandomForestRegressor
print 'using random forest classify......'
rf=RandomForestRegressor()

rf.fit(data_trn,label_trn)
rf_pre=rf.predict(data_tst)
# print rf_pre
# rf_pro=rf.predict_proba(data_tst)
# print 'score:',np.mean(rf_pre==label_tst)
count_tn,count_fp,count_fn,count_tp=0,0,0,0
for i in xrange(len(label_tst)):
    if label_tst[i]==0:
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

# from sklearn.metrics import roc_curve
# fpr,tpr,thresholds=roc_curve(label_tst,rf_pro[:,1])
# plt.plot(fpr,tpr,)
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.show()
