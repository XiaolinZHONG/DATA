#coding=utf-8
#@author:xiaolin
#@file:CSMonltl_V0.4.py
#@time:2016/8/23 13:31

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
data=df_1.ix[:100000,0:-1]
label=df_1.ix[:100000,-1]
print 'Samples_data_shape:',data.shape
print '#'*50
noshow=['recent20_noshow_rate']
people=['age','customer_value','real_name_certify','gender','bound_valid_email',
        'registerduration','member_level']
pricorder=['max_deal_price','avg_deal_price','fguarantee_last_year_deal_orders',
          'last_6m_deal_orders','last_6m_deal_amount','last_year_deal_orders','last_year_deal_amount']
prefer=['htl_consuming_capacity','avg_htl_star','ratio_business_htl','htl_star_prefer',
        'flt_price_sitivity','htl_advanced_date','flt_generous_index']
other=['generous_sting_tag','profile_advanced_date','ctrip_cash_amount',
       'credit_card_succ_rate','phone_type','base_active']


def Modelling1(data,label,testsize=0.2):
    from sklearn.cross_validation import train_test_split

    label=np.array(label).ravel()
    data_trn,data_tst,label_trn,label_tst=\
        train_test_split(data,label,test_size=testsize)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    print 'using random forest classify......'
    rf=RandomForestClassifier(criterion='gini',max_features='sqrt',
                              n_estimators=50,min_samples_leaf=10,random_state=50)
    rf.fit(data_trn,label_trn)
    rf_pre=rf.predict(data_tst)
    rf_pro=rf.predict_proba(data_tst)
    print 'score:',np.mean(rf_pre==label_tst)
    print 'AUC-ROC:',roc_auc_score(label_tst,rf_pre)
    count_tn,count_fp,count_fn,count_tp=0,0,0,0
    for i in xrange(len(label_tst)):
        if label_tst[i]==0:
            if rf_pro[i,0]>0.5:
                count_tn+=1
            else:
                count_fp+=1
        else:
            if rf_pro[i,0]>0.5:
                count_fn+=1
            else:
                count_tp+=1
    print 'Total:', len(label_tst)
    print 'FP被分为好人的坏人:', count_fp, 'TN正确分类的坏人:', count_tn
    print 'FN被分为坏人的好人:', count_fn, 'TP正确分类的好人:', count_tp

    from sklearn.metrics import roc_curve
    fpr,tpr,thresholds=roc_curve(label_tst,rf_pro[:,1])
    plt.plot(fpr,tpr,)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()
def Modelling2(data,label,testsize=0.2):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty='l2', C=10)
    print 'using logistic regression'
    from sklearn.cross_validation import train_test_split
    label = np.array(label).ravel()
    data_trn, data_tst, label_trn, label_tst = \
        train_test_split(data, label, test_size=testsize)
    lr.fit(data_trn, label_trn)
    proba = lr.predict_proba(data_tst)
    tst_pre = lr.predict(data_tst)
    print 'score:', lr.score(data_tst, label_tst)
    # print type(label_tst)
    # print proba

    count_tn, count_fp, count_fn, count_tp = 0, 0, 0, 0
    for i in xrange(len(label_tst)):
        if label_tst[i] == 0:
            if proba[i, 0] > 0.5:
                count_tn += 1
            else:
                count_fp += 1
        else:
            if proba[i, 0] > 0.5:
                count_fn += 1
            else:
                count_tp += 1
    print 'Total:', len(label_tst)
    print 'FP被分为好人的坏人:', count_fp, 'TN正确分类的坏人:', count_tn
    print 'FN被分为坏人的好人:', count_fn, 'TP正确分类的好人:', count_tp

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(label_tst, proba[:, 1])
    plt.plot(fpr, tpr, )
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()



print '##    对noshow数据处理   ####'
'''
首先查看数据的分布，发现有很多数据的取值是-1，数据主要分布式是-1,0,1.
暂时不对noshow数据做处理。
'''
# dgp.Plot_data_df(data[noshow],label,noshow)


# plt.show()
# Modelling1(data[noshow],label,0.1)
# Modelling2(data[noshow],label,0.2)
#
# from Learning_cure_Plot import learn_curve_plot
# from sklearn.ensemble import RandomForestClassifier
# rf=RandomForestClassifier(criterion='entropy',max_features='auto',n_estimators=10)
# learn_curve_plot(rf,'RandomForest',data[noshow],label)



print '##    对people数据处理   ####'

dgp.Plot_data_df(data[people],label,people)

'''
age 中0岁和100岁以上的的年龄实际上是违规的，但是我们发现，0岁的违约率非常高，所以不剔除0岁。
把60岁以上的全部归为60，其他数据离散化
'''
############################
age=np.array(data[['age']]).ravel()
def juge(x):
    if x>60:
        return 60
    elif x<0:
        return 0
    else:
        return x
age2=map(juge,age)
age=DataFrame(age2,index=data.index,columns=['age'])

age=dgp.Discretedata(age)
############################
registerduration=dgp.Discretedata(data[['registerduration']])
############################
customer_value=dgp.Discretedata(data[['customer_value']])
###########################
people_old=['real_name_certify','gender','bound_valid_email','member_level']
people_new=pd.concat([data[people_old],age,registerduration,customer_value],axis=1)

# Modelling1(people_new,label,0.3)
# Modelling2(people_new,label,0.2)
# from Learning_cure_Plot import learn_curve_plot
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(penalty='l1', C=10)
# learn_curve_plot(lr,'logistic',people_new,label)


print '##    对pricorder数据处理   ####'


'''
通过观察发现数据比较大需要进行离散化,在进行离散化之前需要对一些异常值进行处理
'''
##########################
max_deal_price=dgp.Balance_Convert(data[['max_deal_price']])
##########################
avg_deal_price=dgp.Balance_Convert(data[['avg_deal_price']])
##########################
pricorder=['fguarantee_last_year_deal_orders',
          'last_6m_deal_orders','last_6m_deal_amount','last_year_deal_orders',
           'last_year_deal_amount']
from sklearn.preprocessing import RobustScaler
rscaler=RobustScaler()
pricorder_temp=rscaler.fit_transform(data[pricorder])
pricorder_new= pd.DataFrame(pricorder_temp, index=data.index, columns=pricorder)
pricorder_new=pd.concat([pricorder_new,max_deal_price,avg_deal_price],axis=1)

print '##    对prefer数据处理    ####'

# dgp.Plot_data_df(data[prefer],label,prefer,dis=False)
# print data[prefer].describe()
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
#########################
'''
这种数据分布不适合使用等距离散化
'''
htl_advan_date=np.array(data[['htl_advanced_date']]).ravel()
def juge(x):
    if x>30:
        return 31
    else:
        return x
htl_advan_date2=map(juge,htl_advan_date)
htl_advanced_date=DataFrame(htl_advan_date2,index=data.index,columns=['htl_advanced_date'])
prefer=['htl_consuming_capacity','avg_htl_star','ratio_business_htl','htl_star_prefer',
        'flt_price_sitivity','flt_generous_index']
prefer_new=pd.concat([data[prefer],htl_advanced_date],axis=1)

print '##    对other数据处理   #####'

other=['profile_advanced_date','ctrip_cash_amount',
       'credit_card_succ_rate','phone_type','base_active']
#######################
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
profile_advanced_date=dgp.Discretedata(profile_advanced_date)#对已经进行异常值处理的进行离散化
#######################
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
ctrip_cash_amount=dgp.Discretedata(ctrip_cash_amount)
#######################
other=['credit_card_succ_rate','phone_type','base_active']
other_new=pd.concat([data[other],ctrip_cash_amount,profile_advanced_date],axis=1)

print '###拼接全部数据###'
data_new=pd.concat([data[noshow],people_new,pricorder_new,prefer_new,other_new],axis=1)
# dgp.Importance_Plot(data_new,label)

# from Learning_cure_Plot import learn_curve_plot
# from sklearn.ensemble import RandomForestClassifier
#
# dgp.Importance_Plot(data,label)
# dgp.Importance_Plot(data_new,label)







