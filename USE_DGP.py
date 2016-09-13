#coding=utf-8
#@author:xiaolin
#@file:USE_DGP.py
#@time:2016/8/4 15:57



from Data_Gather_Preprocess import *
import time, timeit
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
dgp=Data_Gather_Preprocess()
adress='D:\\data\\newmodel.csv'
df_2,label=dgp.Read_Data(adress,0,1)

###Basic information
# mobile=['samemobile','common_mobile_cnt','contact_mobile_count']
# sign=['signmonths','is_quick_uid']
# people=['grade','age','common_has_child']
# valid=['emailvalid','isindentify','addressvalid']
#
# person_info=['common_mobile_cnt','contact_mobile_count',
#              'signmonths','age'
#              ]

#####mobile process

# dgp.Plot_data_df(df_2[mobile],label,mobile)
# # dgp.Importance_Plot(df_2[mobile],label)
# common_mobile_cnt_new=dgp.Discretedata(df_2[['common_mobile_cnt','contact_mobile_count']])
# df_2_new=pd.concat([df_2[['samemobile']],common_mobile_cnt_new],axis=1)
# # print df_2_new.head()
# dgp.Plot_data_df(df_2_new,label,mobile)
# # dgp.Importance_Plot(df_2_new,label)
# #df_2_new=dgp.Scaler_data(df_2_new)
# data_pca=dgp.PCA_Process(df_2_new,label,1)
#
# ######sign process
# dgp.Plot_data_df(df_2[sign],label,sign)
# # dgp.Importance_Plot(df_2[sign],label)
# sign_new=dgp.Scaler_data(df_2[['signmonths']],strac=['signmonths'])
# #对于注册时间长度的数据，我们缩放到0到1区间
# # dgp.Plot_data_np(sign_new,label,['signmonths'])
#
# df_2_new=pd.concat([df_2_new,sign_new,df_2['is_quick_uid']],axis=1)


#########people process
# dgp.Plot_data_df(df_2[people],label,people)
# # dgp.Importance_Plot(df_2[people],label)
# age_new=dgp.Scaler_data(df_2['age'],strac=['age'])
# df_2_new=pd.concat([df_2_new,age_new],axis=1)
#
#
#
# ##########valid process
# # dgp.Plot_data_df(df_2[valid],label,valid)
# # dgp.Importance_Plot(df_2[valid],label)
# df_2_new=pd.concat([df_2_new,df_2[valid]],axis=1)
# print df_2_new.head()
#
#
# #####################
# strac=df_2_new.columns
# # print strac
# df_2_new2=dgp.Scaler_data(df_2[person_info],strac=person_info)
# df_2_new=dgp.Scaler_data(df_2_new,strac=strac)
# print df_2_new.head()
# dgp.Importance_Plot(df_2_new,label)
# dgp.Importance_Plot(df_2_new2,label)
#
# from sklearn.cross_validation import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np
#
# data_trn,data_tst,target_trn,target_tst=\
#     train_test_split(df_2_new,label,test_size=0.8)
# clf_rf=RandomForestClassifier()
# rf=clf_rf.fit(data_trn,target_trn)
# prs=clf_rf.score(data_trn,target_trn)
#
# print 'Train score:',prs
# prd=clf_rf.predict(data_tst)
# print 'Decision result:',prd,'Precesion:',np.mean(prd==target_tst)
#
# data_trn,data_tst,target_trn,target_tst=\
#     train_test_split(df_2_new2,label,test_size=0.8)
# clf_rf=RandomForestClassifier()
# rf=clf_rf.fit(data_trn,target_trn)
# prs=clf_rf.score(data_trn,target_trn)
#
# print 'Train score:',prs
# prd=clf_rf.predict(data_tst)
# print 'Decision result:',prd,'Precesion:',np.mean(prd==target_tst)














###Bank and card information
# balance=['balance_tmoney',  'balance_wallet',  'balance_refund']
# pay=['pay_count','mth12_credit_pay_count','mth12_debit_pay_count',
#      'mth12_self_pay_amount','mth12_pay_count','mth12_pay_amount']
# bank=['bankclass','cardclass','credit_card_count','bank_card_count',
#       'self_card_count','debit_card_count']
#
#


# dgp.Plot_data_df(df_2[pay],label,pay)
# dgp.Plot_data_df(df_2[bank],label,bank)
# dgp.Plot_data_df(df_2[balance],label,balance)









###Login info
# login=['haspaypwd','month12_loginday_count','month12_modify_pwd_count']

# dgp.Plot_data_df(df_2[login],label,login)














###Order info
# order_success= ['success_order_count', 'success_order_amount',
#          'last_success_order_amount', 'success_order_quantity',
#          'month12_success_order_amount', 'month12_success_order_count']
# order_total=['total_order_count','total_order_quantity','total_order_amount']
# order_type=['order_type_count','month12_order_type_count']
# order_self=['self_order_amount','month6_self_order_amount',
#             'month12_self_order_amount','self_order_count',
#             'month12_self_order_count']
# order_withdrow=['mth12_withdrow_count','mth12_withdrow_amount']
# order_aboard=['aboard_order_amount','aboard_order_count',
#               'month12_aboard_order_count','month12_aboard_order_amount']
# order_bill=['month12_bill_payord_count','month12_bill_paysord_count',
#       'month12_bill_pays_ratio','month12_bill_cardpay_count',
#       'month12_bill_thridpay_count','month12_bill_ordertype_count',
#       'month12_bill_platform_count']
# order_orders=['dealorders','month12_order_amount']
# order_lead=['month12_avg_leadtime']

# dgp.Plot_data_df(df_2[order_bill],label,order_bill)
# dgp.Plot_data_df(df_2[order_self],label,order_self)
# dgp.Plot_data_df(df_2[order_aboard],label,order_aboard)
# dgp.Plot_data_df(df_2[order_type],label,order_type)
# dgp.Plot_data_df(df_2[order_success],label,order_success)
# dgp.Plot_data_df(df_2[order_lead],label,order_lead)
# dgp.Plot_data_df(df_2[order_withdrow],label,order_withdrow)











###Other info
# interactive=['month12_comment_count','month12_complaint_count',
#              'month12_complrefund_count']


# dgp.Plot_data_df(df_2[interactive],label,interactive)
















#dgp.Plot_data_df(df_2[order_aboard],label,order_aboard)
# print df_2[order_aboard].head()
# df_2_order=dgp.Discretedata(df_2[order_aboard])
# print df_2_order.head()
# dgp.Plot_data_df(df_2_order,label,order_aboard)
#balance_new=dgp.Balance_Convert(df_2[balance])
#print balance_new.head()
#print pd.concat([balance_new,df_2],axis=1,join='inner').head()
#dgp.PCA_Process(df_3_balance,label,1)

def LR(data,label,scor,title):
    from sklearn.cross_validation import train_test_split
    import numpy as np

    data_trn,data_tst,target_trn,target_tst=\
        train_test_split(data,label,test_size=0.8)

    # clf_rf=RandomForestClassifier(class_weight='balanced')
    # rf=clf_rf.fit(data_trn,target_trn)
    # prs=clf_rf.score(data_trn,target_trn)
    #
    # print 'Train score:',prs
    # prd=clf_rf.predict(data_tst)
    # print 'Decision result:',prd
    # print 'Precesion:',np.mean(prd==target_tst)

    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression(C=5,class_weight='auto')
    lr.fit(data_trn,target_trn)
    proba=lr.predict_proba(data_tst)
    print '准确度：',np.mean(lr.predict(data_tst)==target_tst),lr.score(data_tst,target_tst)
    from Learning_cure_Plot import *
    learn_curve_plot(lr,title,data_trn,target_trn,train_sizes=np.linspace(0.1,1,10))
    # print label
    # print lr.coef_.shape
    # print lr.intercept_.shape
    # print proba

    a=float(scor/2)
    b=float((a*0.2)/0.693)
    coef_new=np.matrix(lr.coef_*b)
    inttercet_new=np.matrix(a+lr.intercept_*b)
    result1=np.dot(data_tst,(coef_new.T))+inttercet_new
    # print result
    result2=scor*proba
    result2=result2[:,0]

    sumb1,sumg1,sumb2,sumg2=0,0,0,0
    bad=[]
    good=[]
    bad_guy=[]
    good_guy=[]
    for i in range(len(target_tst)):
        if result1[i]>=float(scor/2):
            if target_tst[i]==0:
                sumg1+=1
                good.append(result1[i])
            else:
                sumb1+=1
        else:
            if target_tst[i] == 1:
                sumg2 += 1
                bad.append(result1[i])
            else:
                sumb2+= 1

    print 'Total:',len(proba),title
    print 'FP被分为好人的坏人:',sumb1,'TN正确分类的坏人:',sumg2
    print 'FN被分为坏人的好人:',sumb2,'TP正确分类的好人:',sumg1

    sumb1,sumg1,sumb2,sumg2=0,0,0,0
    bad=[]
    good=[]
    bad_guy=[]
    good_guy=[]
    for i in range(len(target_tst)):
        if result2[i]>=float(scor/2):
            if target_tst[i]==0:
                sumg1+=1
            else:
                sumb1+=1
        else:
            if target_tst[i] == 1:
                sumg2 += 1
            else:
                sumb2+= 1

    print 'Total:',len(proba)
    print 'FP被分为好人的坏人:',sumb1,'TN正确分类的坏人:',sumg2
    print 'FN被分为坏人的好人:',sumb2,'TP正确分类的好人:',sumg1

    import matplotlib.pylab as plt
    import seaborn as sns
    sns.distplot(result1,label='all_scal')
    sns.distplot(result2, label='all_proba')
    # sns.distplot(bad_guy,label='bad')
    # sns.distplot(bad,label='error')
    # sns.distplot(good,label='bad correct')
    # sns.distplot(good_guy,label='good')
    plt.title(title)
    plt.legend()
    plt.show()
    return result1,result2,good,bad
def RFR(data,label,scor,title):
    from sklearn.cross_validation import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np

    data_trn,data_tst,target_trn,target_tst=\
        train_test_split(data,label,test_size=0.8)

    rfr=RandomForestRegressor(max_features='sqrt',min_samples_leaf=10,n_jobs=-1,oob_score=True,random_state=10)
    rfr.fit(data_trn,target_trn)
    proba=rfr.predict(data_tst)

    score=[]
    for i ,value in enumerate(proba):
        score.append(scor*(1-value))

    sumb,sumg,sumb2,sumg2=0,0,0,0
    bad=[]
    good=[]
    bad_guy=[]
    good_guy=[]
    for i in range(len(target_tst)):
        if target_tst[i]==1:
            if score[i]>float(scor/2):
                bad.append(score[i])
                sumb+=1
            else:
                good.append(score[i])
                sumg+=1
            bad_guy.append(score[i])
        else:
            if score[i]>float(scor/2):
                sumg2+=1
            else:
                sumb2+=1
            good_guy.append(score[i])


    print title,'Total:',len(proba)
    print 'FP被分为好人的坏人:',sumb,'TN正确分类的坏人:',sumg
    print 'FN被分为坏人的好人:',sumb2,'TP正确分类的好人:',sumg2
    import matplotlib.pylab as plt
    import seaborn as sns
    # sns.distplot(score,label='all')
    sns.distplot(bad_guy,label='bad')
    # sns.distplot(bad,label='error')
    # sns.distplot(good,label='bad correct')
    sns.distplot(good_guy,label='good')
    plt.title(title)
    plt.legend()
    plt.show()
    return score



###############################################################################################
order = ['success_order_count',
         'last_success_order_amount', 'success_order_quantity',
         'month12_success_order_amount', 'month12_success_order_count', 'total_order_count',
         'total_order_quantity',
         'self_order_amount', 'month6_self_order_amount',
         'self_order_count', 'mth12_withdrow_count',
         'month12_bill_payord_count',
         'month12_bill_pays_ratio', 'month12_bill_cardpay_count',
         'month12_bill_thridpay_count',  'month12_avg_leadtime']

data_order, scaler_order = dgp.Scaler_data(df_2[order], strac=order)
print 'ORDER',data_order.shape
orscore,orscore2,orgood,orbad=LR(data_order,label,175,title='ORDER')
# orscore=RFR(data_order,label,175,title='ORDER')
##############################################################################################
interactive=['month12_complaint_count',
             'month12_complrefund_count']
data_inter,scaler_inter=dgp.Scaler_data(df_2[interactive],strac=interactive)
print  'INTER'
inscore,inscore2,ingood,inbad=LR(data_inter,label,25,title='INTER')
# inscore=RFR(data_inter,label,50,title='INTER')
###############################################################################################
money =['balance_tmoney',  'balance_wallet',  'balance_refund',
         'pay_count','mth12_credit_pay_count','mth12_debit_pay_count',
     'mth12_self_pay_amount','mth12_pay_count','mth12_pay_amount',
     'bankclass','cardclass','credit_card_count','bank_card_count',
      'self_card_count','debit_card_count']
data_money,scaler_money=dgp.Scaler_data(df_2[money],strac=money)
print 'MONEY'
moscore,moscore2,mogood,mobad=LR(data_money,label,175,title='MONEY')
# moscore=RFR(data_money,label,175,title='MONEY')
#############################################################################################
people=['samemobile','common_mobile_cnt','contact_mobile_count','signmonths',
        'is_quick_uid','grade','age','common_has_child','emailvalid','isindentify',
        'addressvalid','common_mobile_cnt','contact_mobile_count','signmonths','age']
data_people,scaler_people=dgp.Scaler_data(df_2[people],strac=people)
print 'PEOPLE'
pescore,pescore2,pegood,pebad=LR(data_people,label,75,title='PEOPLE')
# pescore=RFR(data_people,label,75,title='PEOPLE')
##############################################################################################
login=['lastloginfromnow']
data_login,scaler_login=dgp.Scaler_data(df_2[login],strac=login)
print 'LOGIN'
loscore,loscore2,logood,lobad=LR(data_login,label,50,'LOGIN')
# loscore=RFR(data_login,label,50,'LOGIN')

result=[]

score=[x1+x2+x3+x4+x5 for x1,x2,x3,x4,x5 in zip(orscore,inscore,moscore,pescore,loscore)]
score2=[x1+x2+x3+x4+x5 for x1,x2,x3,x4,x5 in zip(orscore2,inscore2,moscore2,pescore2,loscore2)]
good=[x1+x2+x3+x4+x5 for x1,x2,x3,x4,x5 in zip(orgood,ingood,mogood,pegood,logood)]
bad=[x1+x2+x3+x4+x5 for x1,x2,x3,x4,x5 in zip(orbad,inbad,mobad,pebad,lobad)]
# sns.distplot(score, label='all')
# sns.distplot(score2, label='all2')
sns.distplot(good, label='good')
sns.distplot(bad, label='bad')
plt.legend()
plt.show()
