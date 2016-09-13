#coding=utf-8
#@author:xiaolin
#@file:CSM_V0.8.py
#@time:2016/9/7 14:45

import time,timeit
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score,classification_report
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from pandas import DataFrame
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import BaggingRegressor
from sklearn.externals import joblib

def Importance_Plot(df_2, label):
    '''
    :param df: DATAFRAME style
    :param label: y vector
    :param threshold: jude threshold
    :return: figure
    '''
    import numpy as np
    import matplotlib.pylab as plt
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    import pandas as pd
    model = ExtraTreesClassifier()
    data1 = np.array(df_2)
    model.fit(data1, label)
    importance = model.feature_importances_
    std = np.std([importance for tree in model.estimators_], axis=0)
    indices = np.argsort(importance)[::-1]
    namedata = df_2
    # Print the feature ranking
    print("Feature ranking:")
    importa = pd.DataFrame(
        {'importance': list(importance[indices]), 'Feature name': list(namedata.columns[indices])})
    print (importa)
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(data1.shape[1]), importance[indices],
            color="g", yerr=std[indices], align="center")
    plt.xticks(range(data1.shape[1]), indices)
    plt.xlim([-1, data1.shape[1]])
    plt.grid(True)
    plt.show()

    modelnew = SelectFromModel(model, prefit=True)
    print ('Select feature num:', modelnew.transform(data1).shape[1])

def learn_curve_plot(estimator,title,X,y,cv=None,train_sizes=np.linspace(0.1,1.0,5)):
    '''
    :param estimator: the model/algorithem you choose
    :param title: plot title
    :param x: train data numpy array style
    :param y: target data vector
    :param xlim: axes x lim
    :param ylim: axes y lim
    :param cv:
    :return: the figure
    '''
    plt.figure()

    train_sizes,train_scores,test_scores=\
        learning_curve(estimator,X,y,cv=cv,train_sizes=train_sizes)
    '''this is the key score function'''
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)

    plt.fill_between(train_sizes,train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,alpha=0.1,color='b')
    plt.fill_between(train_sizes,test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'o-',color='b',label='training score')
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label='cross valid score')
    plt.xlabel('training examples')
    plt.ylabel('score')
    plt.legend(loc='best')
    plt.grid('on')
    plt.title(title)
    plt.show()

def corr_analysis(data):
    corr=data.corr()
    #Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(20, 20))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(110, 10,as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,linewidths=.5,
            cbar_kws={"shrink": .6},annot=True,annot_kws={"size":8} )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

def Desc_Scaler(data, des=[0.95], feature_range=(0, 1),detail=False):
    from sklearn import preprocessing
    import pandas as pd
    from pandas import DataFrame

    describe=data.describe(des)
    if detail==True:
        print (describe)
    y=describe.shape[1]
    x=[]
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
    for i in range(y):
        threshold=describe.ix[-2,i]
#         print ('Threshold:',threshold)
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

def Report(label_tst,pre):
    count_tn,count_fp,count_fn,count_tp=0,0,0,0
    for i in range(len(label_tst)):
        if label_tst[i]==0:
            if pre[i]<0.5:
                count_tn+=1
            else:
                count_fp+=1
        else:
            if pre[i]<0.5:
                count_fn+=1
            else:
                count_tp+=1
    print ('Total:', len(label_tst))
    print 'FP被分为好人的坏人:', count_fp, 'TN正确分类的坏人:', count_tn, '坏人正确率：',round(float(count_tn)/float((count_fp+count_tn)),3)
    print 'FN被分为坏人的好人:', count_fn, 'TP正确分类的好人:', count_tp, '好人正确率：',round(float(count_tp)/float((count_fn+count_tp)),3)


adress1='D:/data/bigtable_2B.csv'

df_0   = pd.read_csv(adress1,sep=',',dtype={'uid_membertype':str,'uid_source':str})
df_1   = df_0.dropna(how='any')
drop   = ['uid_membertype','uid_source']
df_1   = df_1.drop(drop,axis=1)
data   = df_1.ix[:,1:-1]
label  = df_1['uid_flag']
print ('Samples shape:',data.shape)
print (data.head())
# sns.set(style='darkgrid',color_codes=True)
# sns.distplot(label,axlabel=False)
# plt.show()
count_1 = float(label.sum())
count_0 = float(label.count() - float(label.sum()))
print ('Good Samples:',count_1,'Bad Samples:',count_0)

relation=['com_passenger_count','com_idno_count','com_mobile_count',
          'ord_success_order_cmobile_count']
interaction=['voi_complaint_count','voi_complrefund_count','voi_comment_count','acc_loginday_count',
             'pro_validpoints','pro_base_active','pro_ctrip_profits','pro_customervalue']
fanacial=['voi_complrefund_count','fai_lackbalance','bil_payord_count',
          'bil_refundord_count','bil_payord_credit_count','bil_payord_debit_count','bil_ordertype_count','bil_platform_count',
          'pro_htl_star_prefer','pro_htl_consuming_capacity','pro_phone_type','ord_success_max_order_amount',
          'ord_total_order_amount','ord_success_first_class_order_count','ord_success_flt_first_class_order_count',
          'ord_success_trn_max_order_amount','ord_success_pkg_order_count'
          ,'ord_success_htl_first_class_order_count','ord_success_htl_max_order_amount','ord_success_htl_aboard_order_amount',
          'ord_success_aboard_order_count']
cosuming=['pro_generous_stingy_tag','pro_base_active','pro_customervalue','pro_phone_type','pro_advanced_date','pro_generous_stingy_tag','pro_htl_star_prefer','pro_htl_consuming_capacity',
          'pro_phone_type','pro_validpoints','pro_ctrip_profits','pro_customervalue','pro_ismarketing',
          'ord_success_max_order_amount','ord_success_order_count','ord_success_avg_leadtime',
          'ord_total_order_amount','ord_cancel_order_count','ord_success_first_class_order_count',
          'ord_success_first_class_order_amount','ord_success_order_type_count','ord_success_aboard_order_count',
          'ord_success_aboard_order_amount','ord_success_order_acity_count','ord_success_flt_last_order_days','ord_success_flt_first_class_order_amount',
         'ord_success_flt_max_order_amount','ord_success_flt_avg_order_pricerate','ord_success_flt_aboard_order_count'
         ,'ord_success_flt_order_count','ord_success_flt_order_acity_count',
         'ord_cancel_flt_order_count','ord_success_htl_last_order_days',
         'ord_success_htl_first_class_order_amount','ord_success_htl_max_order_amount','ord_success_htl_order_refund_ratio',
         'ord_success_htl_aboard_order_count','ord_success_htl_guarantee_order_count',
         'ord_success_htl_noshow_order_count','ord_success_htl_order_count','ord_cancel_htl_order_count'
         ,'ord_success_pkg_last_order_days','ord_success_pkg_max_order_amount',
         'ord_success_trn_last_order_days','ord_success_trn_max_order_amount','ord_success_trn_order_count',
         'ord_success_trn_order_amount']
people=['uid_grade','uid_dealorders','uid_emailvalid','uid_age','uid_mobilevalid',
        'uid_addressvalid','uid_isindentify','uid_authenticated_days','uid_signupdays',
        'uid_signmonths','uid_lastlogindays','uid_samemobile','ord_success_order_cmobile_count',
        'com_mobile_count','com_has_child','pro_generous_stingy_tag','pro_base_active',
        'pro_customervalue','pro_phone_type']

data_new=Desc_Scaler(data,[0.75,0.96])

label_xc=np.array(label).ravel()
data_trn,data_tst,label_trn,label_tst=train_test_split(data_new,label_xc,test_size=0.2)

# #Using random forest regression to classify
# rfr_xc=RandomForestRegressor(random_state=500)
# rfr_xc.fit(data_trn,label_trn)
# rfr_pre_xc=rfr_xc.predict(data_tst)
# print ('Using random forest regression')
# print ('AUC-ROC:',roc_auc_score(label_tst,rfr_pre_xc))
# answer=rfr_pre_xc>0.5
# print (classification_report(label_tst,answer,target_names=['neg','pos']))
# Report(label_tst,rfr_pre_xc)
# joblib.dump(rfr_xc, "D:/project_csm/model/rfr_xc_train_model.m")
# #Using GBDT to regression
# gbtr_all=GradientBoostingRegressor(n_estimators=250,random_state=50)
# gbtr_all.fit(data_trn,label_trn)
# gbtr_pre_all=gbtr_all.predict(data_tst)
# print ('Using GBT regression')
# print ('AUC-ROC:',roc_auc_score(label_tst,gbtr_pre_all))
# answer=gbtr_pre_all>0.5
# print (classification_report(label_tst,answer,target_names=['neg','pos']))
# Report(label_tst,gbtr_pre_all)
# joblib.dump(gbtr_all, "D:/project_csm/model/gbtr_xc_train_model.m")
#
# tst_pre=rfr_pre_xc*0.2+gbtr_pre_all*0.8
# print ('AUC-ROC:',roc_auc_score(label_tst,tst_pre))
# answer=tst_pre>0.5
# print (classification_report(label_tst,answer,target_names=['neg','pos']))
# Report(label_tst,tst_pre)
#
# Importance_Plot(data_new,label)

def Predict_test_11(data_tst):
    rfr        = joblib.load("D:/project_csm/model/rfr_xc_train_model.m")
    rfr_pre_xc = rfr.predict(data_tst)
    gbtr        = joblib.load("D:/project_csm/model/gbtr_xc_train_model.m")
    gbtr_pre_all = gbtr.predict(data_tst)
    tst_pre = rfr_pre_xc * 0.2 + gbtr_pre_all * 0.8
    print tst_pre
    return tst_pre

Predict_test_11(data_tst)

# rfr_5_xc=RandomForestRegressor(random_state=500)
# rfr_5_xc.fit(data_trn[relation],label_trn)
# rfr_5_pre_xc=rfr_5_xc.predict(data_tst[relation])
# joblib.dump(rfr_5_xc, "D:/project_csm/model/rfr_5_xc_train_model.m")
# rfr_4_xc=RandomForestRegressor(random_state=500)
# rfr_4_xc.fit(data_trn[cosuming],label_trn)
# rfr_4_pre_xc=rfr_4_xc.predict(data_tst[cosuming])
# joblib.dump(rfr_4_xc, "D:/project_csm/model/rfr_4_xc_train_model.m")
# rfr_3_xc=RandomForestRegressor(random_state=500)
# rfr_3_xc.fit(data_trn[people],label_trn)
# rfr_3_pre_xc=rfr_3_xc.predict(data_tst[people])
# joblib.dump(rfr_3_xc, "D:/project_csm/model/rfr_3_xc_train_model.m")
# rfr_2_xc=RandomForestRegressor(random_state=500)
# rfr_2_xc.fit(data_trn[fanacial],label_trn)
# rfr_2_pre_xc=rfr_2_xc.predict(data_tst[fanacial])
# joblib.dump(rfr_2_xc, "D:/project_csm/model/rfr_2_xc_train_model.m")
# rfr_1_xc=RandomForestRegressor(random_state=500)
# rfr_1_xc.fit(data_trn[interaction],label_trn)
# rfr_1_pre_xc=rfr_1_xc.predict(data_tst[interaction])
# joblib.dump(rfr_1_xc, "D:/project_csm/model/rfr_1_xc_train_model.m")

def Predict_test_51(data_tst):
    rfr_5_xc        = joblib.load("D:/project_csm/model/rfr_5_xc_train_model.m")
    rfr_5_pre_xc = rfr_5_xc.predict(data_tst[relation])
    rfr_4_xc        = joblib.load("D:/project_csm/model/rfr_4_xc_train_model.m")
    rfr_4_pre_xc = rfr_4_xc.predict(data_tst[cosuming])
    rfr_3_xc        = joblib.load("D:/project_csm/model/rfr_3_xc_train_model.m")
    rfr_3_pre_xc = rfr_3_xc.predict(data_tst[people])
    rfr_2_xc        = joblib.load("D:/project_csm/model/rfr_2_xc_train_model.m")
    rfr_2_pre_xc = rfr_2_xc.predict(data_tst[fanacial])
    rfr_1_xc        = joblib.load("D:/project_csm/model/rfr_1_xc_train_model.m")
    rfr_1_pre_xc = rfr_1_xc.predict(data_tst[interaction])
    gbtr        = joblib.load("D:/project_csm/model/gbtr_xc_train_model.m")
    gbtr_pre_all = gbtr.predict(data_tst)
    rfr_xc = rfr_5_pre_xc * 0.1 + rfr_4_pre_xc * 0.1 + rfr_3_pre_xc * 0.25 + rfr_2_pre_xc * 0.3 + rfr_1_pre_xc * 0.25
    tst_pre = rfr_xc * 0.2 + gbtr_pre_all * 0.8
    print tst_pre
    return tst_pre

Predict_test_51(data_tst)