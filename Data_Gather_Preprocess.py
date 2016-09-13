#%matplotlib inline
#coding=utf-8

class Data_Gather_Preprocess(object):
    import time,timeit
    def __init__(self):
        pass

    def Read_Data(self,adress,label_crd,data_crd=0,stract=None,sampletrick=True):
        '''
        :param adress: string style 'string' if your system is linux pay attention :../
        :param label_crd: int
        :param data_crd:  int
        :param stract:  list style with string ['string']
        :param sampletrick: if you want to drop the nan the trigger is True
        :return: the data read from the csv file
        '''
        import pandas as pd
        from pandas import DataFrame
        import numpy as np
        import matplotlib.pylab as plt
        df_0=pd.read_csv(adress,sep=',')
        #which means that the separate by ','
        if sampletrick:
            df_1=df_0.dropna(how='any')
        else:
            df_1=df_0
        print 'Reading data ....'
        #get the data and the target/label
        data_1=np.array(df_1.ix[:,data_crd:])
        label=np.array(df_1.ix[:,label_crd])
        print 'Samples_data_shape:',data_1.shape
        print '#'*50
        if stract!=None:
            df_2=df_1[stract] #use to stract the data you want
        else:
            df_2=df_1.ix[:,data_crd:]# just used to transfer the data
        #print 'The data you want to stract:','\n',df_2.head()
        #print '#'*50
        #print 'The data describe:','\n',df_2.describe()
        return df_2,label



    def Plot_data_df(self,df_2,label,strac,pair=True):
        ''''''
        import matplotlib.pylab as plt
        import math
        import seaborn as sns
        import pandas as pd

        x = range(df_2.shape[0])
        maxn = df_2.shape[1]
        fig = plt.figure(1,figsize=(10,maxn+10))
        sns.set(style='darkgrid',color_codes=True)
        style = 'white'
        for i in xrange(maxn):
            y=df_2.ix[:,i]
            plot_context = fig.add_subplot(maxn, 2, 2*i+1)
            plot_context.scatter(x, y, c=label,s=10,alpha=0.7)

            plt.title(strac[i])
            fig.add_subplot(maxn, 2, 2*i + 2)
            sns.distplot(y,axlabel=False)
            fig.tight_layout()
            plt.title(strac[i])

        df_label=pd.DataFrame({'label':list(label)})
        df_2=pd.concat([df_2,df_label],axis=1,join='inner')
        if pair==True:
            sns.pairplot(df_2,vars=strac,hue='label',)
        plt.show()

    def Plot_data_np(self, df_2, label=None, strac=None):
        ''''''
        import matplotlib.pylab as plt
        import math
        import seaborn as sns
        x = range(len(df_2))
        maxn =df_2.shape[1]
        fig = plt.figure(1, figsize=(maxn * 3, maxn * 6))
        sns.set(style='darkgrid')
        style = 'white'
        for i in xrange(maxn):
            y=df_2[:,i]
            plot_context = fig.add_subplot(maxn, 2, 2 * i + 1)
            if label == None:
                plot_context.scatter(x, y, s=10, alpha=0.7)
            else:
                plot_context.scatter(x, y, c=label, s=10, alpha=0.7)
            if strac:
                plt.title(strac[i])
            else:
                pass
            fig.add_subplot(maxn, 2, 2 * i + 2)
            sns.distplot(y, axlabel=False)
            if strac:
                plt.title(strac[i])
            else:
                pass
        plt.show()

    def Scaler_data(self,df_2,feature_range=(0,1),label=None,strac=True):
        ''''''
        from sklearn import preprocessing
        import pandas as pd
        min_max_scaler=preprocessing.MinMaxScaler(feature_range=feature_range)
        if label==None:
            df_2_scaler=min_max_scaler.fit_transform(df_2)
        else:
            df_2_scaler = min_max_scaler.fit_transform(df_2,label)


        df_2_scaler= pd.DataFrame(df_2_scaler, index=df_2.index, columns=strac)
        return df_2_scaler,min_max_scaler




    def Discretedata(self,data):
        '''
        This algorithm use the df describe function to separate
        the data by min 25% 50% 75% max. obviously, this program
        is not fit for the 0-1 data
        :param data: DATA FRAME style
        :return: new discreted data
        '''
        import pandas as pd
        from pandas import DataFrame
        import numpy as np
        print 'Discretizing Data......'
        for i in xrange(data.shape[1]):
            x = data.ix[:, i].values
            maxval=data.ix[:,i].max()
            minval=data.ix[:,i].min()
            step=(maxval-minval)/10
            c9=round(minval+9*step)
            c8=round(minval+8*step)
            c7=round(minval+7*step)
            c6=round(minval+6*step)
            c5=round(minval+5*step)
            c4=round(minval+4*step)
            c3=round(minval+3*step)
            c2=round(minval+2* step)
            c1=round(minval+step)
            for j, value in enumerate(x):
                if value > c9:
                    x[j] = 9
                elif value >= c8:
                    x[j] = 8
                elif value >= c7:
                    x[j] = 7
                elif value >= c6:
                    x[j] = 6
                elif value >= c5:
                    x[j] = 5
                elif value >= c4:
                    x[j] = 4
                elif value >= c3:
                    x[j] = 3
                elif value >= c2:
                    x[j] = 2
                elif value >= c1:
                    x[j] = 1
                else:
                    x[j] = 0
        print 'Discretizing Done !'
        return data


    def PCA_Process(self,df_2,label=None,n_comp=None):
        ''''''
        from sklearn import decomposition
        if n_comp==None:
            n_comp='mle'
        else:
            pass
        pca=decomposition.PCA(n_components=n_comp)
        df_2_pca_new=pca.fit_transform(df_2,y=label)
        print '#' * 50
        print 'PCA score:', '\n', pca.explained_variance_ratio_.cumsum()
        print '#' * 50
        print 'new feature:', '\n', df_2_pca_new.shape,df_2_pca_new
        return df_2_pca_new


    def Importance_Plot(self,df_2, label):
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
        model.fit(data1,label)
        importance = model.feature_importances_
        std = np.std([importance for tree in model.estimators_], axis=0)
        indices = np.argsort(importance)[::-1]
        namedata = df_2
        # Print the feature ranking
        print("Feature ranking:")
        importa = pd.DataFrame(
            {'importance': list(importance[indices]), 'Feature name': list(namedata.columns[indices])})
        print importa
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(data1.shape[1]), importance[indices],
                color="g", yerr=std[indices], align="center")
        plt.xticks(range(data1.shape[1]), indices)
        plt.xlim([-1, data1.shape[1]])
        plt.grid(True)
        plt.show()

        modelnew=SelectFromModel(model,prefit=True)
        print 'Select feature num:',modelnew.transform(data1).shape[1]



    def Balance_Convert(self,x):
        '''
        :param x: dataframe style
        :return: converted result
        '''
        def convert(x):
            for i, value in enumerate(x.values):
                if value > 10000:
                    x.values[i] = 4
                elif value > 2000:
                    x.values[i] = 3
                elif value > 500:
                    x.values[i] = 2
                elif value > 0:
                    x.values[i] = 1
                else:
                    x.values[i] = 0
            return x

        for j in xrange(x.shape[1]):
            convert(x.ix[:, j])
        return x





if __name__ == '__main__':
    pass
