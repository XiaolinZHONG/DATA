#coding=utf-8

def Importance_Plot(data,label=None,threshold=0):
    '''
    :param data: DATAFRAME style
    :param label: y vector
    :param threshold: jude threshold
    :return: figure
    '''
    import numpy as np
    import matplotlib.pylab as plt
    from sklearn.ensemble import ExtraTreesClassifier
    model=ExtraTreesClassifier()
    data1=np.array(data)
    model.fit(data1,label)
    importance=model.feature_importances_
    std = np.std([importance for tree in model.estimators_],axis=0)
    indices = np.argsort(importance)[::-1]
    namedata=data
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(data1.shape[1]):
        if importance[indices[f]]>threshold:
            print("%d." % (f + 1)), \
                namedata.columns[indices[f]],'\n','importance:',importance[indices[f]]

    # Plot the feature importances of the forest
    plt.figure(figsize=(20, 8))
    plt.title("Feature importances")
    plt.bar(range(data1.shape[1]), importance[indices],
            color="g", yerr=std[indices], align="center")
    plt.xticks(range(data1.shape[1]), indices)
    plt.xlim([-1, data1.shape[1]])
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    pass
