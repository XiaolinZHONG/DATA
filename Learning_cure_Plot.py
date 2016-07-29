#coding=utf-8
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
import numpy as np

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


if __name__ == '__main__':
    pass
