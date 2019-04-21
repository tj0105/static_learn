#Author:LDY
# -*- coding: utf-8 -*-

from __future__ import division
import math
import pandas

from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
##knn 欧氏距离
def euclideanKnn(x_train,x_test,y_train,y_test):
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(x_train, y_train)
    print(knn.score(x_test, y_test))
##5折交叉验证寻找最佳的k值:
def fiveFolder(x,y):
    k_rane=[i+1 for i in range (20)]
    k_ranges=[str(i) for i in k_rane]
    print(k_rane)
    scores=[]
    for k in k_rane:
        knn=KNeighborsClassifier(n_neighbors=k)
        score=cross_val_score(knn,x,y,cv=5,scoring='accuracy')
        scores.append(score.mean())
    print(scores)
    print(max(scores))
    print(scores.index(max(scores)))
    pltScores(k_rane,k_ranges,scores)
##画图
def pltScores(k_range,k_ranges,scores):
    plt.style.use('seaborn-darkgrid')
    fig=plt.figure()
    scoreImg=fig.add_axes([0.11,0.1,0.8,0.8])
    plt.ylim(0.9,1)
    plt.xticks(k_range,k_ranges)
    scoreImg.plot(k_range,scores,color='b',marker='',lw=1)
    scoreImg.scatter(k_range,scores,color='r',s=50)
    plt.show()

##其他距离的knn算法
def otherDistanceKNN(x_train,x_test,y_train,y_test):
    knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
    knn.fit(x_train, y_train)
    print(knn.score(x_test, y_test))
    knn = KNeighborsClassifier(n_neighbors=3, metric='chebyshev')
    knn.fit(x_train, y_train)
    print(knn.score(x_test, y_test))

##使用BAGGING方法寻找最佳k值
def baggingKNN(x,y):
    k_rane = [i+1 for i in range(20)]
    k_ranges = [str(i) for i in k_rane]
    scores = []
    for k in k_rane:
        bagging=BaggingClassifier(KNeighborsClassifier(n_neighbors=k),
                                  max_samples=0.9,n_estimators=11,bootstrap=True)
        bagging.fit(x,y)
        scores.append(bagging.score(x,y))
    print(scores)
    print(max(scores))
    print(scores.index(max(scores)))
    pltScores(k_rane,k_ranges,scores)

if __name__=='__main__':
    url="wine.data"
    names=['Label','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13']
    dateset=pandas.read_csv(url,names=names)
    origindata=dateset.iloc[range(0,178),range(1,14)]
    normaldata=StandardScaler().fit_transform(origindata)
    # print(origindata.describe())
    x=normaldata
    y=dateset.iloc[range(0,178),range(0,1)].values.reshape(1,178)[0]
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4)

    '''
     test 欧氏距离
    '''
    # euclideanKnn( x_train,x_test,y_train,y_test)
    '''
    test 5折交叉验证
    '''
    # fiveFolder(x,y)
    '''
    test 降维
    '''
    # pca=PCA(n_components=6)
    # x_pca=pca.fit_transform(x)
    # fiveFolder(x_pca,y)

    '''
    test 其他距离
    '''
    # otherDistanceKNN(x_train,x_test,y_train,y_test)

    '''
    使用LDA降维
    '''
    # lda=LinearDiscriminantAnalysis();
    # x_lda=lda.fit_transform(x,y)
    # fiveFolder(x_lda, y)

    '''
    使用BAGGING方法寻找最佳k值
    '''
    baggingKNN(x,y)

