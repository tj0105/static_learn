
from __future__ import division
import math

from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
def loadDataSet(filename):
    datamat=[]
    fr=open(filename)
    for line in fr.readline():
        curline=line.strip().split(',')
        print(curline)
        fltline=map(float,curline)
        datamat.append(fltline)
    return np.array(datamat)
## this is 线性回归
from sklearn.linear_model import LinearRegression
def lineReg(xtrain,ctrain,k,xtest,ytest,ptest):
    ploy = PolynomialFeatures(k)
    xtrain = ploy.fit_transform(xtrain)
    xtest = ploy.fit_transform(xtest)
    model = LinearRegression()
    model.fit(xtrain,ctrain)
    # print (model.coef_)
    predictions = model.predict(xtest)
    # print (predictions.shape)
    # k = 0
    # for i, prediction in enumerate(predictions):
    #     if prediction > 0.5:
    #         prediction = 1
    #         if prediction == ctrain[i]:
    #             k += 1
    #     else:
    #         prediction = 0
    #         if prediction == ctrain[i]:
    #             k += 1
    # return k
    errorRate = 0
    for i,prediction in enumerate(predictions):
        if prediction > 0.5:
            prediction = 1
            errorRate += ptest[i]*(1 - ytest[i])
        else:
            prediction = 0
            errorRate += ptest[i]*(ytest[i])
    return errorRate

## this is ridge regression
from sklearn.linear_model import Ridge
def ridgeReg(xtrain,ctrain,k,xtest,ytest,ptest,alpha):
    ploy = PolynomialFeatures(k)
    xtrain = ploy.fit_transform(xtrain)
    xtest = ploy.fit_transform(xtest)
    model = Ridge(alpha = alpha)
    model.fit(xtrain,ctrain)
   # print model.get_params(deep = True)
   # print model.coef_
   # print model.coef_.shape
    #print xtest.shape
    predictions = model.predict(xtest)
    errorRate = 0
    for i,prediction in enumerate(predictions):
        if prediction > 0.5:
            prediction = 1
            errorRate += ptest[i]*(1 - ytest[i])
        else:
            prediction = 0
            errorRate += ptest[i]*( ytest[i])
    #print errorRate
    return errorRate

## this is logisticReg
from sklearn.linear_model import LogisticRegressionCV
def logisticReg(xtrain,ctrain,k,xtest,ytest,ptest):
    poly = PolynomialFeatures(k)
    xtrain = poly.fit_transform(xtrain)
    xtest = poly.fit_transform(xtest)
    model = LogisticRegressionCV(cv=5,solver='liblinear')
    model.fit(xtrain,ctrain)
    predictions = model.predict(xtest)
    errorRate = 0
    for i,prediction in enumerate(predictions):
        if prediction > 0.5:
            prediction = 1
            errorRate += ptest[i]*(1 - ytest[i])
        else:
            prediction = 0
            errorRate += ptest[i]*(ytest[i])

    #print errorRate
    return errorRate

## this is lasso regression
from sklearn.linear_model import Lasso
def lossoReg(xtrain,ctrain,k,xtest,ytest,ptest,alpha):
    poly = PolynomialFeatures(k)
    xtrain = poly.fit_transform(xtrain)
    xtest = poly.fit_transform(xtest)
    model = Lasso(alpha=alpha)
    model.fit(xtrain, ctrain)
    predictions = model.predict(xtest)
    errorRate = 0
    tmp = 0
    for i, prediction in enumerate(predictions):
        if prediction > 0.5:
            tmp = ptest[i] * (1 - ytest[i])
            errorRate += tmp
        else:
            tmp = ptest[i] * (ytest[i])
            errorRate += tmp
    return errorRate

## this is LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
def LDARef(xtrain,ctrain,xtest,ytest,ptest):
    model=LDA()
    model.fit(xtrain,ctrain)
    predictions=model.predict(xtest)
    errorRate = calResults(predictions)
    return errorRate

from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
## kemas
def kmeans(xtrain,ctrain,xtest,ytest,ptest,k):
    xtrain=PolynomialFeatures(2).fit_transform(xtrain)
    xtest=PolynomialFeatures(2).fit_transform(xtest)
    n_clusters = k
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(xtrain)
    kmeans_labels = kmeans.labels_
    # fig2 = plt.figure()
    # xtrainImg=fig2.add_subplot(111)
    # xtrainImg.set_xlim(-3,5)
    # xtrainImg.set_ylim(-3,4)
    # xtrainImg.set_xlabel('x_2')
    # xtrainImg.set_xlabel('x_1')
    # colors=['b','g','r','c','m','k']
    # for i,label in enumerate(kmeans_labels):
    #     xtrainImg.scatter(x1_train[i],x2_train[i],color=colors[label],marker='o')
    # plt.show()
    mulClassifiter = OneVsRestClassifier(LinearSVC())
    mulClassifiter.fit(xtrain, kmeans_labels)
    model = mulClassifiter.predict(xtrain)
    # print(model)
    mul_trans_bin=mulclassifiterCovertBinary(model,n_clusters,ctrain)
    testPrelabel=mulClassifiter.predict(xtest)
    testBinlabel=[mul_trans_bin[2][i] for i in testPrelabel]
    # print(testPrelabel[0])
    # print(testBinlabel[0])
    errorRate = 0
    for i, prediction in enumerate(testBinlabel):
        if prediction > 0.5:
            prediction = 1
            errorRate += ptest[i] * (1 - ytest[i])
        else:
            prediction = 0
            errorRate += ptest[i] * (ytest[i])

    # print errorRate
    return errorRate
##多类分类器转换为二类分类器
def mulclassifiterCovertBinary(model,n_clusters,ctrain):
    mul_trans_bin=np.zeros((3,n_clusters))
    # print(mul_trans_bin)
    # print(model)
    # print(mul_trans_bin)
    for i ,label in enumerate(model):
            mul_trans_bin[int(ctrain[i])][label] += 1
    for i in range(n_clusters):
        if(mul_trans_bin[0][i]>=mul_trans_bin[1][i]):
            mul_trans_bin[2][i]=0
        else:mul_trans_bin[2][i]=1
    # print(mul_trans_bin)
    return mul_trans_bin
##计算错误率
def calResults(predictions):
    errorRate = 0
    for i, prediction in enumerate(predictions):
        if prediction > 0.5:
            prediction = 1
            errorRate += ptest[i] * (1 - ytest[i])
        else:
            prediction = 0
            errorRate += ptest[i] * (ytest[i])
    return errorRate
def printfResults(results):
    m = 1
    k = -1
    print(results)
    for i in range(len(results)):
        if results[i] < m:
            m = results[i]
            k = i
    print(m)
    print(k)
    plt.plot(results)
    plt.show()

if __name__=='__main__':
    xtrain = np.loadtxt('xtrain.txt',delimiter=',')
    ctrain = np.loadtxt('ctrain.txt')
    xtest = np.loadtxt('xtest.txt',delimiter=',')
    ytest =np.loadtxt('c1test.txt')
    ptest =np.loadtxt('ptest.txt')
    # # xtest=loaddatset('xtrain.txt')
    # # print(xtest)
    # print(xtrain.shape)
    # print(ctrain.shape)
    # print(xtest.shape)
    # print(ytest.shape)
    # print(ptest.shape)
    x1_train=[i[0] for i in xtrain]
    x2_train=[i[1] for i in xtrain]
    # # print(x1_train)
    # # print(ctrain)
    # fig1=plt.figure()
    # xtrainImg=fig1.add_subplot(111)
    # xtrainImg.set_xlim(-3,5)
    # xtrainImg.set_ylim(-3,4)
    # xtrainImg.set_xlabel('x_2')
    # xtrainImg.set_xlabel('x_1')
    # for i,label in enumerate(ctrain):
    #     if(label==1):xtrainImg.scatter(x1_train[i],x2_train[i],color='b',marker='o')
    #     else : xtrainImg.scatter(x1_train[i],x2_train[i],color='r',marker='o')
    # plt.show()

    '''
     # test the lineReg
    #  # '''
    # results = []
    # for i in range(1, 20, 1):
    #     result = lineReg(xtrain, ctrain, i, xtest, ytest, ptest)
    #     results.append(result)
    # printfResults(results)

    '''
     test for redge Reg
    '''
    ##设定不同的惩罚系数
    # alpha_range=[0,0.1,0.01,0.001,10,100]
    # results =[]
    # for j in  alpha_range:
    #     result=ridgeReg(xtrain,ctrain,7,xtest,ytest,ptest,j)
    #     results.append(result)
    # printfResults(results)


    '''
    test for logistic Reg polynomials
    
    '''
    # result = logisticReg(xtrain, ctrain, 5, xtest, ytest, ptest)
    # print(result)
    # results = []
    # for i in range(1,20,1):
    #     result = logisticReg(xtrain,ctrain,i,xtest,ytest,ptest)
    #     results.append(result)
    # printfResults(results)

    '''
       test for losso
    #    '''

    # results=[]
    # alpha_range = [0.1, 0.01, 0.001,1, 10, 100]
    # for j in alpha_range:
    #     tmp = (lossoReg(xtrain, ctrain,7,xtest, ytest, ptest, j))
    #     results.append(tmp)
    # printfResults(results)

    '''
       test for LDA
    '''
    # result=LDARef(xtrain,ctrain,xtest,ytest,ptest)
    # print(result)

    '''
    test k-means
    '''
    results=[]
    for i in range(1,20,1):
        result=kmeans(xtrain,ctrain,xtest,ytest,ptest,i)
        results.append(result)
    printfResults(results)

