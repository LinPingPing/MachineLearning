# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:25:12 2018

@author: Liu
"""

import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

def linearRegression():
    print('加载数据\n')
    data = loadtxtAndcsv_data('data.txt',',',np.float64)
    X = np.array(data[:,0:-1],dtype=np.float64)
    y = np.array(data[:,-1],dtype=np.float64)

    #归一化操作
    scaler = StandardScaler()
    scaler.fit(X)
    X_train = scaler.transform(X)
    X_test = scaler.transform(np.array([1650,3]).reshape(1,-1))#只有一个值

    #线性拟合
    model = linear_model.LinearRegression()
    model.fit(X_train,y)

    #预测结果
    result = model.predict(X_test)
    print(model.coef_) # Coefficient of the features 决策函数中的特征系数
    print(model.intercept_) # 又名bias偏置,若设置为False，则为0
    print(result)


def loadtxtAndcsv_data(fileName,split,dataType):
    return np.loadtxt(fileName,delimiter=split,dtype=dataType)

def loadnpy_data(fileName):
    return np.load(fileName)


if __name__=="__main__":
    linearRegression()
