import numpy as np
import pandas as pd
# 定义代价函数
def computerCost(X,y,theta):
    	m = len(y)
    	j = 0
    	j = (np.transpose(X*theta-y))*(X*theta-y)/(2*m)
    	return j

# 计算梯度
# num_iters: 迭代次数
# alpha：学习速率0.1 0.3 0.01 0.03
def gradientDescent(X,y,theta,alpha,num_iters):
    	m = len(y);
    	n =len(theta)
    
    	temp = np.matrix(np.zeros(n,num_iters))
    
    	j_history = np.zeros(num_iters,1)
    
    	for i in range(num_iters):
    		h = np.dot(X,theta)
    		temp = theta -((alpha/m)*(np.dot(np.transpose(X),h-y)))
    		theta = temp[:,i]
    		j_history[i] = computerCost(X,y,theta)
    		print(".")
    	return theta,j_history


# 特征归一化
def featureNormaliza(X):
    	X_norm = np.array(X)
    	mu = np.zeros((1,X.shape[1])) #均值
    	sigma = np.zeros((1,X.shape[1]))#标准差
    	mu = np.mean(X_norm,0) #0指定为列，1指定为行
    	sigma = np.std(X_norm,0)#求每一列的标准差
    	for i in range(X.shape[1]):  #遍历列
            X_norm[:i] = (X_norm[:i]-mu[i])/sigma[i]# 归一化
    
    	return X_norm,mu,sigma

