# coding:utf-8
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname =r"c:\windows\fonts\simsun.ttc", size=14)   # 解决windows环境下画图汉字乱码问题

#加载txt   csv文件
def loadtxtAndcsv_data(fileName,split,dataType):
    try:
        return np.loadtxt(fileName,delimiter=split,dtype=dataType)
    except:
        raise '文件加载失败'

#加载npy文件
def loadnpy_data(fileName):
    return np.load(fileName)

#做二维图
def plot_data(X,y):
    pos = np.where(y==1)   #找到y==1的坐标位置
    neg = np.where(y==0)   #找到y=0的位置
    #做图
    plt.figure(figsize=(15,12))
    plt.plot(X[pos,0],X[pos,1],'ro')
    plt.plot(X[neg,0],X[neg,1],'bo')
    plt.title(u'两个类别的散点图',fontproperties = font)
    plt.show()

#映射为多项式,由两个特征映射为5个特征
def mapFeature(X1,X2):
    degree = 2
    out = np.ones((X1.shape[0],1))
    '''
    以degree = 2为例，X1和X2映射为X1X2、X1^2、X2^2
    '''
    for i in np.arange(1,degree+1):
        for j in range(i+1):
            temp = X1**(i-j)*(X2**j)
            out = np.hstack((out,temp.reshape(-1,1)))
    return out

#sigmoid函数
def sigmoid(z):
    h = np.zeros((len(z),1))  #初试化长度
    h = 1.0/(1.0+np.exp(-z))
    return h

#定义代价函数
def costFunction(init_theta,X,y,init_lambda):
    m = len(y)
    J = 0   #初试化代价值

    h = sigmoid(np.dot(X,init_theta))
    theta1 = init_theta.copy()
    theta1[0] = 0  #theta(0)为一个常数项，没必要正则化

    temp = np.dot(np.transpose(theta1),theta1)  #正则化项
    J = (-np.dot(np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h))
    +temp*init_lambda/2)/m   #正则化的代价方程
    return J

#计算梯度
def gradient(theta,X,y,init_lambda):
    m = len(y)
    grad = np.zeros((theta.shape[0]))

    h = sigmoid(np.dot(X,theta))
    theta1 = theta.copy()
    theta1[0]=0

    grad = np.dot(np.transpose(X),h-y)/m+init_lambda/m*theta1#正则化梯度
    return grad

#画决策边界
def plotDecisionBoundary(theta,X,y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    #做图
    plt.figure(figsize=(15,12))
    plt.plot(X[pos,0],X[pos,1],'ro')
    plt.plot(X[neg,0],X[neg,1],'bo')
    plt.title(u"决策边界",fontproperties = font)

    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)

    z = np.zeros((len(u),len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(mapFeature(u[i].reshape(1,-1),v[j].reshape(1,-1)),theta)

    z = np.transpose(z)
    plt.contour(u,v,z,[0,0.01],linewidths=2.0)# 画等高线，范围在【0，0.01】，为近似边界
    plt.show()

#预测
def predict(X,theta):
    m = X.shape[0]
    p = np.zeros((m,1))
    p = sigmoid(np.dot(X,theta))  #预测结果是一个概率值

    for i in range(m):
        if p[i]>0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p


def LogisticRegression():
    data = loadtxtAndcsv_data('data2.txt',',',np.float64)
    X = data[:,0:-1]
    y = data[:,-1]#最后一列为标签

    plot_data(X,y)#作图

    X = mapFeature(X[:,0],X[:,1]) #特征映射
    theta = np.zeros((X.shape[1],1)) #初试化theta
    init_lambda= 0.01  #设置学习率

    J = costFunction(theta,X,y,init_lambda)
    print(J)
    #result = optimize.fmin(costFunction, initial_theta, args=(X,y,initial_lambda))    #直接使用最小化的方法，效果不好
    '''调用scipy中的优化算法fmin_bfgs（拟牛顿法Broyden-Fletcher-Goldfarb-Shanno）
    - costFunction是自己实现的一个求代价的函数，
    - initial_theta表示初始化的值,
    - fprime指定costFunction的梯度
    - args是其余测参数，以元组的形式传入，最后会将最小化costFunction的theta返回
    '''
    result = optimize.fmin_bfgs(costFunction,theta,fprime=gradient,args=(X,y,init_lambda))  #返回最终theta值
    p = predict(X,result)
    print('在训练集上的准确度为 %f%%'%np.mean(np.float64(p==y)*100))

    X = data[:,0:-1]
    y = data[:,-1]
    plotDecisionBoundary(result,X,y)

if __name__ == "__main__":
    LogisticRegression()
