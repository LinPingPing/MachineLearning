import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 解决windows环境下画图汉字乱码问题
# 定义代价函数
def computerCost(X,y,theta):
    m = len(y)
    j = 0
    theta = theta.reshape(3,-1)
    j = (np.transpose(X.dot(theta)-y)).dot(X.dot(theta)-y)/(2*m) #计算代价J
    return j

# 计算梯度
# num_iters: 迭代次数
# alpha：学习速率0.1 0.3 0.01 0.03
def gradientDescent(X,y,theta,alpha,num_iters):
    	m = len(y);
    	n =len(theta)
    
    	temp = np.matrix(np.zeros((n,num_iters)))
    
    	j_history = np.zeros((num_iters,1))
    
    	for i in range(num_iters):
    		h = np.dot(X,theta)
    		temp[:,i] = theta - ((alpha/m)*(np.dot(np.transpose(X),h-y)))
    		theta = temp[:,i]
    		j_history[i] = computerCost(X,y,theta)
    	return theta , j_history


# 特征归一化
def featureNormaliza(X):
    	X_norm = np.array(X)
    	mu = np.zeros((1,X.shape[1])) #均值
    	sigma = np.zeros((1,X.shape[1]))#标准差
    	mu = np.mean(X_norm,0) #0指定为列，1指定为行
    	sigma = np.std(X_norm,0)#求每一列的标准差
    	for i in range(X.shape[1]):  #遍历列
            X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i]# 归一化
    
    	return X_norm,mu,sigma

#主程序
def linearRession(alpha=0.01,num_iters=400):
    print('加载数据....\n')
    
    data = loadtxtAndcsv_data('data.txt',',',np.float64)
    X =data[:,0:-1]
    y = data[:,-1]
    m = len(y)
    col = data.shape[1]
    
    X,mu,sigma = featureNormaliza(X) #归一化数据集
    plot2d(X)       #查看归一化后的图形
    
    X = np.hstack((np.ones((m,1)),X))   #在X前面加一列1
    
    print('\n执行梯度下降算法\n')
    
    theta = np.zeros((col,1))
    y = y.reshape(-1,1)  #转化为列向量
    theta, J_history=gradientDescent(X,y,theta,alpha,num_iters)
    
    plot_loss(J_history,num_iters)
    
    return mu,sigma,theta

#加载txt csv文件
def loadtxtAndcsv_data(fileName,split,dataType):
    return np.loadtxt(fileName,delimiter=split,dtype=dataType)

#加载npy文件
def loadnpy_data(fileName):
    return np.load(fileName)

#画二维图
def plot2d(x):
    plt.scatter(x[:,0],x[:,1])
    plt.show()
    
#每次迭代损失图
def plot_loss(J_history,num_iters):
    x = np.arange(1,num_iters+1)
    plt.plot(x,J_history)
    plt.xlabel('迭代次数',fontproperties=font)
    plt.ylabel('损失',fontproperties=font)
    plt.title('损失随迭代次数变化',fontproperties=font)
    plt.show()
    
#测试linearRession函数
def testLinearRession():
    mu,sigma,theta = linearRession(0.01,400)
    
#预测学习效果
def predict(mu,sigma,theta):
    result = 0
    predict = np.array([1650,3])
    norm_predict = (predict-mu)/sigma
    final_predicction = np.hstack((np.ones(1),norm_predict))
    
    result = np.dot(final_predicction,theta)
    return result

if __name__ == '__main__':
    testLinearRession()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
