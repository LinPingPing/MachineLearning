import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import optimize
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 解决windows环境下画图汉字乱码问题

def loadmat_data(fileName):
    return sio.loadmat(fileName)

#显示100个数字
def display_data(imgData):
    sum = 0
    '''
    显示100个数（若是一个一个绘制将会非常慢，可以将要画的数字整理好，放到一个矩阵中，
    显示这个矩阵即可）
    - 初始化一个二维数组
    - 将每行的数据调整成图像的矩阵，放进二维数组
    - 显示即可
    '''
    pad = 1
    display_array = -np.ones((pad+10*(20+pad),pad+10*(20+pad)))
    for i in range(10):
        for j in range(10):
            display_array[pad+i*(20+pad):pad+i*(20+pad)+20,pad+j*(20+pad):pad+j*(20+pad)+20
            ] = (imgData[sum,:].reshape(20,20,order='F'))
            sum+=1
    plt.imshow(display_array,cmap = 'gray')#显示灰度图像
    plt.axis('off')
    plt.show()



#定义sigmiod函数
def sigmoid(z):
    h = np.zeros((len(z),1))

    h = 1/(1+np.exp(-z))
    return h

#定义损失函数
def costFunction(theta,X,y,Lambda):
    m = len(y)
    J = 0

    h = sigmoid(np.dot(X,theta))
    theta1 = theta.copy()
    theta1[0] = 0

    temp = np.dot(np.transpose(theta1),theta)
    J = (-np.dot(np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h))+temp*Lambda/2)/m
    return J

#定义梯度
def gradient(theta,X,y,Lambda):
    m = len(y)
    grad = np.zeros(theta.shape[0])

    h = sigmoid(np.dot(X,theta))
    theta1 = theta.copy()
    theta1[0] = 0

    grad =np.dot(np.transpose(X),h-y)/m+Lambda/m*theta1
    return grad

def predict_oneVsAll(all_theta,X):
    '''
    n:样本数
    m:特征数
    X:n*m的矩阵
    p:标签类别
    theta:p*m
    '''
    n = X.shape[0] #样本数
    num_labels = all_theta.shape[0]  #类别数量
    p = np.zeros((n,1))
    X = np.hstack((np.ones((n,1)),X))#在X前面加一列1

    h = sigmoid(np.dot(X,np.transpose(all_theta)))#预测返回n*p的矩阵
    '''
    返回h中每一行最大值所在的列号
    - np.argmax(h, axis=1)返回h中每一行的最大值（是某个数字的最大概率）的序号
    '''
    p = np.argmax(h,axis=1).reshape(-1,1)
    return p

#相当于进行10次逻辑回归分类
def oneVsAll(X,y,num_labels,Lambda):
    m,n = X.shape
    all_theta = np.zeros((n+1,num_labels)) #每一列对应该类的theta，共有10类
    X= np.hstack((np.ones((m,1)),X))
    class_y = np.zeros((m,num_labels))
    theta = np.zeros((n+1,1))  #初始化一个类的theta

    #映射y
    for i in range(num_labels):
        class_y[:,i] = np.int32(y==i).reshape(1,-1)

    '''遍历每个分类，计算对应的theta值'''
    for i in range(num_labels):
        result_theta = optimize.fmin_bfgs(costFunction,theta,fprime=gradient,args=(X,class_y[:,i],Lambda))
        all_theta[:,i] = result_theta.reshape(1,-1)

    all_theta = np.transpose(all_theta)
    return all_theta

def logisticRegression_oneVsAll():
    data = loadmat_data('data_digits.mat')
    X = data['X']  #获取X数据，每行对应20*20的数字
    y = data['y']
    m,n = X.shape
    num_labels = 10

    #随机显示几行数据
    rand_index = [t for t in [np.random.randint(0,m) for x in range(100)]]
    display_data(X[rand_index,:])

    print(y.shape)
    Lambda= 0.1 #正则化系数
    all_theta = oneVsAll(X,y,num_labels,Lambda)

    p = predict_oneVsAll(all_theta,X)
    print('预测准确度为:%f%%' % np.mean(np.float64(p==y.reshape(-1,1))*100))

if __name__ == '__main__':
    logisticRegression_oneVsAll()
