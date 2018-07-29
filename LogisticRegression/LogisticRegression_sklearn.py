# coding:utf-8
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler#引入归一化包
from sklearn.cross_validation import train_test_split

#加载文件
def loadtxtAndcsv_data(fileName,split,dataType):
    return np.loadtxt(fileName,delimiter=split,dtype=dataType)

def loadnpy_data(fileName):
    return np.load(fileName)

def logisticRegression():
    data = loadtxtAndcsv_data('data1.txt',',',np.float64)
    X = data[:,0:-1]
    y = data[:,-1]

    #划分数据集和测试集
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 20)

    #归一化
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.fit_transform(x_test)

    #逻辑回归
    model = LogisticRegression()
    model.fit(x_train,y_train)

    #预测
    predict = model.predict(x_test)
    right = sum(np.float64(y_test==predict))#将bool类型转化为数字

    predict = np.hstack((y_test.reshape(-1,1),predict.reshape(-1,1)))
    print(predict)
    print('测试的准确率:%f %%' % (right*100/len(y_test)))


if __name__ == '__main__':
    logisticRegression()
