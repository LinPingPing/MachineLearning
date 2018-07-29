from scipy import io as spio
import numpy as np
#from sklearn import SVM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def logisticRegression():
    data = spio.loadmat('data_digits.mat')
    X= data['X']
    y = data['y']
    y = np.ravel(y) #转化为一维数组

    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

    model = LogisticRegression()
    model.fit(x_train,y_train)#预测

    predict = model.predict(x_test)
    print('预测准确度:%f%%'%np.mean(np.float64(predict==y_test)*100))

if __name__ == '__main__':
    logisticRegression()
