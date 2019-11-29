# -*- coding:utf-8 -*- 

import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier as kNN
from joblib import dump, load
from tensorflow.keras import Sequential, layers 
from keras.utils import to_categorical
from tensorflow.keras.models import load_model
from collections import defaultdict
from sklearn import metrics
import matplotlib.pyplot as plt


"""
(a) the cross-validation of 5 subsamples, 
(b) the confusion matrix, and
(c) the ROC curve for one class vs. all other classes
"""

def load_data():
    digits, target = load_digits(return_X_y = True)
    N, D = digits.shape
    stone = (3 * N) // 4
    train_x, train_y = digits[:stone], target[:stone]
    test_x, test_y = digits[stone:], target[stone:]
    return train_x, train_y, test_x, test_y


def prepare(modelName,  x_train, y_train, x_test, y_test):
    # prepare for deep model 
    if modelName == 'deepModel1' :
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    elif modelName == 'deepModel2':
        x_train = x_train.reshape(x_train.shape[0],8,8,1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0],8,8,1).astype('float32')
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    else:
        print('error')

    return x_train, y_train, x_test, y_test


def load_save_model(modelName):
    if modelName=='knn1' or modelName=='knn2':
        return load(modelName + '.joblib')

    elif modelName=='deepModel1' or modelName=='deepModel2':
        return load_model(modelName+'.h5')



def evalute(data, label):
    """
    自定义错误率
    """
    score = library_model.predict(data)
    error = np.sum(score != label) / label.size
    return error


def knn1():
    return kNN(n_neighbors = 10)


class Model:
    '''
    This is the hand-design kNN model needed in Part 3 and further
    '''
    def __init__(self,  k = 10):
        # k in the hyperparameter of kNN
        self.train_x = []
        self.train_y = []
        self.k = k
    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
    def predict(self, test_x):
        N_test, D = test_x.shape
        N_train, D = self.train_x.shape
        test_x = test_x.reshape(1, N_test, D)
        train_x = self.train_x.reshape(N_train, 1, D)
        diff = np.sum(np.power(train_x - test_x, 2), axis = 2)
        nearest = self.train_y[np.argpartition(diff, self.k, axis = 0)[:self.k,:]]
        
        labels = np.zeros(N_test, dtype = np.int8)
        for i in range(N_test):
            labels[i] = np.argmax(np.bincount(nearest[:,i]))
        return labels

def knn2():
    """
    model = Model()
    model.fit(train_x, train_y)
    dump(model, 'kNN.joblib')
    """
    return Model()


def deepModel1():
    """
    without convolutional layer 
    """
    model = Sequential() # 构建一个空的序贯模型
    model.add(layers.Dense(512,activation='relu',input_shape=(64,)))
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    return model  


def deepModel2():
    """
    with convolutional layer 
    """
    model = Sequential() # 构建一个空的序贯模型
    model.add(layers.Conv2D(16,(3,3),padding='same',input_shape=(8,8,1),activation='relu'))
    model.add(layers.Conv2D(8,(3,3),activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10,activation='softmax'))
    return model 


def train(modelName, x_train, y_train, x_test, y_test, saveFlag=True):
    """
    train model
    """
    if modelName=='knn1' or modelName=='knn2':
        if modelName=='knn1':
            model = knn1()
        else:
            model = knn2()
        model.fit(x_train,y_train)
        if saveFlag == 'True':
            dump(model, modelName+ '.joblib')

    elif modelName=='deepModel1' or modelName=='deepModel2':
        if modelName=='deepModel1':
            model = deepModel1()
        else:
            model = deepModel2()
        
        x_train, y_train, x_test, y_test = prepare(modelName, x_train, y_train, x_test, y_test)

        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.fit(x_train,y_train,epochs=10,batch_size=300,verbose=2)
        
        if saveFlag == 'True':
            model_save_path = modelName + ".h5"
            model.save(model_save_path)
    
    return model, x_test, y_test



def task1():
    """
    (a) the cross-validation of 5 subsamples
    """
    d1 = defaultdict(list)
    digits, target = load_digits(return_X_y = True)
    N, D = digits.shape
    stone = N // 5
    start = 0
    end = N // 5
    for i in range(5):
        x_train = np.vstack((digits[0:start],digits[end:N]))
        x_test = digits[start: end]
        y_train = np.vstack((target[0:start],target[end:N]))
        y_test = target[start: end]
        
        for modelName in ['knn1', 'knn2', 'deepModel1', 'deepModel2']:
            model, x_test, y_test = train(modelName, x_train, y_train, \
                                          x_test, y_test, saveFlag=False)

            y_pre = model.predict(x_test)
            # cal acc
            error = evalute(y_pre, y_test)
            if modelName not in d1:
                d1[modelName] = []
            else:
                d1[modelName].append(error)
            
        start += stone
        end += stone
    # 平均错误率
    for key in d1:
        d1[key] = sum(d1[key]) / len(d1[key]) 
    print(d1)
        
    
def task2():
    """
    (b) the confusion matrix, and
    """
    train_x, train_y, test_x, test_y = load_data()
    i = 0
    for modelName in ['knn1', 'knn2', 'deepModel1', 'deepModel2']:
        if i>=2:
            Flag = True
        else:
            Flag = False
        model, x_test, y_test = train(modelName, train_x, train_y, 
                                      test_x, test_y, saveFlag=Flag)  
        
        if modelName == 'knn1' or modelName == 'knn2':
            y_pred = model.predict(x_test)
            print("Confusion_matrix  of %s is: \n" % modelName, \
                              metrics.confusion_matrix(test_y, y_pred))

        elif modelName == 'deepModel1' or modelName == 'deepModel2':
            y_pred = model.predict_classes(x_test)            
            print("Confusion_matrix  of %s is: \n" % modelName, \
                          metrics.confusion_matrix(test_y, y_pred))
        
        
def task3():
    """
    (c) the ROC curve for one class vs. all other classes
    """
    x_train, y_train,  x_test, y_test = load_data()
    y_train = [0 if item==0 else 1 for item in y_train]
    y_test = [0 if item==0 else 1 for item in y_test]

    fpr = [] 
    tpr = [] 
    for modelName in ['knn1', 'knn2', 'deepModel1', 'deepModel2']:
        model, x_test, y_test = train(modelName, x_train, y_train, \
                                             x_test, y_test, saveFlag=False)  
        y_pred = model.predict(x_test)
        fpr.append(metrics.roc_curve(y_test, y_pred)[0]) 
        tpr.append(metrics.roc_curve(y_test, y_pred)[1]) 
    
    plt.plot(fpr[0], tpr[0], marker = 'o') 
    plt.plot(fpr[1], tpr[1], marker = '*') 
    plt.plot(fpr[2], tpr[2], marker = '^') 
    plt.plot(fpr[3], tpr[3], marker = '.') 
    plt.show()
    


if __name__ == '__main__':
    print("begin...")
    task1()
    task2()
    task3()
    print("over!")






