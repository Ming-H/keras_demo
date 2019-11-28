import numpy as np
# from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier as kNN
from joblib import dump, load
# from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Sequential, layers 
from keras.utils import to_categorical

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


def load_save_model(modelName, load=True):
    if load:
        return load(modelName + '.joblib')
    else:
        return dump(modelName + '.joblib')


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


def train(modelName, x_train, y_train, x_test, y_test):
    """
    train model
    """
    if modelName=='knn1' or if modelName=='knn2':
        model.fit(x_train,y_train)

    elif modelName=='deepModel1' or modelName=='deepModel2':
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        train_history=model.fit(x_train,y_train,epochs=10,batch_size=300,verbose=2)


def task1():
    """
    (a) the cross-validation of 5 subsamples
    """


def task2():
    """
    (b) the confusion matrix, and
    """


def task3():
    """
    (c) the ROC curve for one class vs. all other classes
    """



if __name__ == '__main__':
    # load data
    x_train, y_train, x_test, y_test = load_data()



    

