import numpy as np

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier as kNN
from joblib import dump, load


def Part1():
    '''
    Part1: Show informations about datasets, and split training set and test set
    '''
    digits, target = load_digits(return_X_y = True)
    N, D = digits.shape
    C = np.max(target) + 1

    print("Number of data entries:", N)
    print("Number of classes:", C)
    
    for i in range(C):
        num = np.sum(target == i)
        print("Number of data entries for class {}:".format(i), num)
    
    for i in range(D):
        min_val = np.min(digits[:, i])
        max_val = np.max(digits[:, i])
        print("Feature {} has minimum value {} and maximum value {}".format(i, min_val, max_val))

    stone = (3 * N) // 4
    train_x, train_y = digits[:stone], target[:stone]
    test_x, test_y = digits[stone:], target[stone:]
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = Part1()

def Part2(train_x, train_y):
    '''
    Part2: Use library function to train a model and save it to a file 
    '''
    sklearn_model = kNN(n_neighbors = 10)
    sklearn_model.fit(train_x, train_y)
    dump(sklearn_model, 'sklearn_kNN.joblib')

Part2(train_x, train_y)

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


def Part3(train_x, train_y):
    '''
    Part3: Train my model and save it to a file
    '''
    model = Model()
    model.fit(train_x, train_y)
    dump(model, 'kNN.joblib')
    
Part3(train_x, train_y)


library_model = load('sklearn_kNN.joblib')
model = load('kNN.joblib')

def Part4():
    '''
    Part4: shows training error and test error for each model
    '''
    train_score = library_model.predict(train_x)
    train_error = np.sum(train_score != train_y) / train_y.size
    print("Training error of library model: ", train_error)
    test_score = library_model.predict(test_x)
    test_error = np.sum(test_score != test_y) / test_y.size
    print("Test error of library model: ", test_error)

    train_score = model.predict(train_x)
    train_error = np.sum(train_score != train_y) / train_y.size
    print("Training error of library model: ", train_error)
    test_score = model.predict(test_x)
    test_error = np.sum(test_score != test_y) / test_y.size
    print("Test error of library model: ", test_error)

Part4()

'''
Part5: Query function and examples
'''

print("Size of test set:", test_y.size)

def library_kNN_query(test_index):
    y = test_x[test_index]
    label = library_model.predict(y.reshape(1, -1))
    print("The image in test set with ID {} is likely to be number {} by library model".format(test_index, label[0]))
    print("In fact, it is {}".format(test_y[test_index]))


def kNN_query(test_index):
    y = test_x[test_index]
    label = model.predict(y.reshape(1, -1))
    print("The image in test set with ID {} is likely to be number {} by my model".format(test_index, label[0]))
    print("In fact, it is {}".format(test_y[test_index]))

# Example

index = 100
library_kNN_query(100)
kNN_query(100)