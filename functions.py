# functions
import numpy as np
import pickle

#===================================

def step_function(x):
    tmp = x > 0
    return tmp.astype(np.int)

#===================================

def sigmoid(x):
    y = 1 / (1+np.exp(-x))
    return y

#===================================

def softmax(x):
    #x는 행렬임.
    c = np.max(x)
    y = np.exp(x-c) / np.sum(np.exp(x-c))
    return y

#====================================

def mean_square_error(y,t):
    return 0.5*np.sum((y-t)**2)

#====================================

def cross_entropy_error(y,t):
    epsilon = 1e-8
    return -np.sum(t*np.log(y+delta))

#====================================


