#networks
from functions import *
import numpy as np
import pickle

# ====================================

def forward_network(x):  # x는 np.array임. 다른 .py 파일에서 x_train이나 x_test같은걸 여기서 넣어주면 됨.
                         # 혹은 x_train[341]같은걸 실험삼아 넣어봐도 됨.

    with open("W1.pkl", "rb") as f:
        W1 = pickle.load(f)

    with open("W2.pkl", "rb") as f:
        W2 = pickle.load(f)

    with open("W3.pkl", "rb") as f:
        W3 = pickle.load(f)

    with open("B1.pkl", "rb") as f:
        B1 = pickle.load(f)

    with open("B2.pkl", "rb") as f:
        B2 = pickle.load(f)

    with open("B3.pkl", "rb") as f:
        B3 = pickle.load(f)

    A1 = np.dot(x, W1) + B1
    Z1 = sigmoid(A1)

    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)

    A3 = np.dot(Z2, W3) + B3
    Z3 = sigmoid(A3)

    Y = softmax(Z3)
    return Y
    #Y는 1*10꼴의 np.array임.

# ====================================

