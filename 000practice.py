import numpy as np
import pickle
from networks import forward_network
from functions import *

with open("x_test.pkl", "rb") as f:
    x_for_test = pickle.load(f)

with open("label_test.pkl", "rb") as f:
    label_for_test = pickle.load(f)

index_for_test = 2008
x = x_for_test[index_for_test] #x는 (1,784)임. x_for_test는 (10000,784)임. x_for_test[0]은 (1,784)임.
label = label_for_test[index_for_test] #label은 (1,1)임. label_for_test는 (10000,10)임. label_for_test[0]은 (1,10)임.
#print(x)
z = forward_network(x) #z는 (1,10)임. forward_network(x)는 (1,10)임.
Is_this_answer = np.argmax(z) #Is_this_answer는 (1,1)임. np.argmax(z)는 (1,1)임.
print("inha :", z)
print("실제 답 :", label, "정답 후보 :", Is_this_answer)

print(cross_entropy_error(z, label))


# count = 0
# for i in range(len(x_for_test)):
#     please_be_answer = np.argmax(forward_network(x_for_test[i]))
#     real_answer = label_for_test[i]
#     if (please_be_answer == real_answer):
#         count += 1
# print("Accuracy :", float(count)/len(x_for_test) * 100, "%")