import numpy as np
import matplotlib.pyplot as plt 
from numpy import random
from sklearn import metrics
# Exercise 1 
# 1.1

x = np.array([[1 ,-1 ,0 ],[1, 0, 0.25],[1,1,1],[1,1, -1]])
y = [-1,1 , 1,-1]
w_k = np.zeros(3)
epochs=0

while epochs<5:
    for i in range(0,4):
        y_predict = np.matmul(w_k.T, x[i].T )
        if y_predict >= 0:
            y_predict=1
        else:
            y_predict=-1
        if y_predict != y[i]:
            w_k_1 = w_k + y[i]*x[i].T
            w_k = w_k_1
    epochs=epochs+1

#1.2 

plt.plot(x, (-x*w_k[2]  -w_k[0])/w_k[1], linestyle='--')
plt.show()

#1.3 
x_n = np.array([1,0,1])
if (np.matmul(w_k.T ,x_n) ) >=0:
    result = 1
else:
    result = -1

print('The result is :', result)

# 1.4 

W_k = random.uniform(size=(3))
epochs=0

while epochs<50:
    for i in range(0,4):
        y_predict = np.matmul(W_k.T, x[i].T )
        if y_predict >= 0:
            y_predict=1
        else:
            y_predict=-1
        if y_predict != y[i]:
            W_k_1 = W_k + y[i]*x[i].T
            W_k = W_k_1
    epochs=epochs+1


plt.plot(x, (-x*W_k[2]  -W_k[0])/W_k[1], linestyle='--')
plt.show()

# EXERCISE 2

#2.1
num_labels = 2
num_features = 2

C0 = random.normal([0,0],1, size=[10,2])
C1 = random.normal([0,3],1, size=[10,2])
C2 = random.normal([2,2],1, size=[10,2])
plt.scatter(C0[:,0],C0[:,1], color = 'red')
plt.scatter(C1[:,0],C1[:,1], color = 'green')
plt.scatter(C2[:,0],C2[:,1], color = 'blue')
plt.show()

#2.2

Featuers = np.concatenate((C0, C1, C2), axis=0)
Featuers = np.concatenate([np.ones((30,1)), Featuers], axis=1)

n_iter=100
Y = labels = np.array([0]*10 + [1]*10 + [2]*10)
ind = np.random.permutation(30)
Featuers = Featuers[ind, :]
y_1 = Y[ind]

W = np.zeros((num_labels+1,num_features+1))

def multi_class_perceptron( weights, C, Y):
    epochs = 0

    while epochs < 100:
        for x, y in zip(C , Y): 
            y_pred = np.argmax(weights.dot(x))
            if y_pred != y: 
                weights[y,:] = weights[y,:] + x
                weights[y_pred,:] = weights[y_pred,:] - x
        epochs = epochs +1
    return weights
    
def accuracy(weights, C , Y):
    y_prediction = []
    for X in C:
        y_prediction.append(np.argmax(weights.dot(X)))

    y_p = np.array(y_prediction)
    count=0
    for true, pred in zip(Y, y_prediction):
        if true == pred:
            count =count+1
    print('Accuracy:', count/len(Y))

W_3 = multi_class_perceptron(W, Featuers, y_1)
accuracy(W_3, Featuers, y_1)

#EXERCISE 4

#Getting image data set 8x8

from sklearn.datasets import load_digits
data = load_digits()


# PLotting the images

""" import matplotlib.pyplot as plt
plt.gray()
for i in range(10):
    plt.matshow(data.images[i])
plt.show()  """

# Split the data

x = data.data
y = data.target
x = np.concatenate([np.ones((x.shape[0],1)), x], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

# Run our Perceptron 

W_1 = np.zeros((len(set(y))+1,x.shape[1]))
W_2 = multi_class_perceptron(W_1, X_train, y_train )
accuracy(W_2, X_test, y_test)

# Run the Sklearn algorithm of the Perceptron

from sklearn.linear_model import Perceptron
clf = Perceptron(fit_intercept=False, shuffle=False)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
