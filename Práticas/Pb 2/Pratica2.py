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

C0 = random.normal([0,0],[1,1], size=[10,2])
C1 = random.normal([0,3],[1,1], size=[10,2])
C2 = random.normal([2,2],[1,1], size=[10,2])
plt.scatter(C0[:,0],C0[:,1], color = 'red')
plt.scatter(C1[:,0],C1[:,1], color = 'green')
plt.scatter(C2[:,0],C2[:,1], color = 'blue')
plt.show()

#2.2

C = np.concatenate((C0, C1, C2), axis=0)
C = np.concatenate([np.ones((30,1)), C], axis=1)

n_iter=100
Y = labels = np.array([0]*10 + [1]*10 + [2]*10)
ind = np.random.permutation(30)
C = C[ind, :]
Y = Y[ind]

weights = np.zeros((num_labels+1,num_features+1))

while epochs < 100:
    for x, y in zip(C , Y): 
        print(y)
        y_pred = np.argmax(np.matmul(weights.T, x))
        if y_pred != y: 
            weights[y,:] = weights[y,:] + x
            weights[y_pred,:] = weights[y_pred,:] - x
    epochs = epochs +1
    
y_prediction = []

for X in C:
    y_prediction.append(np.argmax(np.matmul(weights.T, X)))

y_p = np.array(y_prediction)
count=0
for true, pred in zip(Y, y_prediction):
    print(true, pred)
    if true == pred:
        count =count+1
        print(count)
print('Accuracy:', count/len(Y))