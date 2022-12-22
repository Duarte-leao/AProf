import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import math

#EXERCISE 1
#1.1
def linear_regression(x,y, x_new):


    #closed form solution 
    w1 = np.linalg.inv(np.matmul(x.T,x))
    W = np.matmul(np.matmul(w1, x.T), y)
    print(W)

    #1.2 
    y_pred = np.matmul(W, x_new)
    print(y_pred)
    return W

def mse(W, x, y):
    #1.4
    y_pred = []
    for X in x:
        y_pred.append((X[1]*W[1] + W[0]))
    y_pred = np.array(y_pred)

    mse = sum((y_pred- y)**2)/ len(y_pred)
    print(mse)
    return

def plot_2d(W, x, y):

    plt.scatter(x[:,1],y)
    print(x*W[1] + W[0])
    plt.plot(x, (x*W[1] + W[0]), linestyle='--')
    plt.show()
    return

def plot_2d_log(W, x, y):

    plt.scatter(x[:,1],y)
    print(x*W[1] + W[0])
    plt.plot(x, (x*W[1] + W[0]), linestyle='--')
    
    plt.show()
    return

""" def plot_3d(W, x , y):    
    y_pred = []
    for X in x:
        y_pred.append((X[2]*W[2]+ X[1]*W[1] + W[0]))
    y_pred = np.array(y_pred)
    print(y_pred)
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(x[:,1],y)
    ax.plot(x[:,1], y_pred, linestyle='--')
    plt.show()
    return  """

X=np.array([[1,-2],[1, -1] ,[1,0] , [1,2]])
Y=np.array([2, 3,1,-1])
X_new = [1,1]
#Exercise 1
weights = linear_regression(X, Y, X_new)
plot_2d(weights, X, Y)

X = np.array([[1,1,1],[1,2,1],[1,1,3],[1,3,3]])
Y = np.array([1.4,0.5, 2,2.5])
X_new = [1,2,3]
weights = linear_regression(X, Y, X_new)
#plot_3d(weights, X, Y)

#Exercise 3

X = np.array([[1,math.log(3)],[1,math.log(4)],[1,math.log(6)],[1,math.log(10)],[1, math.log(12)]])

Y = np.array([1.5,9.3, 23.4, 45.8,60.1])

weights = linear_regression(X, Y, [1,1])
plot_2d_log(weights, X, Y)

X = np.array([[1,(3**2)],[1,(4**2)],[1,(6**2)],[1,10**2],[1, 12**2]])
linear_regression(X, Y, [1,1])

#Exercice 4
alpha = 1
X = np.array([[1,-1, 0],[1,0,0.25],[1,1,1],[1,1,-1]])
Y = np.array([0,1,1,0])
weights = np.zeros((3))
dLoss =0
W1=np.zeros((3))
#stochastic Gradient descent (The dLoss  x , y and y_pred should have been random)
for i in range (0,1):
    
    y_pred = 1/(1+np.exp(np.matmul(weights.T, X.T)))
    #for x, y , y_p in zip(X,Y, y_pred):
    dLoss = dLoss + X[0,:]* (Y[0]-y_pred[0])
    W1 = W1 + alpha* dLoss
    weights = W1
    print(weights)
# %%


#Exercice 5

def accuracy(weights, C , Y):
    y_prediction = []
    for X in C:
        y_prediction.append(np.argmax(weights.dot(X)))

    y_p = np.array(y_prediction)
    count=0
    for true, pred in zip(Y, y_p):
        if true == pred:
            count =count+1
    print('Accuracy:', count/len(Y))

from sklearn.datasets import load_digits
data = load_digits()

x = data.data
y = data.target
x = np.concatenate([np.ones((x.shape[0],1)), x], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

weights = np.zeros((len(set(y)),x.shape[1]))
W1 = np.zeros((len(set(y)),x.shape[1]))

y_onehot = np.zeros((y_train.size, y_train.max()+1))
y_onehot[np.arange(y_train.size), y_train] = 1

alpha = 0.001
dLoss=0
for i in range (0,100):
    
    Y_pred = 1/(1+np.exp(np.matmul(weights,(X_train.T))))
    ind = np.random.randint(0, y_train.shape[0])
    dLoss = dLoss + np.matmul(X_train.T ,(y_onehot))- np.matmul(X_train.T,Y_pred.T)
    W1 = W1 + alpha* dLoss.T
    weights = W1

y_pred = accuracy(weights, X_test, y_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(fit_intercept=False, penalty='none')
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))