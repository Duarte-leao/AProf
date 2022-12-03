import numpy as np
import matplotlib.pyplot as plt 
# Exercise 1 

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
        print(w_k)
    epochs=epochs+1

plt.plot(x, (-x*w_k[2]  -w_k[0])/w_k[1], linestyle='--')
plt.show()