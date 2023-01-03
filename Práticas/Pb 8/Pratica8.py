# Exercise 1 

import numpy as np
from scipy.ndimage import convolve


def convolution(input, kernel, stride):


    width = int((input.shape[0] - kernel.shape[0] / stride) + 1)
    height = width
    output = []
    for i in range (0 , height):
        for j in range (0 , width):

            output.append(np.sum(input[i:kernel.shape[0]+i, j:kernel.shape[1]+j] * (kernel)))

    output=np.array(output).reshape((height, width))
    print(output)
    return output

def max_pooling(output, stride , window ):

    output_new =[]
    for i in range(0,output.shape[0], stride):
        for j in range(0,output.shape[0], stride):
            output_new.append(np.max(output[i:window+i, j:window+j]))

    output_new=np.array(output_new).reshape((window, window ))
    return output_new
        










######################## MAIN #################################
stride1 = 1
size_output = 4
input1 = np.array([[20, 35, 35, 35, 35, 20],
                   [29, 46, 44, 42, 42, 27],
                   [16, 25, 21, 19, 19, 12],
                   [66, 120, 116, 154, 114, 62],
                   [74, 216, 174, 252, 172, 112],
                   [70, 210, 170, 250, 170, 110]])

kernel1 = np.array([[1 , 1 , 1], [1, 0 , 1], [ 1, 1, 1]])

output1 = convolution(input1, kernel1 , stride1)

stride_pooling = 2
window_shape = 2
output1_pooling = max_pooling(output1, stride_pooling, window_shape )    

print(output1_pooling)