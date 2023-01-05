# Exercise 1 

import numpy as np
from scipy.ndimage import convolve
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


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


# Exercise 1 

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

#Exercise 3

class CNN(nn.Module):
    
    def __init__(self):

        # call the parent constructor
        super(CNN, self).__init__()

        # initialize first set of CONV => POOL layers => RELU 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # initialize second set of CONV => RELU => POOL layers

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # initialize first (and only) set of FC => RELU layers

        self.fc1 = nn.Linear(in_features=320, out_features=50)  #The number 320 depedens of the output create by the preovius layers

        # initialize our softmax classifier

        self.fc2 = nn.Linear(in_features=50, out_features=10)
        self.logSoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x ):
        # Batch size = 64, images 28x28 =>
        #x.shape = [64, 1, 28, 28]
        x = nn.functional.relu(self.maxpool1(self.conv1(x)))
        # Convolution with 5x5 filter without padding and 10 channels =>
        #     x.shape = [64, 10, 24, 24] since 24 = 28 - 5 + 1
        # Max pooling with stride of 2 =>
        #     x.shape = [64, 10, 12, 12]
        x = self.conv2_drop(self.conv2(x))
        x = self.maxpool2(x)
        x = nn.functional.relu(x)
        # Convolution with 5x5 filter without padding and 20 channels =>
        #     x.shape = [64, 20, 8, 8] since 8 = 12 - 5 + 1
        # Max pooling with stride of 2 =>
        #     x.shape = [64, 20, 4, 4] since 20x4x4 = 320
        x = x.view(-1, 320)
        # Reshape =>
        #x.shape = [64, 320]
        x = self.fc1(x)

        x = nn.functional.relu(x)

        x  = self.fc2(x)

        output = self.logSoftmax(x)

        return output




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mnist_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
mnist_train_dataset = datasets.MNIST('../data', download=True, train=True,transform=mnist_transform)
print(mnist_train_dataset)
mnist_val_dataset = datasets.MNIST('../data', download=True, train=False,transform=mnist_transform)
print(mnist_train_dataset)
LR = 0.01
batch = 64
EPOCHS = 10

model = CNN().to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainSteps = len(mnist_train_dataset) // batch
valSteps = len(mnist_val_dataset) // batch

H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

mnist_train_dataset = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True)
mnist_val_dataset = DataLoader(mnist_val_dataset, batch_size=64, shuffle=True)

#mnist_train_dataloader_numpy = DataLoader(mnist_train_dataset, batch_size=len(mnist_train_dataset))
#mnist_val_dataloader_numpy = DataLoader(mnist_val_dataset, batch_size=len(mnist_val_dataset))

#torch.tensor(mnist_train_dataloader_numpy)

for e in range(0, EPOCHS):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    valCorrect = 0
    # loop over the training set
    for (x, y) in mnist_train_dataset:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        output = model(x)
        loss = criterion(output, y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += loss
        trainCorrect += (output.argmax(1) == y).type(torch.float).sum().item()
    
    # switch off autograd for evaluation
    with torch.no_grad():
    # set the model in evaluation mode
        model.eval()
    # loop over the validation set
    for (x, y) in mnist_val_dataset:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # make the predictions and calculate the validation loss
        output = model(x)
        torch.tensor(y, dtype=torch.int8)
        totalValLoss += criterion(output, y)
        # calculate the number of correct predictions
        valCorrect += (output.argmax(1) == y).type(
            torch.float).sum().item()

        # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    # calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(mnist_train_dataset)
    valCorrect = valCorrect / len(mnist_val_dataset)
    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
        avgValLoss, valCorrect)) 

""" for epoch in range(EPOCHS):
    for (x, y) in mnist_train_dataset:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        output = model(x)
        loss = criterion(output, y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() """