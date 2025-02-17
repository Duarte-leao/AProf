#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils

import time


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)   
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        prediction = self.predict(x_i)
        if prediction != y_i:
          self.W[prediction] -= x_i
          self.W[y_i] += x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        un_probabilities = np.dot(self.W, x_i)
        probabilities = np.exp(un_probabilities) / np.sum(np.exp(un_probabilities))
        for j in range(np.shape(self.W)[0]):
            if j == y_i:
                error = probabilities[j] - 1
            else:
                error = probabilities[j]
            self.W[j] -= learning_rate * error * x_i


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().

    def softmax1(self, x):
        e_x = np.exp(x - np.max(x)) # - np.max(x) helps prevent underflow issues when exponentiating small values
        return e_x / e_x.sum(axis=0)

    def __init__(self, n_classes, n_features, hidden_size,layers):
        # Initialize an MLP with a single hidden layer.
        self.n_classes = n_classes 
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.weights1 = np.random.normal(0.1, 0.1, size=(self.hidden_size,self.n_features)) 
        self.weights2 = np.random.normal(0.1, 0.1, size=(self.n_classes, self.hidden_size))  
        self.bias1 = np.zeros((hidden_size,1))
        self.bias2 = np.zeros((self.n_classes,1))


    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        
        z1 = np.dot(self.weights1,X.T) + self.bias1
        h1 = np.maximum(0, z1)
        z2 = np.dot(self.weights2,h1) + self.bias2
        f = self.softmax1(z2)
        y_pred = np.argmax(f, axis=0)
        return y_pred.ravel()



    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat= self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        print(n_correct / n_possible)
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        for x, Y in zip(X,y):
            x = x.reshape((self.n_features,1)) 
            
            z1 = np.dot(self.weights1,x) + self.bias1 
            h1 = np.maximum(0, z1) # Relu activation
            z2 = np.dot(self.weights2,h1) + self.bias2 
            f = self.softmax1(z2) # Softmax activation

            y_one = np.zeros((self.n_classes,1))        
            y_one[Y] = 1
            
            dw2 =  np.dot((f - y_one), h1.T)
            db2 = f - y_one
            z1_d = z1.T > 0 # Relu derivative
            dz1 = np.dot(self.weights2.T,(f - y_one)) * z1_d.T
            dw1 = np.dot(dz1,x.T)
            db1 = dz1 

            self.weights2 = self.weights2 - learning_rate * dw2   
            self.bias2 = self.bias2 - learning_rate * db2
            self.weights1 = self.weights1 - learning_rate * dw1
            self.bias1 = self.bias1 - learning_rate * db1
        


    



def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        a = time.time()
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))
        print(valid_accs)

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
