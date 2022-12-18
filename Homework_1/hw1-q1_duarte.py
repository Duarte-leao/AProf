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
    # def update_weight(self, x_i, y_i, **kwargs):
    #     raise NotImplementedError

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
        print(n_correct / n_possible)
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
    def __init__(self, n_classes, n_features, hidden_size, layers):
        # Initialize an MLP with a single hidden layer.
        self.n_classes = n_classes
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.W = np.array([np.random.normal(0.1,0.1**2, size=(hidden_size, n_features)), np.random.normal(0.1,0.1**2, size=(n_classes, hidden_size))], dtype=object) 
        self.biases = np.array([np.zeros((hidden_size, 1)), np.zeros((n_classes, 1))], dtype=object)



    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        scores = np.dot(self.W[1], np.maximum(0, np.dot(self.W[0], X.T) + self.biases[0])) + self.biases[1] 
        pred_labels = scores.argmax(axis=0)  
        return pred_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        print(n_correct / n_possible)
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, learning_rate) 

    def update_weight(self, x_i, y_i, learning_rate):
        hidden_nodes_s = np.array([np.dot(self.W[0], x_i)]) # hidden layer
        hidden_nodes_z = np.maximum(0, hidden_nodes_s.T + self.biases[0]) # ReLU of hidden layer
        last_nodes_s = np.dot(self.W[1], hidden_nodes_z) # output layer
        y_pred = self.softmax(last_nodes_s + self.biases[1]) # softmax of output layer
        y_true = np.zeros((self.n_classes,1)) # gold labels
        y_true[y_i] = 1     
        ReLU_prime = np.where(hidden_nodes_s.T > 0, 1, 0) # ReLU derivatives of hidden layer
        d_Loss = y_pred - y_true # loss derivative
        self.biases[1] -= learning_rate * d_Loss # update biases of output layer
        aux_var = learning_rate * np.dot(self.W[1].T, d_Loss) * ReLU_prime # auxiliar variable
        self.biases[0] -= aux_var # update biases of hidden layer
        self.W[0] -= aux_var * x_i # update weights of hidden layer
        self.W[1] -= learning_rate * d_Loss * hidden_nodes_z.T  # update weights of output layer



    def softmax(self, x):
        e_x = np.exp(x - np.max(x)) # - np.max(x) helps prevent underflow issues when exponentiating small values
        return e_x / e_x.sum(axis=0)



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
        print('time',time.time()-a)
    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
