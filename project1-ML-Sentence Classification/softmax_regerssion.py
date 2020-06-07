#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/4/15 20:58
@author: phil
"""

import numpy as np


def softmax(z):
    # 稳定版本的softmax，对z的每一行进行softmax
    z -= np.max(z, axis=1, keepdims=True)  # 先减去该行的最大值
    z = np.exp(z)
    z /= np.sum(z, axis=1, keepdims=True)
    return z


class SoftmaxRegression:
    def __init__(self):
        self.num_of_class = None  # 类别数量
        self.n = None   # 数据个数
        self.m = None   # 数据维度
        self.weight = None  # 模型权重 shape (类别数，数据维度)
        self.learning_rate = None

    def fit(self, X, y, learning_rate=0.01, epoch=10, num_of_class=5, print_loss_steps=-1, update_strategy="batch"):
        self.n, self.m = X.shape
        self.num_of_class = num_of_class
        self.weight = np.random.randn(self.num_of_class, self.m)
        self.learning_rate = learning_rate

        # 将y换为独热码矩阵，每一行独热码表示一个label
        y_one_hot = np.zeros((self.n, self.num_of_class))
        for i in range(self.n):
            y_one_hot[i][y[i]] = 1

        loss_history = []

        for e in range(epoch):
            # X (n, m) 每一行表示一个样本
            # weight (C, m) 每一行处理一个类别
            loss = 0
            if update_strategy == "stochastic":
                rand_index = np.arange(len(X))
                np.random.shuffle(rand_index)
                for index in list(rand_index):
                    Xi = X[index].reshape(1, -1)
                    prob = Xi.dot(self.weight.T)
                    prob = softmax(prob).flatten()
                    loss += -np.log(prob[y[index]])
                    self.weight += Xi.reshape(1, self.m).T.dot((y_one_hot[index] - prob).reshape(1, self.num_of_class)).T

            if update_strategy == "batch":
                prob = X.dot(self.weight.T)   # (n, C) 每个样本被预测为各个类别
                prob = softmax(prob)

                for i in range(self.n):
                    loss -= np.log(prob[i][y[i]])

                # 书中给的损失函数
                weight_update = np.zeros_like(self.weight)
                for i in range(self.n):
                    weight_update += X[i].reshape(1, self.m).T.dot((y_one_hot[i] - prob[i]).reshape(1, self.num_of_class)).T
                self.weight += weight_update * self.learning_rate / self.n

            loss /= self.n
            loss_history.append(loss)
            if print_loss_steps != -1 and e % print_loss_steps == 0:
                print("epoch {} loss {}".format(e, loss))
        return loss_history

    def predict(self, X):
        prob = softmax(X.dot(self.weight.T))
        return prob.argmax(axis=1)

    def score(self, X, y):
        pred = self.predict(X)
        return np.sum(pred.reshape(y.shape) == y) / y.shape[0]