#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 22:27:00 2019

@author: spatri

Objective: Gradient Descent
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval
import torch

class gradient_descent():
    def __init__(self, data):
        self.y, self.x = data[:,1], data[:,0]
        a0, b0 = 1.8, 1.4 
        iterations = 100
        final_a_gd, final_b_gd, self.loss_gd = self.descent(a0, b0, iterations, learning_rate=0.05)
        final_a_mom, final_b_mom, self.loss_mom = self.momentum(a0, b0, iterations, gamma=0.9, eta=0.05)
        final_a_nest, final_b_nest, self.loss_nest = self.nesterov(a0, b0, iterations, gamma=0.9, eta=0.05)
        it = np.arange(1, iterations+1, 1)
        fig = plt.figure(figsize=(9, 3), dpi=100)
        ax = fig.add_subplot(1, 3, 1)
        ax.plot(it, self.loss_gd, color='blue')
        ax.set_title('Gradient Decent')
        ax = fig.add_subplot(1, 3, 2)
        ax.plot(it, self.loss_mom, color='red')
        ax.set_title('Momentum')
        ax = fig.add_subplot(1, 3, 3)
        ax.plot(it, self.loss_nest, color='green')
        ax.set_title('Nesterov')
        plt.tight_layout()
        plt.show()
    def func(self, a, b):
        return polyval(self.x, [a, b])
    
    def MSEloss(self, a, b):
        return np.mean((self.y - self.func(a, b))**2)
    
    def gradients(self, a, b, e=1e-9):
        da = (self.MSEloss(a+e, b) - self.MSEloss(a-e, b))/(2*e)
        db = (self.MSEloss(a, b+e) - self.MSEloss(a, b-e))/(2*e)
        return da, db
    
    def descent(self, a, b, iterations=100, learning_rate=0.05):
        loss = []
        da, db = 0, 0
        for i in range(iterations):
            da, db = self.gradients(a, b)
            a -= learning_rate*da
            b -= learning_rate*db
            loss.append(self.MSEloss(a, b))
        return a, b, loss
    
    def momentum(self, a, b, iterations=100, gamma=0.9, eta=0.05):
        loss = []
        da, db, va, vb = 0, 0, 0, 0
        for i in range(iterations):
            da, db = self.gradients(a, b)
            va *= gamma
            vb *= gamma
            va += eta*da
            vb += eta*db
            a -= va
            b -= vb
            loss.append(self.MSEloss(a, b))
        return a, b, loss

    def nesterov(self, a, b, iterations=100, gamma=0.9, eta=0.05):
        loss = []
        da, db, va, vb = 0, 0, 0, 0
        for i in range(iterations):
            da, db = self.gradients(a - gamma*va, b - gamma*vb)
            va *= gamma
            vb *= gamma
            va += eta*da
            vb += eta*db
            a -= va
            b -= vb
            loss.append(self.MSEloss(a, b))
        return a, b, loss

class LinearRegressionModel(torch.nn.Module): 
    def __init__(self): 
        super(LinearRegressionModel, self).__init__() 
        self.linear = torch.nn.Linear(1, 1)  # One in and one out 
  
    def forward(self, x): 
        y_pred = self.linear(x) 
        return y_pred 

class otherOptimizationAlgos():
    def __init__(self, data):
        self.x = data[:,0].reshape(100,1)
        self.y = data[:,1].reshape(100,1)
        print(self.x.shape)
        our_model = LinearRegressionModel()        
        criterion = torch.nn.MSELoss(reduction = 'mean')
        #criterion = torch.nn.MSELoss(size_average = True) 
        optimizers = []
        optimizers.append(torch.optim.SGD(our_model.parameters(), lr = 0.01))
        optimizers.append(torch.optim.Adam(our_model.parameters(), lr = 0.01))
        optimizers.append(torch.optim.Adagrad(our_model.parameters(), lr = 0.01)) 
        optimizers.append(torch.optim.RMSprop(our_model.parameters(), lr = 0.01, alpha=0.9))
        epoch_loss = []
        for i in range(4):
            epoch_loss.append([])
        for i, optimizer in enumerate(optimizers):
            for epoch in range(100):
                y_pred = our_model(torch.autograd.Variable(torch.Tensor(self.x)))
                loss = criterion(y_pred, torch.autograd.Variable(torch.Tensor(self.y)))
                epoch_loss[i].append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #print("Epoch {} : Loss {}".format(epoch, loss.item()))
        it = np.arange(1, 101, 1)
        fig = plt.figure(figsize=(8, 2), dpi=100)
        ax = fig.add_subplot(1, 4, 1)
        ax.plot(it, epoch_loss[0], color='pink')
        ax.set_title('SGD')
        ax = fig.add_subplot(1, 4, 2)
        ax.plot(it, epoch_loss[1], color='magenta')
        ax.set_title('Adam')
        ax = fig.add_subplot(1, 4, 3)
        ax.plot(it, epoch_loss[2], color='gray')
        ax.set_title('Adagrad')
        ax = fig.add_subplot(1, 4, 4)
        ax.plot(it, epoch_loss[3], color='black')
        ax.set_title('RMSprop')
        plt.tight_layout()
        plt.show()
            
if __name__=='__main__':
    np.random.seed(2)
    data = np.loadtxt('data_ex1.csv', delimiter=',')
    ex3abc = gradient_descent(data)
    ex3d = otherOptimizationAlgos(data)

        