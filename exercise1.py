#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 17:53:07 2019

@author: spatri

Objective: Overfitting
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval, polyfit
from sklearn.model_selection import train_test_split

class exercise1():
    # Segregating data to x and y.
    def __init__(self, data): 
        print('*********************** Data Scatter plot *****************************************')
        self.x, self.y = data[:,0], data[:,1]
        
    # Performing a polynomial fit and displaying the fit.
    def polynomial_regression_plt(self, max_k):
        min_x, max_x = min(self.x), max(self.x)
        min_y, max_y = min(self.x), max(self.y)
        xInt = np.arange(min_x, max_x, .01)
        self.train_losses = []
        max_k += 1
        print('******************* Univeriate Polynomial Fitting **********************************')
        fig = plt.figure(figsize=(7, 14), dpi=100)
        self.coeff = []
        # Iteratively doing a polynomial fit for k - 0 to 12.
        for k in range(max_k):
            self.coeff.append(polyfit(self.x, self.y, k))
            self.train_losses.append(np.mean((self.y - polyval(self.x, self.coeff[k]))**2))
            ax = fig.add_subplot(int(max_k/3)+1, 3, k+1)
            ax.plot(xInt, polyval(xInt, self.coeff[k]), label= r'P(x)', color='green')
            ax.scatter(self.x, self.y, s=5, label= r'data $\{x_i,y_i\}_{i=1..N}$',color='red')
            ax.set_title('k = ' + str(k))
            ax.axis([min_x, max_x, min_y, max_y])
            ax.grid()
            plt.tight_layout()
        plt.show()
    
    
    # Loss function plot.
    def loss_plot(self):
        self.train_losses.pop(0)
        max_k = len(self.train_losses)
        k = np.arange(1, max_k+1, 1)
        print('************************* Loss as the degree of the hypothesis increases ************************')
        plt.plot(k, self.train_losses, color='blue')
        plt.xlabel('Degree')
        plt.ylabel('Loss')
        plt.title('Loss vs Degree of hypothesis')
        plt.grid()
        plt.legend()
        plt.show()
    
    # Displaying raw data scatterplot.    
    def display_data(self):
        plt.figure(1)
        plt.plot(self.x,self.y,'o', color='blue')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.show()
        
    # Displaying train loss and test loss.
    def train_test_losses(self, test_data, max_k):
        test_x, test_y = test_data[:,0], test_data[:,1]
        self.test_losses = []
        max_k += 1
        for k in range(max_k):
            self.test_losses.append(np.mean((test_y - polyval(test_x, self.coeff[k]))**2))
        k = np.arange(2, max_k+1, 1)
        self.train_losses.pop(0)
        self.test_losses.pop(0)
        print('***************************** A plot emphasising overfitting **********************************')
        plt.plot(k, self.train_losses, color='blue', label= r'Train Loss')
        plt.plot(k, self.test_losses, color='green', label= r'Test Loss')
        plt.xlabel('Degree')
        plt.ylabel('Loss')
        plt.title('Loss vs Degree of hypothesis')
        plt.grid()
        plt.legend()
        plt.show()    
        
# Main Block.
if __name__=='__main__':
    # Pseudo random generator.
    np.random.seed(22)
    data = np.loadtxt('data_ex1.csv', delimiter=',')
    # Solution to 1A
    ex1a = exercise1(data)
    ex1a.display_data()
    ex1a.polynomial_regression_plt(12)
    ex1a.loss_plot()
    # Solution to 1B
    train_data, test_data = train_test_split(data,test_size=0.2)
    print('************************* Training only on 80% data **********************************************')
    ex1b_train_test = exercise1(train_data)
    ex1b_train_test.polynomial_regression_plt(12)
    ex1b_train_test.train_test_losses(test_data, 12)
    """
    Solution to 1c
    We can see that k = 9 has the least loss value. Although, I would think the original polynomial
    has the degree k = 3, because after k = 3, the algorithm couldn't drastically increase the accuracy.
    """