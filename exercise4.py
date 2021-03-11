#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:56:46 2019

@author: spatri
"""

"""
Created on Tue Jan 29 22:28:55 2019
@author: spatri
"""

from torchvision import datasets
import torchvision.transforms as transforms  # transform PIL image to tensor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class LinearClassifier():
    def __init__(self):
        self.MNIST_training_set_tensor = datasets.FashionMNIST('data_fashionMNIST', train=True, download=True,
                                                  transform=transforms.ToTensor())
        self.MNIST_test_set_tensor = datasets.FashionMNIST('data_fashionMNIST', train=False, download=True,
                                              transform=transforms.ToTensor())
        self.P = dict()
        self.P['numEpoch'] = 2
        self.P['learning_rate'] = .0001
        self.P['batchSize'] = 4
        self.myLoader_train = DataLoader(self.MNIST_training_set_tensor, shuffle=True, batch_size=self.P['batchSize'])
        self.myLoader_test = DataLoader(self.MNIST_test_set_tensor, shuffle=False, batch_size=self.P['batchSize'])
        self.myModel = nn.Linear(28 * 28, 10)
        self.myLoss = nn.CrossEntropyLoss()
        self.label_fashion = dict([(0,'T-shirt'),(1,'trouser'),(2,'pullover'),(3,'dress'),(4,'coat'),
                          (5,'sandal'),(6,'shirt'),(7,'sneaker'),(8,'bag'),(9,'boot')])
        self.training()
        
    def training(self):
        optimizer = torch.optim.Adam(self.myModel.parameters(), lr=self.P['learning_rate'])
        loss_func = []
        for epoch in range(self.P['numEpoch']):
            print('-- epoch ' + str(epoch))
            running_loss = 0.0
            miniBatch = 0
            for X, y in self.myLoader_train:
                optimizer.zero_grad()
                N, _, nX, nY = X.size()
                score = self.myModel(X.view(N, nX * nY))
                loss = self.myLoss(score, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach().numpy()
                miniBatch += 1
            loss_func.append(running_loss/ miniBatch)
            print(' average loss: ' + str(running_loss / miniBatch))
        for k in range(10):
            w = self.myModel.state_dict()['weight']
            plt.clf()
            plt.imshow(w[k].view(28, 28), vmin=-.5, vmax=.5, cmap='seismic')
            plt.title('After training, template for ' + str(self.label_fashion[k]))
            plt.colorbar(extend="both")
            plt.show()
        it = np.arange(1, self.P['numEpoch']+1, 1)
        plt.plot(it, loss_func, color='black')
        plt.title('Loss vs epochs')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
if __name__=='__main__':
    lc = LinearClassifier()