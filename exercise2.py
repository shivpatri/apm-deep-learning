#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 18:52:00 2019

@author: spatri

Objective: Curse of Dimensionality
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from PIL import Image

class curse_of_dimensionality():
    def __init__(self, max_dim):
        np.random.seed(0)
        self.dist = []
        for i in range(max_dim):
            self.dist.append(self.avg_distance(i+1,10000))
        k = np.arange(1, max_dim+1, 1)
        plt.plot(k, self.dist, color='blue', label= r'Average Distance')
        plt.xlabel('Dimensions')
        plt.ylabel('Average Distance')
        plt.title('***************** Curse of Dimensionality ****************')
        plt.show()
        
    def avg_distance(self, dim, no_samples):
        dist = 0
        for i in range(no_samples):
            x = np.random.rand(dim, 1)
            y = np.random.rand(dim, 1)
            dist = dist + distance.euclidean(x, y)/no_samples
        return dist

class similar_yet_not_so_similar():
    def __init__(self, path):
        print('*************** Similar yet not so similar ****************')
        img = Image.open(path)
        img2 = self.create_similar_image(img)
        self.image_distance(img, img2)    
    
    def create_similar_image(self, img):
        img2 = np.copy(img)
        img2 = np.float32(img2)
        # Generating similar image using powerlaw transformation.
        img2 = (255*((img2/255.0)**(1.3))).astype('uint8')
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(img)
        ax.set_title('Original Image')
        ax.axis('off')
        ax = fig.add_subplot(1, 2, 2)
        ax.set_title('Transformed Image')
        ax.imshow(img2)
        ax.axis('off')
        plt.show()
        return img2
    
    def image_distance(self, img, img2):
        img = np.float32(img).flatten()
        img2 = np.float32(img2).flatten()
        print('The distance between the two images is: ' + str(distance.euclidean(img, img2)))
        

if __name__=='__main__':
    ex2a_b = curse_of_dimensionality(20)
    ex2c = similar_yet_not_so_similar(path='monalisa.jpg')