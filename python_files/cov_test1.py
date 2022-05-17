#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 09:38:45 2021

@author: aureoleday
"""

import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)


# Normal distributed x and y vector with mean 0 and standard deviation 1
x = np.random.normal(0, 1, 500)
y = np.random.normal(0, 1, 500)
# z = np.random.normal(0, 1, 500)
X = np.vstack((x, y)).T

X = X - np.mean(X,0)
sx,sy = 0.7, 2.4
scale = np.array([[sx,0],[0,sy]])
theta = 2*np.pi*0.3
R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

Y = X.dot(scale.dot(R))

eve,eva = np.linalg.eig(np.cov(Y.T))

for e,v in zip(eve,eva.T):
    print(e,v)
    plt.plot([0,3*np.sqrt(e)*v[0]],[0,3*np.sqrt(e)*v[1]],'k-',lw=2)



plt.scatter(Y[:, 0], Y[:, 1])
plt.title('Generated Data')
plt.axis('equal');