#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:29:46 2021

@author: aureoleday
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(0,1,500)
# x = 0.01*np.sin(2*np.pi*0.01*np.arange(500))
y = np.random.normal(0,1,500)

# X = np.vstack((x,y))
X = np.c_[[x,y]]

# A  = np.array([[,],[]])
plt.scatter(X[0,:], X[1,:],color='green')
plt.axis('equal')

sx,sy = 1,10
S = np.array([[sx,0],[0,sy]])
# theta = 1.1071
theta = np.pi/4
R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
T = np.dot(R,S)

Y = np.dot(T,X)

plt.scatter(Y[0,:], Y[1,:],color='red')
plt.axis('equal')

C = np.cov(Y)
print("cov of C:",C)

eva,eve = np.linalg.eig(C)
Rr,Sr = eve,np.diag(np.sqrt(eva))
Tr = np.linalg.inv(np.dot(Rr,Sr))
Z = np.dot(Tr,Y)

# fig = plt.figure()
# ax = fi

plt.scatter(Z[0,:], Z[1,:],color='blue')
plt.axis('equal')
print("eva,eve: ",eva,eve)