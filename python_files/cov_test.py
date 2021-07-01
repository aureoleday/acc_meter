#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:47:02 2021

@author: aureoleday
"""

import numpy as np
import matplotlib.pyplot as plt

N = 500
theta = np.pi/6
k1 = 4
k2 = 1

S = np.array([[k1,0],[0,k2]]).T
R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]).T
x= np.random.randn(N)
y= np.random.randn(N)
p = np.array(list(zip(x,y)))
p = S.dot(p.T)
p = R.dot(p)
# A = np.dot(R,S)
# q = np.dot(A,p.transpose())

plt.axis('equal')
plt.scatter(p[0,:],p[1,:])
print(np.cov(p[0,:],p[1,:]))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# # bx = fig.add_subplot(212)
# ax.scatter(q[0,:],q[1,:])
# # bx.scatter(q[:,0],q[:,1])
# print(np.cov(q[0,:],q[1,:]))