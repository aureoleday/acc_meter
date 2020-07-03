#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 08:48:33 2020

@author: aureoleday
"""

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from math import sin,pi

def logstic(k,x):
    
#    x1=0.6*x*(1-x)
    x1=k*sin(pi*x)
#    x1=λ*sin(pi*x)
    return x1

def gg(k,x):
    x1=1-k*x**2
    return x1

plt.figure(figsize=(20, 12))
for k in np.arange(0.5,2,0.01):
    print(k)
#for k=0.5:0.01:2;
    x=[0.3]
    for i in range(300):
#    for i=1:300
        temp=x[i]
        p=logstic(k,temp)
#        p=gg(k,temp)
        x.append(p)
    for t in range(150,300):
        plt.plot(k,x[t],'.b') 
