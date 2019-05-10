# -*- coding: utf-8 -*-
"""
Created on Wed May  8 08:54:44 2019

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt 


x = np.arange(0,1000)
xh = np.sin(x)

fig = plt.figure()
ax = fig.add_subplot(311)

ax.bar(x,xh)

plt.show()
#print(x)