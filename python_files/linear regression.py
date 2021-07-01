#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 09:50:05 2021

@author: aureoleday
"""

import numpy as np

def calc_lr(target,samples):
    a = []
    if samples.size == samples.shape[0]:
        X = np.c_[np.ones(samples.shape[0]),samples] 
        a = (np.matrix(np.dot(X.T,X)).I).dot(X.T).dot(target)
    else:
        for col in samples.T:
            X = np.c_[np.ones(samples.shape[0]),col]
            w = (np.matrix(np.dot(X.T,X)).I).dot(X.T).dot(target)
            
            if a == []:   
                a = w          
            else:
                a = np.r_[a,w]
    return a.T

data = np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=0)
y = data[:,0]   
x = data[:,1:]

w = calc_lr(y,x)

np.savetxt("result_w.csv", w, delimiter=",",fmt='%10.5f')

print(w)

