# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:28:42 2019

@author: Administrator
"""

import numpy as np

def upsample(din,sps,mode=0):
    if mode==0:
        return np.kron(din,np.append(1,np.zeros(sps-1)))
    else:
        return np.kron(din,np.ones(sps))

class dll:
    def __init__(self,sps=4):
        early
    
    
if __name__ == "__main__":

    ta = np.arange(10)
    
    print(upsample(ta,4))
    
    
