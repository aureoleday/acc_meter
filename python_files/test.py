#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:48:23 2021

@author: aureoleday
"""

import numpy as np
import matplotlib.pyplot as plt 

# fs = 2000
# f0 = 100
# N = 8

# a = np.sin(2*np.pi*f0*np.arange(fs*N/f0)/fs)
# b = np.cos(2*np.pi*f0*np.arange(fs*N/f0)/fs)

# plt.plot(a)
# plt.plot(b)

# class Ghost:
#     def __init__(self,hltv):
#         self.hltv = hltv
#         self.min = 0
#         self.max = 0
    
#     def fractal(self,ohlcv):
#         hl = ohlcv[:,2:4]
#         hl_reg = np.zeros(2)
#         direct = 0
#         step = 0
#         for x in hl:
#             if di
        

class Acount:
    def __init__(self,cdict):
        self.cdict = cdict
        
    def uopen(self,price,bdir):
        self.cdict['dir'] = bdir
        
        amount = self.cdict['bal']*self.cdict['scale_up'][0]
        cnt = amount/price
        
        self.cdict['bal'] -= amount
        
        self.cdict['pos'] = cnt
        
        self.cdict['fsm'] = 1
        return 0
    
    def uscale(self,price):
        if self.cdict['fsm'] < 3:
            amount = self.cdict['bal']*self.cdict['scale_up'][self.cdict['fsm']]
        else:
            return -1
        cnt = amount/price
        
        self.cdict['bal'] -= amount
        
        self.cdict['pos'] += cnt
        
        self.cdict['fsm'] += 1
        return 0
    
    def uclose(self,price):
        self.cdict['dir'] = 0
        self.cdict['bal'] += self.cdict['pos']*price
        self.cdict['pos'] = 0
        self.cdict['fsm'] = 0
        return 0
    
    def ufollow(self,ohlcv):
        return 0
    
    def statics(self):
        print("balance:%f,pos:%f\n" %(self.cdict['bal'],self.cdict['pos']))

config = dict({

    'bal':  float(10000),
    'pos':  float(0),
    'fsm'   :   1,
    'timeout':  0,
    'dir'   :   0,

    'hltv':{
        'high'  :   list([0,0,0]),
        'low'   :   list([0,0,0]),
        'timeout':  list([0,0,0]),
        'vol'   :   list([0,0,0]),
        },
    'scale_up'  :   [0.5,0.67,1],
    })


d1 = Acount(config)
d1.open(61)
# 