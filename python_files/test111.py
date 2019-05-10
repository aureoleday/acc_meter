# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:52:34 2019

@author: scott
"""
import numpy as np


class my_mav(object):
    def __init__(self,row,col):
        self.mav_buf = np.zeros((int(row),int(col)))
        self.row_ind = 0
        self.cnt = 0
        self.row_max = row
        
    def insert(self,din):
        self.mav_buf[self.row_ind] = din
        self.row_ind += 1
        if(self.row_ind >= self.row_max):
            self.row_ind = 0
            
        if(self.cnt < self.row_max):
            self.cnt += 1
        else:
            self.cnt = self.cnt
            
        return self.mav_buf.sum(axis=0)/self.cnt

mav_inst = my_mav(8,12)

tt = mav_inst.insert(np.ones(12))
print(tt)