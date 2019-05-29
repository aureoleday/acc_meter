#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:45:56 2018

@author: Administrator
"""

'private ring buffer implementation'

__author__ = 'aureole'

import numpy as np

class RingBuffer(object):
    def __init__(self, size, dim=1, padding=None):
        self.size = size
        self.dim = dim
        self.padding = size if padding is None else padding
        self.buffer = np.zeros((self.size+self.padding, dim))
        self.counter = 0
        self.flag = 0

    def append(self, data):
        """this is an O(n) operation"""
        np_data = np.array(data)              
        np_data = np_data if (len(np_data.shape)>1) else np_data.reshape(1,self.dim)
        np_data = np_data[-self.padding:]
        n = np_data.shape[0]
        if self.remaining < n: self.compact()
        self.buffer[self.counter+self.size:][:n] = np_data
        self.counter += n

    def reset_flag(self):
        self.flag = 0
    @property
    def remaining(self):
        return self.padding-self.counter
    @property
    def view(self):
        """this is always an O(1) operation"""
        return np.array(self.buffer[self.counter:][:self.size])
    def compact(self):
        """
        note: only when this function is called, is an O(size) performance hit incurred,
        and this cost is amortized over the whole padding space
        """
        print('compacting')
        self.buffer[:self.size] = self.view
        self.counter = 0
        self.flag = 1

if __name__ == '__main__':
    rb = RingBuffer(10,3)
    for i in range(4):
        rb.append([1,2,3])
        print(rb.view)
    for i in range(20):
        rb.append((np.arange(6)*i).reshape(2,3))
        print(rb.view)  #test overflow


